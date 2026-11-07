"""MCP server for KVM control — thin client that delegates to KVM server.

All hardware operations (serial, capture) are delegated to the KVM server
via TCP.  OCR is run locally using frames fetched from the KVM server.
"""

import asyncio
import base64
import contextlib
import datetime
import io
import json
import logging
import os
import time
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import ImageContent, TextContent, Tool
from PIL import Image
from serial_hid_kvm.client import KvmClient, KvmClientError
from serial_hid_kvm.hid_keycodes import validate_chars

from . import actionlog, colour, frames
from .config import config
from .ocr import TerminalOCR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
_client: KvmClient | None = None
_ocr: TerminalOCR | None = None
_log: "actionlog.ActionLog | None" = None


def get_client() -> KvmClient:
    global _client
    if _client is None:
        _client = KvmClient(config.kvm_host, config.kvm_port)
        _client.connect()
        logger.info("Connected to KVM server")
    return _client


def get_ocr() -> TerminalOCR:
    global _ocr
    if _ocr is None:
        _ocr = TerminalOCR(config.tesseract_cmd)
    return _ocr


def get_log() -> actionlog.ActionLog:
    global _log
    if _log is None:
        _log = actionlog.ActionLog(actionlog.resolve_path(), actor=actionlog.AGENT)
    return _log


# Tools that put input into the target. These are the ones that must be logged
# before they act, and that carry an action class.
INPUT_TOOLS = {
    "type_text",
    "send_key",
    "send_key_sequence",
    "mouse_move",
    "mouse_click",
    "mouse_drag",
    "mouse_scroll",
    "execute_and_read",
}

ACTION_CLASS_PROP = {
    "type": "string",
    "enum": list(actionlog.CLASSES),
    "description": (
        "Action class recorded in the engagement action log: RO read-only, "
        "LIC low-impact change, SC state-changing. Defaults to RO. Set it "
        "honestly; where an action could be either, use the higher class."
    ),
}


@contextlib.contextmanager
def _acting(action: str, cls: str = "RO", context: str = ""):
    """Log an input action, then perform it.

    The row goes down BEFORE the keystrokes, so a crash mid-send can never
    leave input that was delivered but unrecorded. The cost is the opposite
    error, a row that says "sent" when transmission failed, so a failure
    appends a correction naming the error. Delivery after a serial fault is
    recorded as uncertain rather than as not having happened: the daemon
    validates a whole string before sending any key, but a fault partway
    through can still leave characters delivered.

    The frame captured just before delivery becomes the row's evidence, so an
    input row answers "what was on screen when this was typed" without relying
    on a nearby capture row having been taken. That is the question asked of an
    input row afterwards: which window had focus, what prompt was showing,
    whether a field was waiting for a password. The operator console already
    records its sends this way, so this makes an agent row and an operator row
    mean the same thing in a merged timeline.

    Evidence is best effort; the log is not. An input that cannot be logged is
    refused, while an input whose pre-send frame cannot be captured still goes,
    because the ability to drive a target must not depend on the capture card.
    A target that is asleep or unplugged has no picture, and reaching it is
    exactly when input is needed.
    """
    log = get_log()
    log.require()  # refuses here, before anything reaches the target
    evidence, note = _presend_evidence()
    log.append(
        action=action,
        result="sent through the HID bridge" + note,
        evidence=evidence,
        cls=cls,
        context=context,
    )
    try:
        yield
    except Exception as exc:
        try:
            log.correction(
                of=action,
                problem=f"transmission FAILED ({exc}); delivery uncertain, verify on screen",
                cls=cls,
                context=context,
            )
        except actionlog.LogUnavailable:
            pass
        raise


def _frame_path(log_dir: str, suffix: str) -> str:
    """A free filename for a frame, within the second.

    The stamp has one-second resolution, so two frames saved in the same second
    under the same suffix used to land on one name and the first was lost. An
    evidence path that silently resolves to another row's frame is worse than
    no path at all, and a burst of key presses reaches that rate easily.
    """
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tag = f"_{suffix}" if suffix else ""
    path = os.path.join(log_dir, f"{stamp}{tag}.jpg")
    n = 2
    while os.path.exists(path):
        path = os.path.join(log_dir, f"{stamp}{tag}-{n}.jpg")
        n += 1
    return path


def _write_frame(image: Image.Image, suffix: str = "") -> str | None:
    """Save a frame to the capture directory. Writes no log row."""
    log_dir = config.capture_log_dir
    if log_dir is None:
        return None
    try:
        os.makedirs(log_dir, exist_ok=True)
        filepath = _frame_path(log_dir, suffix)
        image.convert("RGB").save(filepath, format="JPEG", quality=85)
        logger.info(f"Capture log saved: {filepath}")
        return filepath
    except Exception as e:
        logger.warning(f"Failed to save capture log: {e}")
        return None


def _presend_evidence() -> tuple[str, str]:
    """The screen as it stood immediately before input is delivered.

    Returns the evidence path and a note for the result cell, either of which
    may be empty. Nothing here raises: see _acting on why input must not be
    made to depend on the capture device.
    """
    if not config.presend_evidence:
        return "", ""
    try:
        image = _capture_image()
    except Exception as exc:
        logger.warning(f"no pre-send frame for the action log: {exc}")
        return "", "; no pre-send frame, capture unavailable"

    stdev = frames.frame_stdev(image)
    path = _write_frame(image, "presend") or ""
    if frames.is_blank(stdev):
        # Worth saying in the row. Input sent against a frame with no picture
        # may have gone nowhere visible, or into a window nobody chose, and
        # the saved frame on its own does not show that this was known.
        return path, f"; screen carried no picture (spread {stdev:.2f})"
    return path, ""


def _save_capture_log(image: Image.Image, suffix: str = "") -> str | None:
    """Save a capture image to the log directory and record it."""
    filepath = _write_frame(image, suffix)
    if filepath is None:
        return None
    try:
        note = colour.describe() if config.colour_correct else "raw frame"
        get_log().append(
            action=f"capture_screen ({suffix or 'capture'})",
            result=f"{image.width}x{image.height}; {note}",
            evidence=filepath,
            cls="RO",
            context="",
        )
    except actionlog.LogUnavailable:
        # A capture that cannot be logged is worth less but is not
        # dangerous. An unlogged INPUT is, and _acting refuses that.
        logger.warning("capture not recorded in the action log")
    return filepath


# Longest a command is waited on before the screen is read anyway. Long
# enough for the slow end of a build review (system_profiler, a large
# find) without hanging a session on something that never settles.
DEFAULT_SETTLE_TIMEOUT_S = 20.0
SETTLE_POLL_S = 0.15
SETTLE_QUIET_FRAMES = 3


async def _wait_for_settle(
    min_wait_s: float,
    timeout_s: float = DEFAULT_SETTLE_TIMEOUT_S,
) -> tuple[Image.Image, float, bool]:
    """Wait until the screen stops changing, then return the frame.

    A fixed wait is wrong in both directions: too short and the screen is read
    while the command is still producing output, which is the dangerous one
    because the result looks like a finished answer; too long and every quick
    command costs the slowest command's wait.

    The capture path itself is not the reason to wait. Measured on this card,
    an instant command's output was in the first frame fetched after Enter,
    within 200ms. What needs waiting for is the command.

    Returns the frame, how long was waited, and whether it settled. A read
    that timed out is reported as such rather than presented as a result: the
    screen may be mid-output, and saying so is the difference between a
    partial answer and a wrong one.
    """
    deadline = time.monotonic() + max(timeout_s, min_wait_s)
    floor = time.monotonic() + max(0.0, min_wait_s)
    started = time.monotonic()

    image = _capture_image()
    previous = frames.signature(image)
    quiet = 0
    while time.monotonic() < deadline:
        await asyncio.sleep(SETTLE_POLL_S)
        image = _capture_image()
        current = frames.signature(image)
        quiet = quiet + 1 if frames.is_quiet(frames.difference(previous, current)) else 0
        previous = current
        if quiet >= SETTLE_QUIET_FRAMES and time.monotonic() >= floor:
            return image, time.monotonic() - started, True
    return image, time.monotonic() - started, False


def _capture_image(quality: int = 85) -> Image.Image:
    """Fetch a frame from KVM server and return as PIL Image.

    The capture device does not return a neutral white (see colour.py), so the
    correction is applied here, at the single point every tool fetches a frame
    through. capture_screen, get_screen_text and execute_and_read therefore all
    see the same corrected picture, and so does anything written to the capture
    log.

    Set SHKVM_COLOUR_CORRECT=0 to get raw frames.
    """
    jpeg_bytes, w, h = get_client().capture_frame_jpeg(quality)
    image = Image.open(io.BytesIO(jpeg_bytes))
    image.load()
    if config.colour_correct:
        image = colour.correct(image)
    return image


# Create MCP server
app = Server("mcp-serial-hid-kvm")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools.

    Every input tool gains an `action_class` property here rather than in each
    schema by hand, so the set cannot drift from INPUT_TOOLS.
    """
    tools = _tool_definitions()
    for tool in tools:
        if tool.name in INPUT_TOOLS:
            tool.inputSchema.setdefault("properties", {})[
                "action_class"
            ] = dict(ACTION_CLASS_PROP)
    return tools


def _tool_definitions() -> list[Tool]:
    return [
        Tool(
            name="type_text",
            description=(
                'Type a string as keyboard input on the target PC. Supports inline tags: {enter}, {tab}, {ctrl+c}, {shift+0x87}, etc. '
                '{0xNN} sends any HID keycode by hex value (0x00-0xFF) for keys without a named tag, e.g. {0x87} = JIS ろ key (International1).\n'
                '\n'
                '**Whitelist-based tag parsing:** Only recognized special key names inside {braces} are interpreted as tags. '
                'Unknown {content} (e.g. {print $1}) is passed through as literal text including the braces. '
                'This means code with curly braces (awk, Python, shell) can be sent without escaping in most cases.\n'
                '\n'
                '**Escaping:** Use {{ and }} to force literal braces when they collide with a recognized tag name '
                '(e.g. {{enter}} to type the literal text "{enter}").\n'
                '\n'
                '**Raw mode (raw=true):** Disables ALL tag interpretation. '
                'Actual line breaks in the input (LF, CRLF, CR) are sent as Enter key presses. '
                'In JSON, \\n is decoded into an actual line break, so it becomes Enter. '
                'To type a literal backslash + n, use \\\\n in JSON.\n'
                '\n'
                'Examples:\n'
                '  "ls -la{enter}"                     → types "ls -la" then presses Enter\n'
                '  "awk \'{print $1}\' file.txt{enter}"  → types the awk command then Enter (braces preserved)\n'
                '  "echo {{enter}}"                    → types "echo {enter}" (escaped to avoid tag)\n'
                '  raw=true: "ls -la\\necho hi\\n"       → types "ls -la", Enter, "echo hi", Enter\n'
                '\n'
                '**Supported characters:** ASCII printable (space through ~), tab, and newline only. '
                'Unicode, CJK, accented characters, and control characters cause an error. '
                'For unsupported characters or binary data, use base64 encoding as a workaround: '
                'encode on the host and type a decode command (e.g. `echo <b64> | base64 -d`) on the target.'
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text with optional {tag} sequences. Tags: {enter}/{return}, {space}, {tab}, {backspace}, {delete}, {insert}, {escape}/{esc}, {up}, {down}, {left}, {right}, {home}, {end}, {pageup}, {pagedown}, {f1}-{f12}, {capslock}, {numlock}, {scrolllock}, {printscreen}, {pause}, {0xNN} for raw HID keycodes (0x00-0xFF). Modifiers: ctrl/lctrl/rctrl, shift/lshift/rshift, alt/lalt/ralt, win/lwin/rwin/gui/super/meta — combine with +: {ctrl+c}, {alt+f4}, {ctrl+shift+del}, {shift+0x87}. Only recognized key names are treated as tags; other braces pass through literally.",
                    },
                    "raw": {
                        "type": "boolean",
                        "description": "If true, disable all {tag} interpretation. Actual line breaks (LF, CRLF, CR) become Enter. In JSON, \\n is decoded into a line break and becomes Enter; \\\\n types literal backslash + n. Default: false.",
                    },
                    "char_delay_ms": {
                        "type": "integer",
                        "description": "Delay between characters in milliseconds (default: 20)",
                    },
                },
                "required": ["text"],
            },
        ),
        Tool(
            name="send_key",
            description="Send a single key press with optional modifier keys (e.g., Ctrl+C, Alt+F4).",
            inputSchema={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Key name: a-z, 0-9, enter, tab, escape, backspace, delete, up, down, left, right, home, end, pageup, pagedown, f1-f12, space, insert, printscreen",
                    },
                    "modifiers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Modifier keys: ctrl, shift, alt, win (gui/super/meta)",
                    },
                },
                "required": ["key"],
            },
        ),
        Tool(
            name="send_key_sequence",
            description="Send a sequence of key steps with optional per-step delays. Useful for complex keyboard operations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "key": {"type": "string", "description": "Key name"},
                                "modifiers": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Modifier keys",
                                },
                                "delay_ms": {
                                    "type": "integer",
                                    "description": "Delay after this step in ms (default: 100)",
                                },
                            },
                            "required": ["key"],
                        },
                        "description": "List of key steps to execute",
                    },
                    "default_delay_ms": {
                        "type": "integer",
                        "description": "Default delay between steps in ms (default: 100)",
                    },
                },
                "required": ["steps"],
            },
        ),
        Tool(
            name="mouse_move",
            description="Move the mouse cursor on the target PC.",
            inputSchema={
                "type": "object",
                "properties": {
                    "x": {
                        "type": "integer",
                        "description": "X coordinate (screen pixels for absolute, offset for relative)",
                    },
                    "y": {
                        "type": "integer",
                        "description": "Y coordinate (screen pixels for absolute, offset for relative)",
                    },
                    "relative": {
                        "type": "boolean",
                        "description": "If true, move relative to current position (default: false)",
                    },
                },
                "required": ["x", "y"],
            },
        ),
        Tool(
            name="mouse_click",
            description="Click a mouse button on the target PC, optionally at a specific position.",
            inputSchema={
                "type": "object",
                "properties": {
                    "button": {
                        "type": "string",
                        "enum": ["left", "right", "middle"],
                        "description": "Mouse button (default: left)",
                    },
                    "x": {
                        "type": "integer",
                        "description": "Optional X screen coordinate to click at",
                    },
                    "y": {
                        "type": "integer",
                        "description": "Optional Y screen coordinate to click at",
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="mouse_drag",
            description="Drag from one position to another (press button at start, move to end, release). Useful for drag-and-drop, selecting text, resizing windows, etc.",
            inputSchema={
                "type": "object",
                "properties": {
                    "start_x": {
                        "type": "integer",
                        "description": "Starting X screen coordinate",
                    },
                    "start_y": {
                        "type": "integer",
                        "description": "Starting Y screen coordinate",
                    },
                    "end_x": {
                        "type": "integer",
                        "description": "Ending X screen coordinate",
                    },
                    "end_y": {
                        "type": "integer",
                        "description": "Ending Y screen coordinate",
                    },
                    "button": {
                        "type": "string",
                        "enum": ["left", "right", "middle"],
                        "description": "Mouse button (default: left)",
                    },
                },
                "required": ["start_x", "start_y", "end_x", "end_y"],
            },
        ),
        Tool(
            name="mouse_scroll",
            description="Scroll the mouse wheel on the target PC.",
            inputSchema={
                "type": "object",
                "properties": {
                    "amount": {
                        "type": "integer",
                        "description": "Scroll amount: positive=up, negative=down (-127 to 127)",
                    },
                },
                "required": ["amount"],
            },
        ),
        Tool(
            name="capture_screen",
            description="Capture the target PC screen via HDMI capture device. Returns the image. Use sparingly as images consume many tokens.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="get_screen_text",
            description="Capture the target PC screen and extract text using OCR. Prefer this over capture_screen for text content.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="execute_and_read",
            description=(
                "Type a command, press Enter, wait for the screen to stop "
                "changing, then capture and OCR it. Waits for the command "
                "rather than for a fixed time, so a slow command is not read "
                "while it is still producing output; if it never settles the "
                "result says so instead of looking finished."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Command to type and execute",
                    },
                    "wait_seconds": {
                        "type": "number",
                        "description": (
                            "Minimum seconds to wait before the screen may be "
                            "considered settled (default: 1.0). Raise it for a "
                            "command that pauses before producing output."
                        ),
                    },
                    "timeout_seconds": {
                        "type": "number",
                        "description": (
                            "Longest to wait for the screen to stop changing "
                            "(default: 20). On timeout the output is returned "
                            "with a warning that it may be partial."
                        ),
                    },
                },
                "required": ["command"],
            },
        ),
        Tool(
            name="get_device_info",
            description="Show connection status and device information for the serial adapter and HDMI capture device.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="list_capture_devices",
            description="List all available video capture devices with their index and name. Use this to find the correct HDMI capture device.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="set_capture_resolution",
            description="Change the HDMI capture resolution. Common values: 1920x1080, 1280x720, 640x480. The actual resolution depends on what the capture device supports.",
            inputSchema={
                "type": "object",
                "properties": {
                    "width": {
                        "type": "integer",
                        "description": "Capture width in pixels (e.g. 1920)",
                    },
                    "height": {
                        "type": "integer",
                        "description": "Capture height in pixels (e.g. 1080)",
                    },
                },
                "required": ["width", "height"],
            },
        ),
        Tool(
            name="set_capture_device",
            description="Switch the active capture device by index or path. Use list_capture_devices first to see available options. Reopens the capture device.",
            inputSchema={
                "type": "object",
                "properties": {
                    "device": {
                        "type": "string",
                        "description": "Device index (e.g. '0', '1') or path (e.g. '/dev/video0')",
                    },
                },
                "required": ["device"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent | ImageContent]:
    """Handle tool calls."""
    try:
        client = get_client()
        # Every input tool is logged before it acts, and refuses if it cannot
        # be logged. The caller sets the class; RO is only the default.
        cls = arguments.get("action_class", "RO")

        if name == "type_text":
            text = arguments["text"]
            char_delay = arguments.get("char_delay_ms")
            raw = arguments.get("raw", False)
            validate_chars(text)
            with _acting(f"type_text: {text}", cls):
                client.type_text(text, char_delay, raw=raw)
            return [TextContent(type="text", text=f"Typed {len(text)} characters")]

        elif name == "send_key":
            key = arguments["key"]
            modifiers = arguments.get("modifiers", [])
            combo = "+".join(list(modifiers) + [key])
            with _acting(f"send_key: {combo}", cls):
                client.send_key(key, modifiers)
            mod_str = "+".join(modifiers) + "+" if modifiers else ""
            return [TextContent(type="text", text=f"Sent: {mod_str}{key}")]

        elif name == "send_key_sequence":
            steps = arguments["steps"]
            default_delay = arguments.get("default_delay_ms", 100)
            summary = ", ".join(
                "+".join(list(s.get("modifiers") or []) + [s.get("key", "?")])
                for s in steps
            )
            with _acting(f"send_key_sequence: {summary}", cls):
                client.send_key_sequence(steps, default_delay)
            return [TextContent(type="text", text=f"Sent {len(steps)} key steps")]

        elif name == "mouse_move":
            x = arguments["x"]
            y = arguments["y"]
            relative = arguments.get("relative", False)
            kind = "relative" if relative else "absolute"
            with _acting(f"mouse_move: {kind} ({x}, {y})", cls):
                client.mouse_move(x, y, relative)
            if relative:
                return [TextContent(type="text", text=f"Moved mouse by ({x}, {y})")]
            else:
                return [TextContent(type="text", text=f"Moved mouse to ({x}, {y})")]

        elif name == "mouse_click":
            button = arguments.get("button", "left")
            x = arguments.get("x")
            y = arguments.get("y")
            where = f" at ({x}, {y})" if x is not None and y is not None else ""
            with _acting(f"mouse_click: {button}{where}", cls):
                client.mouse_click(button, x, y)
            return [TextContent(type="text", text=f"Clicked {button}{where}")]

        elif name == "mouse_drag":
            start_x = arguments["start_x"]
            start_y = arguments["start_y"]
            end_x = arguments["end_x"]
            end_y = arguments["end_y"]
            button = arguments.get("button", "left")
            action = (
                f"mouse_drag: {button} ({start_x}, {start_y}) to ({end_x}, {end_y})"
            )
            with _acting(action, cls):
                client.mouse_down(button, start_x, start_y)
                await asyncio.sleep(0.05)
                client.mouse_move(end_x, end_y)
                await asyncio.sleep(0.05)
                client.mouse_up(button, end_x, end_y)
            return [TextContent(
                type="text",
                text=f"Dragged {button} from ({start_x}, {start_y}) to ({end_x}, {end_y})",
            )]

        elif name == "mouse_scroll":
            amount = arguments["amount"]
            with _acting(f"mouse_scroll: {amount}", cls):
                client.mouse_scroll(amount)
            direction = "up" if amount > 0 else "down"
            return [TextContent(type="text", text=f"Scrolled {direction} by {abs(amount)}")]

        elif name == "capture_screen":
            image = _capture_image()
            _save_capture_log(image, "capture")
            # Use JPEG to keep size under 20MB (base64 limit)
            quality = 85
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=quality)
            # If still too large, reduce quality then resize
            while buffer.tell() > 10_000_000 and quality > 20:
                quality -= 15
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=quality)
            if buffer.tell() > 10_000_000:
                image = image.resize((image.width // 2, image.height // 2))
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=60)
            b64_image = base64.standard_b64encode(buffer.getvalue()).decode("utf-8")
            return [ImageContent(
                type="image",
                data=b64_image,
                mimeType="image/jpeg",
            )]

        elif name == "get_screen_text":
            image = _capture_image()
            _save_capture_log(image, "ocr")
            text = get_ocr().extract_text(image)
            return [TextContent(type="text", text=text)]

        elif name == "execute_and_read":
            command = arguments["command"]
            wait_seconds = arguments.get("wait_seconds", 1.0)
            timeout_seconds = arguments.get("timeout_seconds", DEFAULT_SETTLE_TIMEOUT_S)

            # Raw mode: no tag interpretation for command text
            validate_chars(command)
            with _acting(f"execute_and_read: {command} then Enter", cls):
                client.type_text(command, raw=True)
                await asyncio.sleep(0.1)
                client.send_key("enter")

            image, waited, settled = await _wait_for_settle(wait_seconds, timeout_seconds)
            _save_capture_log(image, "exec")
            text = get_ocr().extract_text(image)
            if not settled:
                # Say it in the returned text, not only the log. A partial
                # screen read as a finished answer is the failure this is
                # here to prevent, and the caller is what needs warning.
                note = (
                    f"[WARNING: the screen was still changing after "
                    f"{waited:.1f}s, so this may be partial output. Capture "
                    f"again, or raise timeout_seconds.]" + chr(10)
                )
                text = note + text
            return [TextContent(type="text", text=text)]

        elif name == "get_device_info":
            info = client.get_device_info()
            return [TextContent(
                type="text",
                text=json.dumps(info, indent=2, ensure_ascii=False),
            )]

        elif name == "set_capture_resolution":
            width = arguments["width"]
            height = arguments["height"]
            result = client.set_capture_resolution(width, height)
            cap_info = result.get("info", {})
            return [TextContent(
                type="text",
                text=f"Resolution set: {cap_info.get('width')}x{cap_info.get('height')} (requested {width}x{height})",
            )]

        elif name == "list_capture_devices":
            result = client.list_capture_devices()
            devices = result.get("devices", [])
            if not devices:
                return [TextContent(type="text", text="No capture devices found.")]
            return [TextContent(
                type="text",
                text=json.dumps(devices, indent=2, ensure_ascii=False),
            )]

        elif name == "set_capture_device":
            device = arguments["device"]
            result = client.set_capture_device(device)
            cap_info = result.get("info", {})
            return [TextContent(
                type="text",
                text=f"Switched to device {device}: {cap_info.get('width')}x{cap_info.get('height')} ({cap_info.get('backend')})",
            )]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except KvmClientError as e:
        logger.error(f"KVM server error in tool {name}: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]
    except Exception as e:
        logger.exception(f"Error in tool {name}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def run():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


def main():
    """Entry point."""
    asyncio.run(run())


if __name__ == "__main__":
    main()



