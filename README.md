# mcp-serial-hid-kvm

MCP (Model Context Protocol) server that gives AI agents full keyboard, mouse, and screen access to a physical PC. Thin client for [serial-hid-kvm](https://github.com/sunasaji/serial-hid-kvm) — all hardware control is delegated via TCP.

## How It Works

```
Claude / AI Agent
  ↕ MCP (stdio)
mcp-serial-hid-kvm        ← this package (thin client + OCR)
  ↕ TCP (localhost:9329)
serial-hid-kvm             ← standalone KVM server (owns hardware)
  ↕ USB Serial + HDMI
Target PC
```

The KVM server (`serial-hid-kvm`) runs as a persistent process owning the serial port and capture device. This MCP server connects to it as a TCP client. Multiple MCP instances (multiple Claude sessions) can share a single KVM server without device conflicts.

## Prerequisites

1. **Hardware**: CH9329+CH340 USB HID cable + USB HDMI capture device (see [serial-hid-kvm](https://github.com/sunasaji/serial-hid-kvm) for details)
2. **serial-hid-kvm** installed and running:
   ```bash
   pip install -e /path/to/serial-hid-kvm
   serial-hid-kvm --api              # with preview window
   serial-hid-kvm --api --headless   # or headless
   ```
3. **Tesseract OCR** (for `get_screen_text` / `execute_and_read`):
   - Linux: `sudo apt install tesseract-ocr`
   - Windows: https://github.com/tesseract-ocr/tesseract

## Installation

```bash
pip install -e .
```

This automatically installs `serial-hid-kvm` as a dependency.

## MCP Client Configuration

### Claude Desktop / Claude Code

```json
{
  "mcpServers": {
    "kvm": {
      "command": "mcp-serial-hid-kvm"
    }
  }
}
```

Custom KVM server address:

```json
{
  "mcpServers": {
    "kvm": {
      "command": "mcp-serial-hid-kvm",
      "env": {
        "SHKVM_API_HOST": "127.0.0.1",
        "SHKVM_API_PORT": "9329"
      }
    }
  }
}
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SHKVM_API_HOST` | `127.0.0.1` | KVM server address |
| `SHKVM_API_PORT` | `9329` | KVM server port |
| `SHKVM_OCR_CMD` | auto-detect | Path to tesseract executable |
| `SHKVM_CAPTURE_LOG_DIR` | platform default | Capture log directory (empty string to disable) |
| `SHKVM_ACTION_LOG` | platform default | Action log file (empty string to disable, which also refuses all input) |
| `SHKVM_COLOUR_CORRECT` | `1` | Apply the capture device's measured colour correction to every frame |
| `SHKVM_PRESEND_EVIDENCE` | `1` | Save the frame from just before each input action and cite it in the log |
| `SHKVM_FOCUS_GUARD` | `1` | Refuse keystrokes when no frame could be captured, or the frame carries no picture |

Hardware settings (`SHKVM_SERIAL_PORT`, `SHKVM_SCREEN_WIDTH`, etc.) are configured on the **KVM server side**, not here. If the target PC uses a non-US keyboard, set `--target-layout` (or `SHKVM_TARGET_LAYOUT`) on the KVM server so that `type_text` and `send_key` produce correct characters.

### Action log

Every tool that puts input into the target writes a row to the action log
*before* it acts, and refuses to act if it cannot write one. The order is
deliberate: a crash partway through a send can leave input that was delivered,
and an unrecorded keystroke on someone else's machine is the one outcome worth
designing against. The opposite error, a row saying "sent" when transmission
failed, is corrected by a following row rather than by deleting anything.

Each input row also cites the frame captured immediately before delivery, so
the row answers "what was on screen when this was typed" on its own: which
window had focus, what was showing, whether a field was waiting for a password.
Evidence is best effort while the log is not, because the ability to drive a
target must not depend on the capture card: a target that is asleep has no
picture, and that is exactly when input is needed. Where there was no picture
the row says so.

Set `SHKVM_ACTION_LOG=""` to turn logging off, which also disables every input
tool. There is no mode that sends input without recording it.

### Typing at a screen nobody can see

Keystrokes are refused when no frame could be captured, or when the frame
carries no picture: the display is asleep, or the capture card has no signal.
The pointer is exempt, because waking a sleeping target means moving the mouse
at exactly such a screen. `SHKVM_FOCUS_GUARD=0` turns it off.

That is the whole check. A lock screen showing a picture is not detected, and
that is deliberate: no cheap whole-frame measure separates one from an unlocked
desktop with a window open, and a check that tried blocked graphical work
instead.

## Available Tools

### Keyboard

| Tool | Description |
|------|-------------|
| `type_text` | Type text with inline tags: `ls -la{enter}`, `{ctrl+c}`, `{alt+f4}`. Whitelist-based: unknown `{content}` passes through literally. Raw mode (`raw=true`) disables tags; actual line breaks become Enter. `char_delay_ms`: delay between keystrokes in ms (default: 20). Only ASCII printable characters, tab, and line breaks are supported; unsupported characters (Unicode, CJK, etc.) cause an error — use base64 encoding as a workaround |
| `send_key` | Single key press with modifiers |
| `send_key_sequence` | Multiple key steps with per-step delays. `default_delay_ms`: delay between steps in ms (default: 100); each step can override with `delay_ms` |

### Mouse

| Tool | Description |
|------|-------------|
| `mouse_move` | Move cursor (absolute or relative) |
| `mouse_click` | Click at optional position |
| `mouse_drag` | Drag from one position to another (drag-and-drop, text selection, etc.) |
| `mouse_scroll` | Scroll wheel |

### Screen

| Tool | Description |
|------|-------------|
| `capture_screen` | Capture screen as image (high token cost) |
| `get_screen_text` | Capture + OCR to text (preferred for text content) |
| `execute_and_read` | Type command, Enter, wait, capture + OCR |

### Device Management

| Tool | Description |
|------|-------------|
| `get_device_info` | Serial port, capture device, config info |
| `list_capture_devices` | List available video devices |
| `set_capture_device` | Switch capture device |
| `set_capture_resolution` | Change capture resolution |

## Architecture

This package is intentionally minimal (~4 files):

```
mcp_serial_hid_kvm/
  server.py    MCP tool handlers → KvmClient TCP calls
  config.py    KVM host/port, tesseract, log settings
  ocr.py       Tesseract OCR (runs locally on fetched frames)
  __init__.py
```

All keyboard/mouse/capture logic lives in `serial-hid-kvm`. This package only translates MCP tool calls to TCP API calls and runs OCR locally.

### Why Separate?

- **No device conflicts** — multiple Claude sessions share one KVM server
- **Independent restarts** — restart the MCP server without losing the KVM connection
- **Standalone use** — `serial-hid-kvm` works without MCP (interactive preview, scripts, other AI frameworks)

## License

[MIT](LICENSE.txt)
