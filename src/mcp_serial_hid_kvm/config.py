"""Configuration for the MCP server (thin client)."""

import os
import platform


def _default_capture_log_dir() -> str:
    """Return the platform-appropriate default directory for capture logs."""
    if platform.system() == "Windows":
        base = os.environ.get("LOCALAPPDATA", os.path.expanduser("~/AppData/Local"))
    else:
        base = os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
    return os.path.join(base, "mcp-serial-hid-kvm", "captures")


class Config:
    """Minimal configuration for the MCP thin-client server."""

    def __init__(self):
        # KVM server connection
        self.kvm_host: str = os.environ.get("SHKVM_API_HOST", "127.0.0.1")
        self.kvm_port: int = int(os.environ.get("SHKVM_API_PORT", "9329"))

        # Local OCR
        self.tesseract_cmd: str | None = os.environ.get("SHKVM_OCR_CMD")

        # Capture device colour correction (see colour.py). On by default: an
        # uncorrected frame misrepresents the target's screen, and a mixed
        # evidence set is worse than a consistently corrected one.
        self.colour_correct: bool = os.environ.get(
            "SHKVM_COLOUR_CORRECT", "1"
        ).strip().lower() not in ("0", "false", "no", "off")

        # Attach the frame captured just before an input action as that
        # action's evidence (see server._presend_evidence). On by default: an
        # input row that cannot show the screen it went into is the weaker
        # half of the record. Set SHKVM_PRESEND_EVIDENCE=0 to turn it off,
        # which costs one frame grab less per keystroke and an answer to
        # "what had focus".
        self.presend_evidence: bool = os.environ.get(
            "SHKVM_PRESEND_EVIDENCE", "1"
        ).strip().lower() not in ("0", "false", "no", "off")

        # Refuse keystrokes when the pre-send frame does not look like an
        # interface: a locked screen, a screensaver or a bare desktop. On by
        # default, because typing blind onto someone else's machine is the
        # failure this exists to prevent and it has happened twice. Pointer
        # calls are never guarded, so waking a sleeping target still works.
        self.focus_guard: bool = os.environ.get(
            "SHKVM_FOCUS_GUARD", "1"
        ).strip().lower() not in ("0", "false", "no", "off")

        # Capture log directory
        raw = os.environ.get("SHKVM_CAPTURE_LOG_DIR")
        if raw is None:
            self.capture_log_dir: str | None = _default_capture_log_dir()
        elif raw == "":
            self.capture_log_dir = None
        else:
            self.capture_log_dir = raw


config = Config()
