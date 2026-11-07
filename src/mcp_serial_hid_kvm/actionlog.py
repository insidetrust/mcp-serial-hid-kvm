"""Append-only action log, shared by the agent and the human operator.

Every action taken against a target gets one row here, whoever took it. The
agent writes through the MCP server; a human writes through the operator
console. One file, one ordered record.

Why this lives in the server rather than in the operator's tooling: for a long
time the agent's rows existed only because the model remembered to write them,
while the console refused to send at all when it could not log. That is the
wrong way round. The side that needs enforcement most had none, and the log
ended up with rows that had to be corrected by hand.

So the rule is the same for both: log, or refuse to act.

Row format, seven columns:

    | timestamp | actor | class | context | action | result | evidence |

The actor column was added on 07/11/2026. Rows written before that have six
columns and no actor; readers should treat a six-column row as legacy rather
than as malformed.
"""

from __future__ import annotations

import datetime
import re
import os
import platform
from pathlib import Path

ENV_VAR = "SHKVM_ACTION_LOG"

# Who took the action. Kept short because it is a column, not a sentence.
OPERATOR = "operator"
AGENT = "agent"
ACTORS = (OPERATOR, AGENT)

# Action classes from the engagement operating rules.
CLASSES = ("RO", "LIC", "SC")

CLASS_LABELS = {
    "RO": "RO - read-only",
    "LIC": "LIC - low-impact change",
    "SC": "SC - state-changing",
}

ROW_COLUMNS = 7

# "2026-11-07 10:31" as written here, or a bare "21:19" as some hand-written
# rows carry. Anything else in the first cell is not an action row.
_TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}\s+)?\d{1,2}:\d{2}(\s*-\s*\d{1,2}:\d{2})?$")


class LogUnavailable(RuntimeError):
    """The action log could not be written, so the caller must not act."""


def _default_path() -> str:
    """Somewhere to log when nobody has said where.

    Deliberately a real default rather than "no log": an unset environment
    variable should not silently buy an unlogged input path. Set the variable
    to an empty string to turn logging off on purpose, which mirrors how
    SHKVM_CAPTURE_LOG_DIR already behaves.
    """
    if platform.system() == "Windows":
        base = os.environ.get("LOCALAPPDATA", os.path.expanduser("~/AppData/Local"))
    else:
        base = os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
    return os.path.join(base, "mcp-serial-hid-kvm", "action-log.md")


def resolve_path(explicit: str | None = None) -> Path | None:
    """Where the log lives. None means logging has been turned off on purpose."""
    if explicit:
        return Path(explicit).expanduser()
    raw = os.environ.get(ENV_VAR)
    if raw is None:
        return Path(_default_path())
    if raw == "":
        return None
    return Path(raw).expanduser()


def redaction(char_count: int) -> str:
    """The stand-in written instead of a secret value.

    The operating rules forbid storing a password in any file, and this is a
    file. Recording that a secret was sent, how long it was, and when, keeps
    the trail complete without keeping the value.
    """
    return f"[redacted secret, {char_count} chars]"


class ActionLog:
    """Append-only writer. One row per action, whoever acted."""

    def __init__(self, path: Path | None, actor: str = OPERATOR):
        if actor not in ACTORS:
            raise ValueError(f"unknown actor {actor!r}, expected one of {ACTORS}")
        self._path = path
        self._actor = actor

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def actor(self) -> str:
        return self._actor

    @property
    def disabled(self) -> bool:
        """Logging was turned off deliberately, rather than being broken."""
        return self._path is None

    @property
    def available(self) -> bool:
        """Whether a row can actually be appended right now.

        Checked by opening for append rather than by testing for existence: a
        path that is configured but unwritable looks fine until the first send,
        which is the worst moment to find out.
        """
        if self._path is None:
            return False
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "a", encoding="utf-8"):
                pass
            return True
        except OSError:
            return False

    def require(self) -> None:
        """Raise unless a row could be written. Call before acting."""
        if self._path is None:
            raise LogUnavailable(
                f"refusing to act: action logging is disabled ({ENV_VAR} is empty). "
                "An unlogged input path would break evidence integrity."
            )
        if not self.available:
            raise LogUnavailable(
                f"refusing to act: cannot write the action log at {self._path}. "
                "An unlogged input path would break evidence integrity."
            )

    def append(
        self,
        action: str,
        result: str,
        evidence: str = "",
        cls: str = "RO",
        context: str = "",
        actor: str | None = None,
        when: datetime.datetime | None = None,
    ) -> str:
        """Append one row and return it. Raises LogUnavailable if it cannot."""
        if self._path is None:
            raise LogUnavailable(f"no action log configured (set {ENV_VAR})")
        who = actor or self._actor
        if who not in ACTORS:
            raise ValueError(f"unknown actor {who!r}, expected one of {ACTORS}")
        if cls not in CLASSES:
            raise ValueError(f"unknown action class {cls!r}, expected one of {CLASSES}")

        stamp = (when or datetime.datetime.now()).strftime("%Y-%m-%d %H:%M")
        row = "| {} | {} | {} | {} | {} | {} | {} |".format(
            stamp,
            who,
            cls,
            _cell(context) or "-",  # "-" rather than repeating the actor column
            _cell(action),
            _cell(result),
            _cell(evidence),
        )
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "a", encoding="utf-8", newline="\n") as fh:
                fh.write(row + "\n")
        except OSError as exc:
            raise LogUnavailable(f"could not append to {self._path}: {exc}") from exc
        return row

    def correction(self, of: str, problem: str, **kw) -> str:
        """Record that the preceding row did not happen as written.

        Rows go down before the action, so a crash can never leave an action
        that was taken but not recorded. The cost is the opposite error, a row
        saying "sent" when transmission failed, and this is how that is put
        right without deleting anything.
        """
        return self.append(
            action=f"CORRECTION to the preceding row: {of}",
            result=problem,
            **kw,
        )


def parse_row(line: str) -> dict | None:
    """Parse one log row, tolerating rows written before the actor column.

    Column count alone does not identify the format. A legacy row can carry a
    spare trailing cell and so arrive with seven, which made an earlier version
    read its class column as the actor and report actors called "RO" and "LOW".
    So the actor is identified by its value, not by its position.
    """
    s = line.strip()
    if not s.startswith("|") or not s.endswith("|"):
        return None
    cells = [c.strip() for c in s[1:-1].split("|")]

    if len(cells) >= ROW_COLUMNS and cells[1] in ACTORS:
        ts, actor, cls, context, action, result, evidence = cells[:ROW_COLUMNS]
    elif len(cells) >= ROW_COLUMNS - 1:
        ts, cls, context, action, result, evidence = cells[: ROW_COLUMNS - 1]
        actor = ""  # legacy row, written before actors were recorded
    else:
        return None

    # The first cell must actually look like a time. This rejects Markdown
    # table headers and the `| --- | --- |` separator rows that sit in a
    # hand-maintained log, which would otherwise be read as actions.
    if not _TIMESTAMP_RE.match(ts):
        return None
    return {
        "time": ts,
        "actor": actor,
        "class": cls,
        "context": context,
        "action": action,
        "result": result,
        "evidence": evidence,
    }


def _cell(value: str) -> str:
    """Make a value safe for one cell of a pipe-delimited row.

    A newline or a pipe in the payload would otherwise split one action across
    rows or forge a column. Both become a visible placeholder rather than a
    Markdown escape: `\\|` renders correctly but is still a pipe, so anyone
    splitting a row on pipes would see a column that was never there.
    """
    text = (value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\n", " <NL> ").replace("|", "<PIPE>")
    return " ".join(text.split()).strip()


