#!/usr/bin/env python3
"""Tests for the server's action log and pre-send evidence.

Zero-dependency harness: no pytest needed, run it directly in the venv that
holds the package.

    python tests/test_server_evidence.py

It is also discoverable by pytest (test_* functions, plain asserts) if that is
installed. Prints "<N> checks, <F> failures" and exits non-zero on any failure.

These cover the rules rather than the transport: an input action is logged
before it is sent, it is refused when it cannot be logged, and the frame
captured just before delivery is recorded as that row's evidence. No test
reaches a device: the frame grab and the client are both stubbed.
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from PIL import Image  # noqa: E402

from mcp_serial_hid_kvm import actionlog, frames, server  # noqa: E402

CHECKS = 0
FAILURES: list[str] = []


def check(condition: bool, what: str) -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(what)


# --------------------------------------------------------------------------
# Stubs
# --------------------------------------------------------------------------


def textured_frame() -> Image.Image:
    """A frame with a picture in it: spread well above the blank threshold."""
    image = Image.new("RGB", (320, 180))
    for x in range(320):
        for y in range(0, 180, 2):
            image.putpixel((x, y), (255, 255, 255))
    return image


def blank_frame() -> Image.Image:
    """What the capture card returns with no HDMI signal: a uniform field."""
    return Image.new("RGB", (320, 180), (0, 0, 0))


def photo_frame() -> Image.Image:
    """A frame with no dominant flat shade, as a photograph has none.

    The real failure looked like this: the lock screen captured on 08/11/2026
    measured a spread of 40.8, far above the blank threshold of 2.0, because a
    wallpaper is the least blank thing on the machine. Spread cannot catch it;
    flatness can.
    """
    image = Image.new("RGB", (320, 180))
    px = image.load()
    for x in range(320):
        for y in range(180):
            # A smooth two-axis ramp with some texture, so no single intensity
            # dominates and the histogram stays spread out.
            v = (x * 255 // 320 + y * 255 // 180) // 2
            px[x, y] = ((v + (x * y) % 37) % 256, v, (255 - v))
    return image


def ui_frame() -> Image.Image:
    """A frame that reads as an interface: a flat ground with text on it."""
    image = Image.new("RGB", (320, 180), (250, 250, 250))
    px = image.load()
    for row in range(10, 170, 12):
        for x in range(8, 300, 2):
            px[x, row] = (20, 20, 20)
    return image


class Harness:
    """A temporary log and capture directory, with the frame grab stubbed."""

    def __init__(self, frame=textured_frame, log: bool = True, presend: bool = True):
        self._frame = frame
        self._log_wanted = log
        self._presend = presend
        self.sent: list[str] = []

    def __enter__(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.captures = root / "captures"
        self.log_path = root / "action-log.md" if self._log_wanted else None

        self._saved = (
            server._log,
            server._capture_image,
            server.config.capture_log_dir,
            server.config.presend_evidence,
        )
        server._log = actionlog.ActionLog(self.log_path, actor=actionlog.AGENT)
        server.config.capture_log_dir = str(self.captures)
        server.config.presend_evidence = self._presend

        def grab(quality: int = 85):
            if self._frame is None:
                raise RuntimeError("capture device unavailable")
            return self._frame()

        server._capture_image = grab
        return self

    def __exit__(self, *exc):
        (
            server._log,
            server._capture_image,
            server.config.capture_log_dir,
            server.config.presend_evidence,
        ) = self._saved
        self.tmp.cleanup()
        return False

    def rows(self) -> list[dict]:
        if self.log_path is None or not self.log_path.exists():
            return []
        text = io.open(self.log_path, encoding="utf-8").read()
        return [r for r in (actionlog.parse_row(l) for l in text.splitlines()) if r]

    def frames_on_disk(self) -> list[str]:
        if not self.captures.exists():
            return []
        return sorted(p.name for p in self.captures.iterdir())


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_input_row_cites_the_frame_it_went_into():
    """The point of the whole change: an input row names a readable frame."""
    with Harness() as h:
        with server._acting("type_text: whoami", "RO", "Terminal"):
            h.sent.append("whoami")

        rows = h.rows()
        check(len(rows) == 1, f"one row expected, got {len(rows)}")
        row = rows[0]
        check(row["actor"] == "agent", f"actor should be agent, got {row['actor']!r}")
        check(row["action"] == "type_text: whoami", f"action was {row['action']!r}")
        check(row["evidence"] != "", "an input row must cite evidence")
        check(
            os.path.exists(row["evidence"]),
            f"evidence path does not exist: {row['evidence']!r}",
        )
        check(
            "presend" in os.path.basename(row["evidence"]),
            f"evidence should be named as a pre-send frame: {row['evidence']!r}",
        )
        # The frame must be readable as an image, not merely present.
        with Image.open(row["evidence"]) as saved:
            saved.load()
            check(saved.size == (320, 180), f"frame size was {saved.size}")


def test_frame_is_on_disk_before_the_row_is_written():
    """Ordering. A row citing a frame that does not exist yet is a dangling
    reference, and the window would be exactly the crash window that the
    log-before-acting rule exists to close."""
    with Harness() as h:
        seen: dict[str, object] = {}
        real_append = server._log.append

        def spy(**kw):
            seen["evidence"] = kw.get("evidence", "")
            seen["existed_at_append"] = bool(kw.get("evidence")) and os.path.exists(
                kw["evidence"]
            )
            return real_append(**kw)

        server._log.append = spy  # type: ignore[method-assign]
        with server._acting("send_key: enter", "RO", "Terminal"):
            pass
        check(
            seen.get("existed_at_append") is True,
            "the pre-send frame must be saved before the row that cites it",
        )


def test_a_refused_action_leaves_no_frame():
    """A send refused for want of a log should not litter the evidence
    directory with a frame belonging to no row."""
    with Harness(log=False) as h:
        refused = False
        try:
            with server._acting("type_text: whoami", "RO", "Terminal"):
                h.sent.append("whoami")
        except actionlog.LogUnavailable:
            refused = True
        check(refused, "an action must be refused when it cannot be logged")
        check(h.sent == [], "nothing may be transmitted after a refusal")
        check(
            h.frames_on_disk() == [],
            f"a refusal should write no frame, found {h.frames_on_disk()}",
        )


def test_blank_frame_is_recorded_but_does_not_block():
    """A target that is asleep has no picture, and reaching it is exactly when
    input is needed. So the row says so and the input still goes."""
    with Harness(frame=blank_frame) as h:
        with server._acting("mouse_move: relative (1, 0)", "RO", "Wake",
                            keyboard=False):
            h.sent.append("wake")

        check(h.sent == ["wake"], "a blank frame must not block the pointer")
        row = h.rows()[0]
        check(
            "no picture" in row["result"],
            f"a blank pre-send frame should be noted in the row: {row['result']!r}",
        )
        check(row["evidence"] != "", "the blank frame is still evidence; it is saved")
        check(frames.is_blank(0.0), "a uniform field must measure as blank")


def test_capture_failure_blocks_keystrokes_but_not_the_pointer():
    """The contract changed when the focus guard went in, deliberately.

    It used to be that an input whose frame could not be captured still went,
    on the grounds that driving a target must not depend on the capture card.
    That is still true of the pointer, which is how a sleeping target is woken.
    It is no longer true of keystrokes: without a frame there is nothing to
    judge, and typing blind is the failure the guard exists to prevent.
    """
    with Harness(frame=None) as h:
        refused = False
        try:
            with server._acting("type_text: whoami", "RO", "Terminal"):
                h.sent.append("whoami")
        except server.FocusRefused:
            refused = True
        check(refused, "keystrokes must be refused when there is no frame")
        check(h.sent == [], "and nothing transmitted")

        with server._acting("mouse_move: relative (1, 0)", "RO", "Wake",
                            keyboard=False):
            h.sent.append("wake")
        check(h.sent == ["wake"], "but the pointer must still go through")
        row = [r for r in h.rows() if "mouse_move" in r["action"]][0]
        check("capture unavailable" in row["result"],
              f"and the row should say there was no frame: {row['result']!r}")


def test_capture_failure_with_the_guard_off_still_sends():
    with Harness(frame=None) as h:
        server.config.focus_guard = False
        try:
            with server._acting("type_text: whoami", "RO", "Terminal"):
                h.sent.append("whoami")
        finally:
            server.config.focus_guard = True

        check(h.sent == ["whoami"], "the override must restore the old contract")
        row = h.rows()[0]
        check(row["evidence"] == "", f"no frame, so no evidence: {row['evidence']!r}")
        check(
            "capture unavailable" in row["result"],
            f"the row should say why there is no frame: {row['result']!r}",
        )


def test_frames_in_the_same_second_do_not_overwrite():
    """The stamp is to the second and a key burst beats that easily. Two rows
    pointing at one file would mean one row citing another's screen."""
    with Harness() as h:
        for _ in range(3):
            with server._acting("send_key: tab", "RO", "Terminal"):
                pass

        evidence = [r["evidence"] for r in h.rows()]
        check(len(set(evidence)) == 3, f"evidence paths must be distinct: {evidence}")
        check(
            len(h.frames_on_disk()) == 3,
            f"three frames expected on disk, found {h.frames_on_disk()}",
        )
        for path in evidence:
            check(os.path.exists(path), f"evidence missing: {path}")


def test_presend_evidence_can_be_turned_off():
    with Harness(presend=False) as h:
        with server._acting("type_text: whoami", "RO", "Terminal"):
            h.sent.append("whoami")

        row = h.rows()[0]
        check(row["evidence"] == "", "evidence should be empty when disabled")
        check(h.frames_on_disk() == [], "no frames should be written when disabled")
        check(
            row["result"] == "sent through the HID bridge",
            f"no note when disabled deliberately: {row['result']!r}",
        )


def test_transmission_failure_appends_a_correction():
    """Pre-existing behaviour, retested here because the ordering changed
    around it."""
    with Harness() as h:
        raised = False
        try:
            with server._acting("type_text: whoami", "RO", "Terminal"):
                raise RuntimeError("serial write failed")
        except RuntimeError:
            raised = True

        check(raised, "the original error must propagate")
        rows = h.rows()
        check(len(rows) == 2, f"a correction row expected, got {len(rows)}")
        check(
            rows[1]["action"].startswith("CORRECTION"),
            f"second row should be a correction: {rows[1]['action']!r}",
        )
        check(
            "delivery uncertain" in rows[1]["result"],
            f"a correction must not claim the input did not land: {rows[1]['result']!r}",
        )


def test_capture_screen_row_still_carries_its_own_evidence():
    """_save_capture_log was refactored onto the shared frame writer."""
    with Harness() as h:
        path = server._save_capture_log(textured_frame(), "capture")
        check(path is not None and os.path.exists(path), f"capture not saved: {path!r}")
        row = h.rows()[0]
        check(row["action"] == "capture_screen (capture)", f"action {row['action']!r}")
        check(row["evidence"] == path, f"evidence {row['evidence']!r} != {path!r}")
        check(row["class"] == "RO", f"a capture is read-only, got {row['class']!r}")


def test_every_input_tool_is_an_acting_tool():
    """INPUT_TOOLS is what gives a tool an action class and a logged row. A
    tool that puts input into the target and is missing from it would send
    unlogged and unevidenced."""
    putting_input_in = {
        "type_text",
        "send_key",
        "send_key_sequence",
        "mouse_move",
        "mouse_click",
        "mouse_drag",
        "mouse_scroll",
        "execute_and_read",
    }
    check(
        server.INPUT_TOOLS == putting_input_in,
        f"INPUT_TOOLS drifted: {sorted(server.INPUT_TOOLS ^ putting_input_in)}",
    )


def test_settle_returns_once_the_screen_stops_changing():
    """A quick command must not cost the slowest command's wait."""
    import asyncio

    frame = textured_frame()
    with Harness() as h:
        server._capture_image = lambda quality=85: frame  # identical every poll
        image, waited, settled = asyncio.run(server._wait_for_settle(0.0, timeout_s=5.0))
        check(settled is True, "an unchanging screen must settle")
        check(waited < 2.0, f"settling took {waited:.2f}s on a static screen")
        check(image is frame, "the settled frame is what comes back")


def test_settle_waits_while_output_is_still_arriving():
    """The case this exists for. A command still writing to the screen must
    not be read as finished: that is the failure that looks like an answer."""
    import asyncio

    with Harness() as h:
        changing = {"n": 0}

        def busy(quality=85):
            # A different picture every poll, as output scrolls past.
            changing["n"] += 1
            image = Image.new("RGB", (320, 180))
            for x in range(0, 320, 2):
                for y in range(180):
                    image.putpixel((x, y), (changing["n"] * 37 % 256, 128, 200))
            return image

        server._capture_image = busy
        _image, waited, settled = asyncio.run(
            server._wait_for_settle(0.0, timeout_s=1.5)
        )
        check(settled is False, "a changing screen must not report as settled")
        check(waited >= 1.5, f"it must wait out the timeout, waited {waited:.2f}s")


def test_settle_honours_the_minimum_wait():
    """Some commands pause before printing anything. Returning during that
    pause would read an empty screen as the finished answer."""
    import asyncio

    frame = textured_frame()
    with Harness() as h:
        server._capture_image = lambda quality=85: frame
        _image, waited, settled = asyncio.run(server._wait_for_settle(1.0, timeout_s=5.0))
        check(settled is True, "it still settles")
        check(waited >= 1.0, f"the minimum wait was not honoured: {waited:.2f}s")


def test_quiet_threshold_is_well_clear_of_capture_noise():
    """QUIET_DIFF is set from a measurement, and the measurement is recorded
    in the module. This guards the ordering, not the exact number."""
    a = frames.signature(textured_frame())
    check(frames.difference(a, a) == 0.0, "a frame must not differ from itself")
    check(frames.is_quiet(0.039), "the measured noise floor must read as quiet")
    check(not frames.is_quiet(5.0), "a real screen change must not read as quiet")


def test_driving_a_graphical_interface_is_not_blocked():
    """Driving a GUI is a first-class use and must not be gated on the screen
    looking like a terminal. A photographic frame is an unlocked desktop as
    often as it is a lock screen: measured on a real target, a desktop with an
    application window open scored 19.3% against a lock screen's 19.0-20.1%.
    """
    with Harness(frame=photo_frame) as h:
        with server._acting("type_text: hello", "RO", "Finder"):
            h.sent.append("hello")
        check(h.sent == ["hello"], "typing into a GUI must not be refused")

        with server._acting("send_key: tab", "RO", "Finder"):
            h.sent.append("tab")
        check("tab" in h.sent, "nor must a key press")


def test_the_refusal_is_itself_recorded():
    """A refusal is evidence: somebody tried to type at a screen nobody could
    see, at a time, and the row cites the frame that was refused on."""
    with Harness(frame=blank_frame) as h:
        try:
            with server._acting("execute_and_read: whoami then Enter", "RO",
                                "Terminal"):
                pass
        except server.FocusRefused:
            pass
        rows = h.rows()
        check(len(rows) == 1, f"one row expected, got {len(rows)}")
        check(rows[0]["action"].startswith("REFUSED"),
              f"the row should record the refusal: {rows[0]['action']!r}")
        check(rows[0]["evidence"] != "", "and cite the frame it refused on")


def test_the_pointer_is_never_guarded():
    """Waking a sleeping target is a mouse move against a screen showing
    nothing. Guarding it would restore the deadlock the wake path breaks."""
    with Harness(frame=photo_frame) as h:
        with server._acting("mouse_move: relative (1, 0)", "RO", "Wake",
                            keyboard=False):
            h.sent.append("wake")
        check(h.sent == ["wake"], "a mouse move must go through regardless")


def test_keystrokes_are_refused_when_there_is_no_frame_at_all():
    """Without a frame there is no way to know where the keystrokes land."""
    with Harness(frame=None) as h:
        refused = False
        try:
            with server._acting("type_text: whoami", "RO", "Terminal"):
                h.sent.append("whoami")
        except server.FocusRefused as exc:
            refused = True
            check("no frame" in str(exc), f"the refusal should say why: {exc}")
        check(refused, "typing blind must be refused")
        check(h.sent == [], "nothing transmitted")


def test_the_guard_can_be_turned_off():
    with Harness(frame=photo_frame) as h:
        server.config.focus_guard = False
        try:
            with server._acting("type_text: whoami", "RO", "Terminal"):
                h.sent.append("whoami")
        finally:
            server.config.focus_guard = True
        check(h.sent == ["whoami"], "the override must work")


def test_the_blank_check_applies_to_every_keyboard_tool():
    """A sleeping display is detectable on any operating system, so unlike the
    flatness check this one is not scoped to shell commands."""
    with Harness(frame=blank_frame) as h:
        for action in ("type_text: hello", "send_key: enter",
                       "execute_and_read: id then Enter"):
            refused = False
            try:
                with server._acting(action, "RO", "Finder"):
                    h.sent.append(action)
            except server.FocusRefused:
                refused = True
            check(refused, f"a blank screen must refuse {action!r}")
        check(h.sent == [], "nothing transmitted against a blank screen")


def test_keystrokes_are_refused_against_a_sleeping_display():
    """The gap the first live use found.

    A frame with no picture is the flattest frame there is, so the interface
    check passes it: 100% of it sits at one shade. The blank check has to run
    first or a sleeping display is the one screen the guard waves through.
    """
    with Harness(frame=blank_frame) as h:
        refused = False
        try:
            with server._acting("type_text: whoami", "RO", "Terminal"):
                h.sent.append("whoami")
        except server.FocusRefused as exc:
            refused = True
            check("no picture" in str(exc), f"the refusal should say why: {exc}")
        check(refused, "keystrokes must be refused against a blank screen")
        check(h.sent == [], "nothing transmitted")



def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        try:
            test()
        except Exception as exc:  # a crashed test is a failure, not a stop
            FAILURES.append(f"{test.__name__} raised {exc!r}")
    for failure in FAILURES:
        print(f"FAIL: {failure}")
    print(f"{CHECKS} checks, {len(FAILURES)} failures")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
