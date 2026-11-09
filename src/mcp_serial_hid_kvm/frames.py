"""Properties of a captured frame, shared by the agent and the operator.

Both paths need the same answer to "is there a picture here", and for the same
reason: a frame with no picture cannot evidence what input went into. The
operator console refuses to send against one; the agent records that the screen
carried nothing, so a row whose keystrokes may have gone nowhere says so.

The definition lives here rather than in each caller so the threshold cannot
drift between them. A row written by one and read beside the other has to mean
the same thing.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# Spread below which a frame is treated as carrying no picture. Measured
# against real captures: a lock screen, a Terminal and a desktop all sit far
# above 10, while a no-signal frame is exactly 0.0.
BLANK_STDEV = 2.0


def frame_stdev(image: Image.Image) -> float:
    """Spread of a frame, as a cheap proxy for "is there a picture here".

    Computed on a downsample: a no-signal frame is uniform at any resolution,
    and this runs on every refresh.
    """
    small = image.convert("L").resize((160, 90))
    return float(np.asarray(small, dtype=np.float32).std())


# Mean absolute difference below which two frames are treated as the same
# picture. Measured on this capture card against a static terminal: the floor
# between consecutive frames, cursor blink and sensor noise included, sat at
# 0.039. This is 25x that, so a blinking cursor cannot read as activity while
# a single new line of output reads as activity easily.
QUIET_DIFF = 1.0


def signature(image: Image.Image):
    """A small greyscale array, for comparing one frame against another.

    Downsampled for the same reason frame_stdev is: whole-screen change is
    what is being detected, and this runs in a polling loop.
    """
    small = image.convert("L").resize((320, 180))
    return np.asarray(small, dtype=np.float32)


def difference(a, b) -> float:
    """Mean absolute difference between two signatures, on a 0-255 scale."""
    return float(np.abs(a - b).mean())


def is_quiet(diff: float) -> bool:
    """Whether two frames are the same picture, allowing for noise."""
    return diff < QUIET_DIFF


# Share of a frame that must sit at one intensity for it to read as a user
# interface rather than a photograph. A window, a terminal or a dialog has a
# dominant flat background whichever theme it uses; a lock screen wallpaper,
# a screensaver or a bare photo desktop does not.
#
# Measured on this engagement's captures, all from a macOS target: lock
# screens and a photo desktop scored 19.0-20.1%, terminals 58.4-94.8%, and a
# light terminal inverted to simulate a dark theme still scored 91.4%. The
# threshold sits in that gap.
#
# That calibration is macOS only, and the number should not be trusted
# elsewhere without re-measuring. A Windows desktop is commonly a solid
# colour, which would read as almost perfectly flat and pass a threshold
# chosen to reject a photographic wallpaper: the opposite failure to the one
# seen here. This is why the check guards only a shell command, where a flat
# screen is a reasonable expectation, and why the blank check below carries
# every keystroke: a sleeping display is black on every operating system.
FLAT_BACKGROUND_MIN = 0.40


def flat_fraction(image: Image.Image, tol: int = 8) -> float:
    """Largest share of the frame sitting within `tol` of one intensity.

    Deliberately not OCR. Reading a lock screen's text was tried first and is
    both unreliable, because it is white text over a photograph rather than the
    terminal text Tesseract is tuned for here, and slow: 3.6 seconds a frame
    against 11 milliseconds for this.
    """
    small = image.convert("L").resize((320, 180))
    g = np.asarray(small, dtype=np.uint8)
    hist = np.bincount(g.ravel(), minlength=256).astype(np.float32)
    widened = np.convolve(hist, np.ones(2 * tol + 1), mode="same")
    return float(widened.max() / g.size)


def has_flat_background(fraction: float) -> bool:
    """Whether a frame is dominated by one flat shade, as a terminal is.

    Used only where a terminal is expected. It is not a lock-screen detector
    and must not be read as one: measured on a macOS target, an unlocked
    desktop with an application window open scored 19.3% against a lock
    screen's 19.0-20.1%, because the same wallpaper fills both and the menu
    bar and dock are too thin and too translucent to shift a whole-frame
    statistic. A top-strip and a dock-band measurement were tried and separate
    no better.

    What is reliably detectable across operating systems is a blank frame, so
    that is what guards every keystroke. This only guards a shell command,
    where expecting something that looks like a shell is reasonable.
    """
    return fraction >= FLAT_BACKGROUND_MIN


def is_blank(stdev: float) -> bool:
    """Whether a measured spread means the frame carries no picture.

    A capture card with no HDMI signal, or one caught while the display is
    still waking, returns a uniform field. Any real screen has texture, so
    near-zero spread across the frame means there is nothing to look at.
    """
    return stdev < BLANK_STDEV


__all__ = [
    "BLANK_STDEV",
    "FLAT_BACKGROUND_MIN",
    "flat_fraction",
    "has_flat_background",
    "QUIET_DIFF",
    "difference",
    "frame_stdev",
    "is_blank",
    "is_quiet",
    "signature",
]
