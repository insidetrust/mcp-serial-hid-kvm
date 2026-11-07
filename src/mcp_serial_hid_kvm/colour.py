"""Colour correction for the HDMI capture path.

The WARRKY MS2130 capture device does not return a neutral white. Measured on
07/11/2026 against the target's own UI, a true white renders as RGB
(255, 240, 219): a warm cast that makes every screenshot look pale brown.

The error is a pure multiplicative gain per channel, not gamma and not an
offset. Four regions that are independently known to be white (a flat Terminal
background, a notification card, the tab bar, the window margin) all returned
the same ratios at different brightness levels:

    B/R = 0.859 .. 0.860      G/R = 0.941 .. 0.945

Constant ratios across brightness is what distinguishes a gain error from a
gamma error, so a single scale factor per channel corrects the whole image.

This is a calibration of a known hardware defect, applied as a fixed,
content-independent transform. It is deliberately NOT an automatic per-image
white balance: an auto white balance would "correct" the genuinely warm colours
of a sunset wallpaper, which would be a manipulation rather than a correction.
The same numbers are applied to every frame regardless of content.

Correction is recorded in the action log and can be turned off, so a raw frame
is always obtainable.

A source-side fix may be preferable where it is available: MS2130 firmware F3
(January 2024) is reported to address dark capture and limited-range YUV
handling. Re-measure before and after any firmware change, because that would
change these numbers.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# Measured white point: what the device returns for a true (255, 255, 255).
# Re-derive with measure_white_point() if the capture device or firmware
# changes, and record the date and method when you do.
MEASURED_WHITE = (255.0, 240.0, 219.0)
MEASURED_ON = "2026-11-07"


def gains_from_white_point(white: tuple[float, float, float]) -> tuple[float, float, float]:
    """Per-channel multipliers that map the measured white to neutral.

    Normalised against the brightest channel rather than against 255, so the
    correction never scales a channel down and never darkens the image. The
    brightest channel is the one already at or near clipping; pushing it
    further would only destroy highlight detail.
    """
    peak = max(white)
    return tuple(peak / max(c, 1e-6) for c in white)  # type: ignore[return-value]


DEFAULT_GAINS = gains_from_white_point(MEASURED_WHITE)


def measure_white_point(
    image: Image.Image, percentile: float = 99.0
) -> tuple[float, float, float]:
    """Measure what this device returns for white, from a frame containing it.

    Point it at a frame with a large genuinely white area, such as a Terminal
    window on the default light profile. Returns the mean of each channel over
    the brightest pixels by luminance.
    """
    a = np.asarray(image.convert("RGB"), dtype=np.float32)
    lum = a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    mask = lum >= np.percentile(lum, percentile)
    if not mask.any():
        raise ValueError("no pixels above the percentile; frame may be blank")
    return (
        float(a[..., 0][mask].mean()),
        float(a[..., 1][mask].mean()),
        float(a[..., 2][mask].mean()),
    )


def correct(
    image: Image.Image, gains: tuple[float, float, float] | None = None
) -> Image.Image:
    """Apply the per-channel gain. Returns a new image; the input is untouched."""
    g = gains if gains is not None else DEFAULT_GAINS
    if all(abs(x - 1.0) < 1e-9 for x in g):
        return image
    a = np.asarray(image.convert("RGB"), dtype=np.float32)
    a *= np.array(g, dtype=np.float32)
    np.clip(a, 0, 255, out=a)
    return Image.fromarray(a.astype(np.uint8), mode="RGB")


def describe(gains: tuple[float, float, float] | None = None) -> str:
    """One line for the action log, so the transform is disclosed and
    reproducible rather than silently baked into the evidence."""
    g = gains if gains is not None else DEFAULT_GAINS
    return (
        f"colour corrected, fixed gain R={g[0]:.4f} G={g[1]:.4f} B={g[2]:.4f} "
        f"(white point {MEASURED_WHITE} measured {MEASURED_ON})"
    )
