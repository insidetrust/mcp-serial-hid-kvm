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


def is_blank(stdev: float) -> bool:
    """Whether a measured spread means the frame carries no picture.

    A capture card with no HDMI signal, or one caught while the display is
    still waking, returns a uniform field. Any real screen has texture, so
    near-zero spread across the frame means there is nothing to look at.
    """
    return stdev < BLANK_STDEV


__all__ = ["BLANK_STDEV", "frame_stdev", "is_blank"]
