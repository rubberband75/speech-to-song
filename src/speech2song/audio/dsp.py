"""Small DSP helpers: finding quiet cut points and applying edge fades."""

from typing import Literal

import numpy as np


def frame_levels_db(mono: np.ndarray, centers: np.ndarray, frame: int) -> np.ndarray:
    """RMS level (dBFS) of a `frame`-sample window centred on each index in `centers`.

    Only the region the windows cover is summed, so `mono` may be a whole recording.
    """
    half = frame // 2
    lo = np.clip(centers - half, 0, len(mono))
    hi = np.clip(centers + half, 0, len(mono))
    base, top = int(lo.min()), int(hi.max())
    region = mono[base:top].astype(np.float64)
    squared = np.concatenate([[0.0], np.cumsum(region**2)])
    energy = (squared[hi - base] - squared[lo - base]) / np.maximum(hi - lo, 1)
    return 10 * np.log10(energy + 1e-12)


def quietest_point(
    mono: np.ndarray,
    lo: int,
    hi: int,
    boundary: int,
    *,
    frame: int,
    tolerance_db: float = 0.5,
    side: Literal["start", "end"] | None = None,
    pad: int = 0,
) -> tuple[int, float]:
    """A cut point in [lo, hi] where the signal is quietest, and its level in dBFS.

    Points within `tolerance_db` of the minimum form quiet stretches; the stretch
    nearest `boundary` is used. For a clip start (`side="start"`) the cut goes `pad`
    samples before the stretch's end (the speech onset), for a clip end `pad` samples
    after its start, so a fade of up to `pad` falls in the quiet. Without `side`, the
    point nearest `boundary` is used. Indices are positions in `mono`.
    """
    if hi < lo:
        raise ValueError("empty search range")
    centers = np.arange(lo, hi + 1)
    levels = frame_levels_db(mono, centers, frame)
    quiet = centers[levels <= levels.min() + tolerance_db]
    breaks = np.flatnonzero(np.diff(quiet) > 1)
    runs = np.split(quiet, breaks + 1)

    def distance(run: np.ndarray) -> int:
        if run[0] <= boundary <= run[-1]:
            return 0
        return int(min(abs(run[0] - boundary), abs(run[-1] - boundary)))

    run = min(runs, key=distance)
    if side == "start":
        index = max(int(run[0]), int(run[-1]) - pad)
    elif side == "end":
        index = min(int(run[-1]), int(run[0]) + pad)
    else:
        index = int(run[np.argmin(np.abs(run - boundary))])
    return index, float(levels[index - lo])


def fade_edges(block: np.ndarray, fade: int) -> np.ndarray:
    """Copy of `block` (frames x channels) with raised-cosine fades of `fade` samples.

    Only the first and last `fade` samples change; the interior stays bit-identical.
    """
    out = np.array(block, dtype=np.float32, copy=True)
    n = min(fade, len(out) // 2)
    if n <= 0:
        return out
    ramp = (0.5 - 0.5 * np.cos(np.pi * (np.arange(n) + 0.5) / n)).astype(np.float32)
    out[:n] *= ramp[:, None]
    out[len(out) - n :] *= ramp[::-1, None]
    return out
