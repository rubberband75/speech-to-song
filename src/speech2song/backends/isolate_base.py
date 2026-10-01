"""Voice isolation plumbing: windowed, streamed processing around a separator function.

The separator maps a (frames, channels) float32 block to a block of the same shape.
Long files are processed in windows, each read with extra context on both sides, and
neighbouring windows are crossfaded, so memory stays bounded for hour-long lectures.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import soundfile as sf

from speech2song.errors import StageError
from speech2song.manifest import temp_path_for

Separator = Callable[[np.ndarray], np.ndarray]
Progress = Callable[[int, int], None]  # (frames done, total frames)

_WAV_LIMIT_BYTES = 4_000_000_000  # plain WAV can't exceed 4 GiB; switch to RF64 before that


def window_bounds(total: int, window: int) -> list[int]:
    """Equal-sized window boundaries covering [0, total), avoiding a tiny last window."""
    count = max(1, round(total / window)) if window > 0 else 1
    return [round(i * total / count) for i in range(count + 1)]


def isolate_stream(
    read: Callable[[int, int], np.ndarray],
    write: Callable[[np.ndarray], None],
    total: int,
    separate: Separator,
    *,
    window: int,
    context: int,
    crossfade: int,
    progress: Progress | None = None,
) -> None:
    """Run `separate` over [0, total) window by window and write the result in order.

    Window i owns [b_i, b_i+1). It is read with `context` frames on each side and
    produces [b_i - crossfade/2, b_i+1 + crossfade/2); the overlap with the previous
    window is crossfaded linearly. Requires context >= crossfade/2.
    """
    bounds = window_bounds(total, window)
    half = crossfade // 2
    if half > context:
        raise ValueError("context must be at least half the crossfade")
    pending: np.ndarray | None = None  # previous window's output over the next crossfade
    last = len(bounds) - 2
    for i in range(last + 1):
        lo, hi = bounds[i], bounds[i + 1]
        out_start, out_end = max(0, lo - half), min(total, hi + half)
        read_start, read_end = max(0, lo - context), min(total, hi + context)
        block = read(read_start, read_end)
        separated = separate(block)
        if separated.shape != block.shape:
            raise StageError(f"separator changed block shape {block.shape} -> {separated.shape}")
        out = np.array(separated[out_start - read_start : out_end - read_start], dtype=np.float32)
        if pending is not None and len(pending):
            ramp = ((np.arange(len(pending)) + 0.5) / len(pending)).astype(np.float32)[:, None]
            out[: len(pending)] = pending * (1 - ramp) + out[: len(pending)] * ramp
        if i < last:
            keep = (hi - half) - out_start
            write(out[:keep])
            pending = out[keep:]
        else:
            write(out)
        if progress is not None:
            progress(hi, total)


def isolate_file(
    src: Path,
    dst: Path,
    separate: Separator,
    *,
    window_s: float,
    context_s: float,
    crossfade_s: float,
    progress: Progress | None = None,
) -> None:
    """Isolate `src` into a float32 WAV `dst` with identical rate, channels and length."""
    tmp = temp_path_for(dst)
    try:
        with sf.SoundFile(str(src)) as fin:
            rate, channels, total = fin.samplerate, fin.channels, fin.frames
            big = total * channels * 4 > _WAV_LIMIT_BYTES
            with sf.SoundFile(
                str(tmp),
                "w",
                samplerate=rate,
                channels=channels,
                subtype="FLOAT",
                format="RF64" if big else "WAV",
            ) as fout:

                def read(start: int, end: int) -> np.ndarray:
                    fin.seek(start)
                    return fin.read(end - start, dtype="float32", always_2d=True)

                isolate_stream(
                    read,
                    fout.write,
                    total,
                    separate,
                    window=round(window_s * rate),
                    context=round(context_s * rate),
                    crossfade=round(crossfade_s * rate),
                    progress=progress,
                )
        written = sf.info(str(tmp))
        if (written.frames, written.channels, written.samplerate) != (total, channels, rate):
            raise StageError(
                f"isolated audio does not match the source: {written.frames} frames, "
                f"{written.channels} ch, {written.samplerate} Hz vs {total}, {channels}, {rate}"
            )
        tmp.replace(dst)
    finally:
        tmp.unlink(missing_ok=True)
