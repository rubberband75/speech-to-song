"""DSP helpers: quiet cut points and edge fades (clips), and the speech guard, loudness,
true peak and limiting (mix). Pure numpy/scipy; no file I/O."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
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


# --- Speech guard -----------------------------------------------------------------------------


GUARD_BAND_HZ = (200.0, 5000.0)  # where music masks words
GUARD_FRAME_S = 0.1  # speech level frames inside a word
GUARD_CORE_DB = 10.0  # a word's level: the mean of its frames within this of its loudest
GUARD_HOP_S = 0.01
GUARD_ATTACK_DB_S = 40.0  # how fast the music dips, ahead of the word
GUARD_RELEASE_DB_S = 12.0  # how fast it comes back after it
GUARD_GROUP_S = 0.5  # across a pause shorter than this the dip holds between two words


def band_filter(audio: np.ndarray, sr: int) -> np.ndarray:
    """`audio` (frames x channels) band-passed to GUARD_BAND_HZ, as float64."""
    from scipy.signal import butter, sosfilt

    sos = butter(2, GUARD_BAND_HZ, btype="bandpass", fs=sr, output="sos")
    signal = _as_2d(audio).astype(np.float64)
    signal[0::2] -= 1e-12  # keeps the filter out of subnormal floats (see loudness_lufs)
    signal[1::2] += 1e-12
    return sosfilt(sos, signal, axis=0)


def channel_levels_db(filtered: np.ndarray, centers: np.ndarray, frame: int) -> np.ndarray:
    """Level (dB) of a `frame`-sample window at each centre, with the channels' energies
    averaged (a mono sum would cancel wide stereo content)."""
    energy = np.mean([10 ** (frame_levels_db(filtered[:, c], centers, frame) / 10)
                      for c in range(filtered.shape[1])], axis=0)  # fmt: skip
    return 10 * np.log10(energy + 1e-12)


def word_level_db(filtered: np.ndarray, start: int, end: int, sr: int) -> float:
    """A word's speech level: the mean energy of its 100 ms frames within GUARD_CORE_DB
    of its loudest, so silence inside loose word timings doesn't count."""
    hop = max(1, round(GUARD_HOP_S * sr))
    levels = channel_levels_db(filtered, np.arange(start, max(end, start + 1), hop),
                               max(2, round(GUARD_FRAME_S * sr)))  # fmt: skip
    core = levels[levels >= levels.max() - GUARD_CORE_DB]
    return float(10 * np.log10(np.mean(10 ** (core / 10))))


def speech_guard(
    speech: np.ndarray,
    music: np.ndarray,
    sr: int,
    words: Sequence[tuple[int, int]],
    *,
    margin_db: float,
) -> tuple[np.ndarray, list[float]]:
    """Per-sample gain (at most 1) for the music, and each word's margin before it:
    under every word (sample spans in `words`), the music in the speech band stays at
    least `margin_db` under the word's level. Across a pause shorter than GUARD_GROUP_S
    the dip holds at the shallower of the two neighbours' dips, so a phrase gets one
    smooth dip, not one pumping per word. The dip is rate-limited (slowly), so it starts
    ahead of the word and recovers after it."""
    frames = len(music)
    hop = max(1, round(GUARD_HOP_S * sr))
    centers = np.arange(0, frames, hop)
    gain = np.zeros(len(centers))
    if not words or len(centers) == 0:
        return np.ones(frames, dtype=np.float32), []
    spoken, heard = band_filter(speech, sr), band_filter(music, sr)
    margins = []
    spans = []
    for start, end in words:
        end = max(end, start + 1)
        under = channel_levels_db(heard, np.array([(start + end) // 2]), end - start)[0]
        margin = word_level_db(spoken, start, end, sr) - under
        margins.append(float(margin))
        spans.append((start, end, min(0.0, margin - margin_db)))
    ordered = sorted(spans)
    for start, end, dip in ordered:
        a, b = np.searchsorted(centers, [start, end])
        gain[a:b] = np.minimum(gain[a:b], dip)
    for (_, end, dip), (start, _, next_dip) in pairwise(ordered):
        if 0 < start - end < GUARD_GROUP_S * sr:  # a short pause: no recovery between words
            a, b = np.searchsorted(centers, [end, start])
            gain[a:b] = np.minimum(gain[a:b], max(dip, next_dip))
    attack, release = GUARD_ATTACK_DB_S * GUARD_HOP_S, GUARD_RELEASE_DB_S * GUARD_HOP_S
    values = gain.tolist()
    for i in range(len(values) - 2, -1, -1):  # dip ahead of the word
        values[i] = min(values[i], values[i + 1] + attack)
    for i in range(1, len(values)):  # recover after it
        values[i] = min(values[i], values[i - 1] + release)
    gain_db = np.interp(np.arange(frames), centers, np.array(values))
    return (10 ** (gain_db / 20)).astype(np.float32), margins


def dips(gain: np.ndarray, sr: int, below_db: float = -0.5) -> list[tuple[float, float, float]]:
    """(start s, end s, deepest dB) of each stretch where `gain` is below `below_db`."""
    with np.errstate(divide="ignore"):
        level = 20 * np.log10(np.maximum(gain, 1e-9))
    low = level < below_db
    edges = np.flatnonzero(np.diff(np.r_[0, low.astype(np.int8), 0]))
    return [(a / sr, b / sr, float(level[a:b].min())) for a, b in zip(edges[::2], edges[1::2],
                                                                      strict=True)]  # fmt: skip


# --- Loudness and peaks ----------------------------------------------------------------------

OVERSAMPLE = 4
_CHUNK = 1 << 18  # samples per oversampling chunk
_PAD = 64  # context on each side of a chunk for the resampling filter


def _as_2d(audio: np.ndarray) -> np.ndarray:
    return audio[:, None] if audio.ndim == 1 else audio


def oversampled_peaks(audio: np.ndarray) -> np.ndarray:
    """Per input sample, the largest absolute value of the 4x-oversampled signal around
    it, over all channels (the ITU-R BS.1770 true-peak approach). Processed in chunks."""
    from scipy.signal import resample_poly

    audio = _as_2d(audio)
    n = len(audio)
    peaks = np.zeros(n, dtype=np.float32)
    for start in range(0, n, _CHUNK):
        end = min(n, start + _CHUNK)
        lo, hi = max(0, start - _PAD), min(n, end + _PAD)
        up = resample_poly(audio[lo:hi].astype(np.float32), OVERSAMPLE, 1, axis=0)
        a = (start - lo) * OVERSAMPLE
        block = np.abs(
            up[a : a + (end - start) * OVERSAMPLE], out=up[a : a + (end - start) * OVERSAMPLE]
        )
        block = block.reshape(end - start, -1).max(axis=1)  # oversampled points x channels
        peaks[start:end] = np.maximum(block, np.abs(audio[start:end]).max(axis=1))
    return peaks


def true_peak_db(audio: np.ndarray) -> float:
    peak = float(oversampled_peaks(audio).max()) if len(audio) else 0.0
    return 20 * math.log10(peak) if peak > 0 else -math.inf


BLOCK_S = 0.4  # BS.1770 gating block
BLOCK_STEP = 0.25  # 75% overlap
ABSOLUTE_GATE = -70.0
RELATIVE_GATE = -10.0


def _k_weighting(sr: int) -> np.ndarray:
    """BS.1770 K-weighting (high shelf, then high-pass) as second-order sections, using
    pyloudnorm's filter design so results match its meter at any sample rate."""
    from pyloudnorm import IIRfilter
    from scipy.signal import tf2sos

    shelf = IIRfilter(4.0, 1 / np.sqrt(2), 1500.0, sr, "high_shelf")
    highpass = IIRfilter(0.0, 0.5, 38.0, sr, "high_pass")
    return np.vstack([tf2sos(shelf.b, shelf.a), tf2sos(highpass.b, highpass.a)])


def loudness_lufs(audio: np.ndarray, sr: int) -> float | None:
    """Integrated loudness (ITU-R BS.1770-4 / EBU R128, gated), or None for silence or
    audio shorter than one 400 ms block. Vectorized; matches pyloudnorm's meter."""
    from scipy.signal import sosfilt

    audio = _as_2d(audio)
    block = round(BLOCK_S * sr)
    if len(audio) < block or not np.any(audio):
        return None
    # A +/-1e-12 (-240 dBFS) Nyquist-rate dither keeps the filter state out of subnormal
    # floats, which make IIR filtering of long silences ~30x slower. Its energy is far
    # below the -70 LUFS gate.
    signal = audio.astype(np.float64)
    signal[0::2] -= 1e-12
    signal[1::2] += 1e-12
    weighted = sosfilt(_k_weighting(sr), signal, axis=0)
    energy = np.empty(len(audio) + 1)
    energy[0] = 0.0
    np.cumsum(np.einsum("ij,ij->i", weighted, weighted), out=energy[1:])  # L/R gain 1.0
    blocks = round((len(audio) / sr - BLOCK_S) / (BLOCK_S * BLOCK_STEP)) + 1
    starts = (BLOCK_S * BLOCK_STEP * np.arange(blocks) * sr).astype(int)
    ends = np.minimum(starts + block, len(audio))
    z = (energy[ends] - energy[starts]) / block
    with np.errstate(divide="ignore"):
        loud = -0.691 + 10 * np.log10(z)
    gated = z[loud >= ABSOLUTE_GATE]
    if len(gated) == 0:
        return None
    relative = -0.691 + 10 * math.log10(gated.mean()) + RELATIVE_GATE
    gated = z[(loud >= ABSOLUTE_GATE) & (loud > relative)]
    value = -0.691 + 10 * math.log10(gated.mean()) if len(gated) else -math.inf
    return value if math.isfinite(value) else None


def db_to_gain(db: float) -> float:
    return 10 ** (db / 20)


# --- Limiting and mastering --------------------------------------------------------------------


def limit(
    audio: np.ndarray,
    sr: int,
    ceiling_db: float,
    *,
    lookahead_ms: float = 5.0,
    release_ms: float = 80.0,
    block: int = 32,
) -> np.ndarray:
    """Look-ahead peak limiter on true (4x oversampled) peaks.

    Per block of samples: the gain that keeps the block's peaks under the ceiling; a
    minimum over the look-ahead window, then a moving average of the same length, so the
    gain glides down before a peak but never sits above what any sample needs; then an
    exponential release back toward unity that never exceeds that envelope.
    """
    audio = _as_2d(audio)
    n = len(audio)
    if n == 0:
        return audio.astype(np.float32)
    ceiling = db_to_gain(ceiling_db)
    peaks = oversampled_peaks(audio)
    blocks = math.ceil(n / block)
    padded = np.concatenate([peaks, np.zeros(blocks * block - n, dtype=np.float32)])
    block_peak = padded.reshape(blocks, block).max(axis=1)
    need = np.minimum(1.0, ceiling / np.maximum(block_peak, 1e-12))
    window = max(1, round(lookahead_ms / 1000 * sr / block))
    ahead = np.lib.stride_tricks.sliding_window_view(
        np.concatenate([need, np.ones(window)]), window + 1
    ).min(axis=1)[:blocks]
    behind = np.concatenate([np.full(window, ahead[0]), ahead])
    sums = np.concatenate([[0.0], np.cumsum(behind)])
    glide = (sums[window + 1 :] - sums[: -window - 1]) / (window + 1)
    recover = 1 - math.exp(-block / sr / (release_ms / 1000))
    gain = np.empty(blocks)
    current = 1.0
    for i, cap in enumerate(np.minimum(glide, need).tolist()):
        current = min(cap, current + (1.0 - current) * recover)
        gain[i] = current
    per_sample = np.repeat(gain, block)[:n].astype(np.float32)
    return (audio * per_sample[:, None]).astype(np.float32)


@dataclass(frozen=True)
class Mastered:
    audio: np.ndarray
    gain_db: float  # applied before the limiter
    lufs: float | None
    true_peak_db: float


def master(audio: np.ndarray, sr: int, target_lufs: float, ceiling_db: float) -> Mastered:
    """Gain to the loudness target, limit true peaks below the ceiling, and correct the
    gain for what the limiter took (a few rounds). A final trim guarantees the ceiling."""
    measured = loudness_lufs(audio, sr)
    if measured is None:
        return Mastered(_as_2d(audio).astype(np.float32), 0.0, None, true_peak_db(audio))
    gain_db = target_lufs - measured
    limiter_ceiling = ceiling_db - 0.3  # margin for gain changes between blocks
    out = limit(audio * db_to_gain(gain_db), sr, limiter_ceiling)
    for _ in range(4):
        lufs = loudness_lufs(out, sr)
        if lufs is None or abs(lufs - target_lufs) < 0.05:
            break
        gain_db += target_lufs - lufs
        out = limit(audio * db_to_gain(gain_db), sr, limiter_ceiling)
    peak = true_peak_db(out)
    if peak > ceiling_db:
        out = out * np.float32(db_to_gain(ceiling_db - peak - 0.01))
        peak = true_peak_db(out)
    return Mastered(out, gain_db, loudness_lufs(out, sr), peak)
