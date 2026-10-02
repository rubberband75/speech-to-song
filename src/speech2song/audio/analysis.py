"""Checks on a generated backing track against its arrangement (spec stage 6): tempo
(allowing half and double time), key, the energy contour over the sections, and
loudness. Arrays in, numbers out; no file I/O."""

import math
from collections.abc import Sequence

import numpy as np

from speech2song.audio.dsp import loudness_lufs
from speech2song.audio.theory import Key, parse_key, rank_keys
from speech2song.models import TakeAnalysis

ANALYSIS_RATE = 11025  # enough for onsets and chroma, and fast
TEMPO_RATIOS = (0.5, 1.0, 2.0)  # half-time feels are expected in this genre
TEMPO_HOP = 128  # onset frames of 11.6 ms
REFINE = 0.04  # refine the tempogram's guess within ±4% of its beat period
TEMPO_DEADBAND = 0.005  # estimates this close count as on tempo
KEY_PENALTY = {"same": 0.0, "relative": 0.15, "other": 0.3}
ENERGY_FLOOR = 0.5  # flag takes whose section levels barely follow the arrangement
DEAD_DB = 32.0  # a section this far under the take's loudest one is near-silent...
DEAD_PENALTY = 0.5  # ...and costs this much score times its share of the music's time


def _mono(audio: np.ndarray, rate: int) -> np.ndarray:
    import librosa

    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    return librosa.resample(mono.astype(np.float32), orig_sr=rate, target_sr=ANALYSIS_RATE,
                            res_type="soxr_hq")  # fmt: skip


def estimate_tempo(mono: np.ndarray) -> float | None:
    """The track's tempo, precise to a fraction of a BPM.

    librosa's tempogram gives an unbiased first guess, but its bins are about 2% apart
    near 120 BPM (it reads 119 as 120.2). The guess is refined by finding the onset
    envelope's autocorrelation peak within ±REFINE of that beat period, interpolated
    between frames with a parabola.
    """
    import librosa

    if len(mono) < ANALYSIS_RATE * 4 or not np.any(mono):
        return None
    onsets = librosa.onset.onset_strength(y=mono, sr=ANALYSIS_RATE, hop_length=TEMPO_HOP)
    guess = float(np.atleast_1d(librosa.feature.tempo(
        onset_envelope=onsets, sr=ANALYSIS_RATE, hop_length=TEMPO_HOP))[0])  # fmt: skip
    if guess <= 0:
        return None
    env = onsets - onsets.mean()
    ac = np.fft.irfft(np.abs(np.fft.rfft(env, n=2 * len(env))) ** 2)[: len(env)]
    if ac[0] <= 0:
        return guess
    fps = ANALYSIS_RATE / TEMPO_HOP
    period = 60 / guess * fps
    lo, hi = max(2, int(period * (1 - REFINE))), int(period * (1 + REFINE)) + 1
    if hi + 1 >= len(ac):
        return guess
    k = lo + int(np.argmax(ac[lo:hi]))
    y0, y1, y2 = ac[k - 1], ac[k], ac[k + 1]
    curve = y0 - 2 * y1 + y2
    lag = k + (0.5 * (y0 - y2) / curve if curve < 0 else 0.0)
    return 60 * fps / lag


def tempo_fit(estimate: float, target: float) -> tuple[float, float]:
    """(ratio, error): the half/same/double reading closest to the target, and the
    relative error at that reading."""
    ratio = min(TEMPO_RATIOS, key=lambda r: abs(math.log(estimate / (target * r))))
    return ratio, abs(estimate / (target * ratio) - 1)


def estimate_key(mono: np.ndarray) -> tuple[Key, float] | None:
    import librosa

    if not np.any(mono):
        return None
    chroma = librosa.feature.chroma_stft(y=mono, sr=ANALYSIS_RATE).mean(axis=1)
    ranked = rank_keys(chroma)
    return (ranked[0][1], ranked[0][0]) if ranked else None


def key_relation(found: Key, target: Key) -> str:
    if found == target:
        return "same"
    if set(found.scale) == set(target.scale):
        return "relative"
    return "other"


def section_levels_db(audio: np.ndarray, rate: int, spans_s: Sequence[tuple[float, float]]):
    levels = []
    for start, end in spans_s:
        block = audio[round(start * rate) : round(end * rate)]
        rms = float(np.sqrt(np.mean(np.square(block)))) if len(block) else 0.0
        levels.append(20 * math.log10(rms) if rms > 0 else -120.0)
    return levels


def energy_correlation(levels_db: Sequence[float], energies: Sequence[float]) -> float | None:
    if len(levels_db) < 3 or np.std(levels_db) == 0 or np.std(energies) == 0:
        return None
    return float(np.corrcoef(levels_db, energies)[0, 1])


def dead_sections(levels_db: Sequence[float]) -> list[int]:
    """Indexes of the sections (all but the last) that are DEAD_DB or more under the
    loudest one."""
    if len(levels_db) < 2:
        return []
    loudest = max(levels_db)
    return [i for i, level in enumerate(levels_db[:-1]) if level <= loudest - DEAD_DB]


def analyze_take(
    audio: np.ndarray,
    rate: int,
    *,
    take: int,
    bpm: float,
    key: str,
    spans_s: Sequence[tuple[float, float]],
    energies: Sequence[float],
    tolerance_bpm: float,
    expected_s: float,
    section_ids: Sequence[str] | None = None,
) -> TakeAnalysis:
    """Tempo, key, section levels and loudness of a take against the arrangement, with
    flags and a score (1 is best). Sections that came out near-silent (DEAD_DB under the
    loudest; the last one is left out, as the ending may fade) are flagged and cost
    score: the mix can lift a quiet bed, not one the model left empty."""
    mono = _mono(audio, rate)
    flags = []
    tempo = estimate_tempo(mono)
    ratio = error = None
    if tempo is not None:
        ratio, error = tempo_fit(tempo, bpm)
        if error > tolerance_bpm / bpm:
            flags.append(f"tempo {tempo:.1f} BPM is off the {bpm:g} BPM target")
    found = estimate_key(mono)
    relation = None
    if found is not None:
        relation = key_relation(found[0], parse_key(key))
        if relation == "other":
            flags.append(f"key reads as {found[0].name}, not {key}")
    levels = section_levels_db(audio, rate, spans_s)
    correlation = energy_correlation(levels, energies)
    if correlation is not None and correlation < ENERGY_FLOOR:
        flags.append(f"section levels follow the arrangement weakly (r={correlation:.2f})")
    dead = dead_sections(levels)
    if dead:
        names = [section_ids[i] if section_ids else f"#{i + 1}" for i in dead]
        flags.append(f"near-silent where music was asked for: {', '.join(names)} "
                     f"({DEAD_DB:g}+ dB under the loudest section)")  # fmt: skip
    seconds = len(audio) / rate
    if abs(seconds - expected_s) > 1.0:
        flags.append(f"{seconds:.1f} s long; the arrangement is {expected_s:.1f} s")
    score = 1.0
    excess = max(0.0, (error or 0.0) - TEMPO_DEADBAND)
    score -= 0.4 * min(1.0, excess / max(tolerance_bpm / bpm - TEMPO_DEADBAND, 1e-9))
    score -= KEY_PENALTY.get(relation or "other", 0.3)
    score -= 0.3 * (1 - max(0.0, correlation if correlation is not None else 0.0))
    total = sum(b - a for a, b in spans_s)
    if dead and total > 0:
        score -= DEAD_PENALTY * sum(spans_s[i][1] - spans_s[i][0] for i in dead) / total
    return TakeAnalysis(
        take=take,
        seconds=round(seconds, 3),
        tempo_bpm=None if tempo is None else round(tempo, 2),
        tempo_ratio=ratio,
        tempo_error=None if error is None else round(error, 4),
        key=None if found is None else found[0].name,
        key_relation=relation,  # type: ignore[arg-type]
        energy_correlation=None if correlation is None else round(correlation, 3),
        section_levels_db=[round(v, 1) for v in levels],
        lufs=None if (lufs := loudness_lufs(audio, rate)) is None else round(lufs, 2),
        flags=flags,
        score=round(score, 3),
    )
