"""Music theory for the speech melody: keys, scale snapping, tempo, rhythm, chords. Pure."""

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
# Conventional tonic names, and the keys whose signatures use flats.
TONIC_NAMES = {
    "major": ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"],
    "minor": ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"],
}
FLAT_KEYS = {"major": {5, 10, 3, 8, 1}, "minor": {2, 7, 0, 5, 10, 3}}
LETTERS = "CDEFGAB"
LETTER_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
PITCH_CLASS = {
    "C": 0, "B#": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "FB": 4, "F": 5,
    "E#": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11,
    "CB": 11,
}  # fmt: skip
SCALES = {"major": (0, 2, 4, 5, 7, 9, 11), "minor": (0, 2, 3, 5, 7, 8, 10)}  # natural minor
# Krumhansl-Kessler key profiles.
PROFILES = {
    "major": np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]),
    "minor": np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]),
}

Mode = Literal["major", "minor"]


@dataclass(frozen=True)
class Key:
    tonic: int  # pitch class 0-11
    mode: Mode

    @property
    def name(self) -> str:
        return f"{TONIC_NAMES[self.mode][self.tonic]} {self.mode}"

    def spell(self, pitch_class: int) -> str:
        """Note name as this key writes it: scale tones take consecutive letters from the
        tonic (Eb minor: Eb F Gb Ab Bb Cb Db); other notes follow the key signature."""
        pitch_class %= 12
        first = LETTERS.index(TONIC_NAMES[self.mode][self.tonic][0])
        for step, pc in enumerate(self.scale):
            if pc == pitch_class:
                letter = LETTERS[(first + step) % 7]
                offset = (pc - LETTER_PC[letter] + 6) % 12 - 6
                return letter + ("#" * offset if offset > 0 else "b" * -offset)
        names = FLAT_NAMES if self.tonic in FLAT_KEYS[self.mode] else SHARP_NAMES
        return names[pitch_class]

    @property
    def scale(self) -> list[int]:
        return [(self.tonic + step) % 12 for step in SCALES[self.mode]]


_KEY_PATTERN = re.compile(r"^\s*([A-Ga-g])\s*([#b]?)\s*(maj(?:or)?|min(?:or)?|m)?\s*$")


def parse_key(text: str) -> Key:
    """ "C", "C major", "Cm", "c# minor", "Bb maj" ... (no mode means major)."""
    match = _KEY_PATTERN.match(text)
    if not match:
        raise ValueError(f"not a key: {text!r} (try 'C minor', 'F#m', 'Bb major')")
    letter, accidental, mode = match.groups()
    tonic = PITCH_CLASS[(letter + accidental).upper()]
    minor = mode is not None and mode.lower() in ("m", "min", "minor")
    return Key(tonic, "minor" if minor else "major")


def estimate_tuning(pitches: Sequence[float], weights: Sequence[float]) -> float:
    """The speaker's offset from the 12-tone grid, in semitones (-0.5 to 0.5).

    A weighted circular mean of each pitch's distance from the nearest semitone.
    """
    if not len(pitches):
        return 0.0
    angles = 2 * np.pi * np.asarray(pitches)
    w = np.asarray(weights, dtype=np.float64)
    mean = math.atan2(float((w * np.sin(angles)).sum()), float((w * np.cos(angles)).sum()))
    offset = mean / (2 * np.pi)
    return float(offset - round(offset))


def pitch_class_histogram(pitches: Sequence[float], weights: Sequence[float]) -> np.ndarray:
    hist = np.zeros(12)
    for pitch, weight in zip(pitches, weights, strict=True):
        hist[round(float(pitch)) % 12] += weight
    return hist


def rank_keys(
    hist: np.ndarray, prefer_mode: Mode | None = None, mode_bonus: float = 0.05
) -> list[tuple[float, Key]]:
    """All 24 keys by Krumhansl-Schmuckler correlation (plus a bonus for the preferred
    mode), best first."""
    if not hist.any():
        return []
    ranked = []
    for mode, profile in PROFILES.items():
        for tonic in range(12):
            score = float(np.corrcoef(hist, np.roll(profile, tonic))[0, 1])
            if mode == prefer_mode:
                score += mode_bonus
            ranked.append((score, Key(tonic, mode)))  # type: ignore[arg-type]
    ranked.sort(key=lambda item: -item[0])
    return ranked


def snap_to_scale(pitch: float, key: Key, strength: float) -> float:
    """Move a pitch toward the nearest scale tone: 0 keeps it, 1 lands on the tone."""
    base = math.floor(pitch)
    candidates = [base + d for d in range(-2, 4) if (base + d) % 12 in key.scale]
    nearest = min(candidates, key=lambda q: (abs(q - pitch), q))
    return pitch + strength * (nearest - pitch)


def place_octaves(
    pitches: Sequence[float],
    *,
    center: tuple[int, int] = (60, 72),
    span: tuple[int, int] = (55, 84),
    shift: int | None = None,
) -> tuple[int, list[float]]:
    """Shift the whole melody by octaves so its median lands in `center` (or by `shift`,
    e.g. the shift another set of phrases of the same speaker took), then fold any note
    outside `span` back by octaves. Returns (overall shift, pitches)."""
    if not len(pitches):
        return shift or 0, []
    if shift is None:
        median = float(np.median(pitches))
        shift = 12 * math.ceil((center[0] - median) / 12)
    shifted = [p + shift for p in pitches]
    folded = []
    for p in shifted:
        while p < span[0]:
            p += 12
        while p > span[1]:
            p -= 12
        folded.append(p)
    return shift, folded


def grid_beats(quantize_grid: str) -> float:
    """'1/8' -> 0.5 beats (in 4/4, a beat is a quarter note)."""
    return 4 / int(quantize_grid.split("/")[1])


def _frac_distance(values: np.ndarray) -> np.ndarray:
    """Distance of each value to the nearest integer (0 to 0.5)."""
    return np.abs(values - np.round(values))


def alignment_cost(
    bpm: float, durations: Sequence[float], onsets: Sequence[float], grid: float
) -> float:
    """How badly clips and syllables sit on the grid at `bpm` (0 = perfectly).

    Clips start on a bar, so each clip's end should land near a beat; syllable onsets
    should land near the quantize grid. Both are measured in grid units (0 to 0.5).
    """
    beat = 60.0 / bpm
    end_error = float(_frac_distance(np.asarray(durations) / beat).mean()) if durations else 0.0
    onset_error = (
        float(_frac_distance(np.asarray(onsets) / (beat * grid)).mean()) if onsets else 0.0
    )
    return end_error + 0.5 * onset_error


DRIFT_PENALTY = 0.04  # cost of moving to the edge of the tolerance range


def search_tempo(
    bpm: float,
    tolerance: float,
    durations: Sequence[float],
    onsets: Sequence[float],
    grid: float,
    step: float = 1.0,
) -> tuple[float, float]:
    """(best bpm, its alignment cost): multiples of `step` within bpm +/- tolerance.

    Moving away from the preset tempo costs a little (DRIFT_PENALTY at the edge of the
    range), so a different tempo has to fit the clips clearly better to win. The default
    step keeps tempos whole: music generators hold "120 BPM" more reliably than 119.5.
    """
    first = math.ceil((bpm - tolerance) / step - 1e-9) * step
    candidates = np.arange(first, bpm + tolerance + 1e-9, step)
    if len(candidates) == 0:
        candidates = np.array([round(bpm / step) * step])
    costs = [alignment_cost(c, durations, onsets, grid) for c in candidates]
    total = [
        cost + DRIFT_PENALTY * abs(c - bpm) / max(tolerance, 1e-9)
        for c, cost in zip(candidates, costs, strict=True)
    ]
    best = min(range(len(candidates)), key=lambda i: (round(total[i], 6), abs(candidates[i] - bpm)))
    return round(float(candidates[best]), 2), costs[best]


@dataclass(frozen=True)
class TimedPitch:
    start: float  # seconds
    end: float
    pitch: float
    velocity: int


@dataclass(frozen=True)
class GridNote:
    start_beat: float
    beats: float
    pitch: float
    velocity: int
    source: int  # index of the TimedPitch it came from


def quantize(notes: Sequence[TimedPitch], bpm: float, grid: float) -> list[GridNote]:
    """Snap starts and ends to the grid (at least one grid step long); when two notes
    land on the same start, the longer one wins; overlaps are trimmed."""
    beat = 60.0 / bpm
    placed: dict[float, GridNote] = {}
    for index, note in enumerate(notes):
        start = round(note.start / beat / grid) * grid
        end = round(note.end / beat / grid) * grid
        candidate = GridNote(start, max(grid, end - start), note.pitch, note.velocity, index)
        if start not in placed or candidate.beats > placed[start].beats:
            placed[start] = candidate
    ordered = [placed[k] for k in sorted(placed)]
    trimmed = []
    for current, following in zip(ordered, [*ordered[1:], None], strict=True):
        if following is not None and current.start_beat + current.beats > following.start_beat:
            current = GridNote(current.start_beat, following.start_beat - current.start_beat,
                               current.pitch, current.velocity, current.source)  # fmt: skip
        trimmed.append(current)
    return trimmed


@dataclass(frozen=True)
class Triad:
    degree: int  # 1-7
    root: int  # pitch class
    quality: Literal["major", "minor", "diminished"]
    root_name: str  # spelled for the key, e.g. "Db" in Eb minor

    @property
    def pitch_classes(self) -> tuple[int, int, int]:
        third = 4 if self.quality == "major" else 3
        fifth = 6 if self.quality == "diminished" else 7
        return self.root, (self.root + third) % 12, (self.root + fifth) % 12

    @property
    def name(self) -> str:
        suffix = {"major": "", "minor": "m", "diminished": "dim"}[self.quality]
        return self.root_name + suffix


def diatonic_triads(key: Key) -> list[Triad]:
    scale = key.scale
    triads = []
    for i in range(7):
        root, third, fifth = scale[i], scale[(i + 2) % 7], scale[(i + 4) % 7]
        span3, span5 = (third - root) % 12, (fifth - root) % 12
        quality = "major" if span3 == 4 else ("diminished" if span5 == 6 else "minor")
        triads.append(Triad(i + 1, root, quality, key.spell(root)))  # type: ignore[arg-type]
    return triads


DEGREE_PRIOR = {
    "major": {1: 0.3, 4: 0.2, 5: 0.2, 6: 0.15, 2: 0.05, 3: 0.05, 7: 0.0},
    "minor": {1: 0.3, 6: 0.2, 4: 0.15, 7: 0.15, 3: 0.1, 5: 0.1, 2: 0.0},
}
COMMON_MOVES = {
    "major": {(1, 4), (1, 5), (1, 6), (4, 5), (4, 1), (5, 1), (5, 6), (6, 4), (2, 5),
              (6, 2), (3, 6), (4, 2)},
    "minor": {(1, 6), (1, 4), (1, 7), (6, 7), (7, 1), (4, 5), (5, 1), (6, 3), (3, 7),
              (4, 1), (7, 3), (6, 4)},
}  # fmt: skip
MOVE_BONUS = 0.15
STAY_BONUS = 0.05


def choose_chords(bars: Sequence[np.ndarray], key: Key) -> list[Triad]:
    """One diatonic triad per bar (Viterbi): cover the bar's melody pitch classes
    (duration-weighted), favour common degrees and common progressions, and start on
    the tonic when the melody allows."""
    if not bars:
        return []
    triads = diatonic_triads(key)
    prior = DEGREE_PRIOR[key.mode]
    moves = COMMON_MOVES[key.mode]

    def emission(weights: np.ndarray, triad: Triad, first: bool) -> float:
        total = weights.sum()
        fit = 0.0
        if total > 0:
            fit = sum(weights[pc] for pc in triad.pitch_classes) / total
            fit += 0.3 * weights[triad.root] / total
        return fit + prior[triad.degree] + (0.2 if first and triad.degree == 1 else 0.0)

    score = [emission(bars[0], t, True) for t in triads]
    back: list[list[int]] = []
    for weights in bars[1:]:
        step_scores, step_back = [], []
        for t in triads:
            options = [
                score[j]
                + (MOVE_BONUS if (p.degree, t.degree) in moves else 0.0)
                + (STAY_BONUS if p.degree == t.degree else 0.0)
                for j, p in enumerate(triads)
            ]
            j = int(np.argmax(options))
            step_scores.append(options[j] + emission(weights, t, False))
            step_back.append(j)
        score, back = step_scores, [*back, step_back]
    path = [int(np.argmax(score))]
    for step_back in reversed(back):
        path.append(step_back[path[-1]])
    return [triads[i] for i in reversed(path)]


_CHORD_PATTERN = re.compile(r"^([A-G])(#{1,2}|b{1,2})?(m|dim)?$")


def parse_chord(name: str) -> tuple[int, int, int]:
    """Pitch classes (root, third, fifth) of a triad written as `Triad.name` writes it:
    "Ebm", "Cb", "F#dim"."""
    match = _CHORD_PATTERN.match(name.strip())
    if not match:
        raise ValueError(f"not a chord: {name!r} (try 'C', 'Ebm', 'Bdim')")
    letter, accidental, quality = match.groups()
    accidental = accidental or ""
    root = (LETTER_PC[letter] + accidental.count("#") - accidental.count("b")) % 12
    third = 3 if quality in ("m", "dim") else 4
    fifth = 6 if quality == "dim" else 7
    return root, (root + third) % 12, (root + fifth) % 12


# --- Section progressions (M7) ----------------------------------------------------------------

# Progressions per section role, as scale degrees, one entry per chord. Minor uses the
# natural minor triads (i ii° III iv v VI VII), major the plain diatonic ones. Builds lean
# on the chords that pull home (VII, v / V), outros come home to the tonic, and the
# ending is the tonic alone.
PROGRESSIONS: dict[str, dict[str, list[tuple[int, ...]]]] = {
    "minor": {
        "intro": [(1, 6), (1, 6, 3, 7), (6, 1)],
        "build": [(6, 7, 4, 5), (4, 6, 7, 7), (6, 4, 7, 7)],
        "drop": [(1, 6, 3, 7), (6, 7, 1, 1), (1, 7, 6, 7), (6, 3, 7, 1)],
        "breakdown": [(6, 3, 7, 1), (4, 1, 6, 7), (1, 3, 6, 7)],
        "outro": [(6, 7, 1, 1), (4, 6, 1, 1)],
        "ending": [(1,)],
    },
    "major": {
        "intro": [(1, 4), (1, 5, 6, 4), (4, 1)],
        "build": [(4, 5, 6, 5), (2, 4, 5, 5), (6, 4, 5, 5)],
        "drop": [(1, 5, 6, 4), (6, 4, 1, 5), (4, 1, 5, 6), (1, 6, 4, 5)],
        "breakdown": [(6, 4, 1, 5), (4, 6, 5, 1), (1, 3, 4, 5)],
        "outro": [(4, 5, 1, 1), (4, 1, 4, 1)],
        "ending": [(1,)],
    },
}
BARS_PER_CHORD = {"intro": 2, "outro": 2, "build": 1, "drop": 1, "breakdown": 1, "ending": 1}
REPEAT_PENALTY = 0.5  # the same progression as this role's previous section


def progression_fit(degrees: Sequence[int], key: Key, weights: np.ndarray) -> float:
    """How well a progression's chords hold a melody's pitch classes (0-1): the mean
    share of the melody's duration that each chord covers."""
    total = float(weights.sum())
    if total <= 0:
        return 0.0
    triads = diatonic_triads(key)
    return sum(sum(weights[pc] for pc in triads[d - 1].pitch_classes) / total
               for d in degrees) / len(degrees)  # fmt: skip


def section_progression(
    role: str, key: Key, bars: int, melody_weights: np.ndarray, avoid: tuple[int, ...] | None
) -> tuple[tuple[int, ...], list[str]] | None:
    """(degrees, one chord name per bar) for a music section of `role`, or None when the
    role has no vocabulary (the caller keeps its own chords). Of the role's progressions,
    the one that best holds the nearby speech melody's pitch classes wins, unless it is
    the one the role's previous section used (`avoid`), so repeated sections differ."""
    options = PROGRESSIONS[key.mode].get(role)
    if not options or bars < 1:
        return None

    def score(degrees: tuple[int, ...]) -> float:
        repeat = REPEAT_PENALTY if degrees == avoid else 0.0
        return progression_fit(degrees, key, melody_weights) - repeat

    best = max(options, key=score)  # ties keep the vocabulary's order
    triads = diatonic_triads(key)
    per_chord = BARS_PER_CHORD.get(role, 1)
    names = [triads[best[(bar // per_chord) % len(best)] - 1].name for bar in range(bars)]
    if role == "outro" and bars >= 2:  # a closing section lands home
        names[-1] = triads[0].name
    return best, names
