"""Speech pitch: pYIN tracking (librosa), contour clean-up, and note segmentation.

Everything after `track_pitch` is pure and works on arrays of fractional MIDI pitch
(NaN for unvoiced frames), so it can be tested with synthetic contours.
"""

import itertools
from dataclasses import dataclass

import numpy as np

ANALYSIS_SR = 22050
FRAME = 1024
HOP = 256  # 11.6 ms
FMIN_HZ, FMAX_HZ = 60.0, 500.0  # covers low male to high child speech


@dataclass
class PitchTrack:
    midi: np.ndarray  # fractional MIDI pitch per frame, NaN when unvoiced
    level_db: np.ndarray  # frame RMS level, dBFS
    hop_s: float

    def time(self, frame: int) -> float:
        return frame * self.hop_s


@dataclass(frozen=True)
class Segment:
    """A stretch of voiced speech that becomes one melody note."""

    start: float  # seconds from the clip start
    end: float
    pitch: float  # median fractional MIDI pitch of its voiced frames
    level_db: float  # peak frame level
    word: str | None


def hz_to_midi(hz: np.ndarray | float) -> np.ndarray:
    return 69 + 12 * np.log2(np.asarray(hz, dtype=np.float64) / 440.0)


def midi_to_hz(midi: np.ndarray | float) -> np.ndarray:
    return 440.0 * 2 ** ((np.asarray(midi, dtype=np.float64) - 69) / 12)


def track_pitch(mono: np.ndarray, sr: int) -> PitchTrack:
    """pYIN F0 per frame (via librosa), cleaned with `clean_contour`."""
    import librosa

    y = mono.astype(np.float32)
    if sr != ANALYSIS_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=ANALYSIS_SR)
    f0, voiced, _ = librosa.pyin(
        y, fmin=FMIN_HZ, fmax=FMAX_HZ, sr=ANALYSIS_SR, frame_length=FRAME, hop_length=HOP
    )
    rms = librosa.feature.rms(y=y, frame_length=FRAME, hop_length=HOP)[0]
    n = min(len(f0), len(rms))
    midi = np.full(n, np.nan)
    ok = voiced[:n] & np.isfinite(f0[:n])
    midi[ok] = hz_to_midi(f0[:n][ok])
    level = 20 * np.log10(rms[:n] + 1e-9)
    return PitchTrack(clean_contour(midi), level, HOP / ANALYSIS_SR)


def voiced_runs(midi: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) frame ranges of consecutive voiced (non-NaN) frames."""
    voiced = np.isfinite(midi).astype(np.int8)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], voiced, [0]])))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))


def median_smooth(midi: np.ndarray, kernel: int = 5) -> np.ndarray:
    """Median filter inside each voiced run (never across unvoiced gaps)."""
    out = midi.copy()
    half = kernel // 2
    for start, end in voiced_runs(midi):
        run = midi[start:end]
        for i in range(len(run)):
            out[start + i] = np.median(run[max(0, i - half) : i + half + 1])
    return out


def fix_octave_jumps(
    midi: np.ndarray, window: int = 43, threshold: float = 7.0, accept: float = 4.0
) -> np.ndarray:
    """Move frames that sit about an octave off the local median back by 12 semitones.

    Frames more than `threshold` semitones from the median of the surrounding
    `window` frames are shifted by an octave if that brings them within `accept`
    semitones; otherwise they are treated as unvoiced.
    """
    out = midi.copy()
    half = window // 2
    for i in np.flatnonzero(np.isfinite(midi)):
        around = midi[max(0, i - half) : i + half + 1]
        reference = np.nanmedian(around)
        if abs(midi[i] - reference) <= threshold:
            continue
        shifted = min((midi[i] - 12, midi[i] + 12), key=lambda p: abs(p - reference))
        out[i] = shifted if abs(shifted - reference) <= accept else np.nan
    return out


def clean_contour(midi: np.ndarray) -> np.ndarray:
    return median_smooth(fix_octave_jumps(midi))


def _split_at_dips(level_db: np.ndarray, start: int, end: int, dip_db: float) -> list[int]:
    """Frame indices inside [start, end) where the level dips `dip_db` below the
    loudest point on both sides (syllable boundaries within a word)."""
    cuts = []
    segment_start = start
    for i in range(start + 1, end - 1):
        if level_db[i] > level_db[i - 1] or level_db[i] > level_db[i + 1]:
            continue  # not a local minimum
        left = level_db[segment_start:i].max()
        right = level_db[i + 1 : end].max()
        if level_db[i] <= min(left, right) - dip_db:
            cuts.append(i)
            segment_start = i
    return cuts


def segment_notes(
    track: PitchTrack,
    words: list[tuple[str, float, float]],
    *,
    min_note_s: float = 0.07,
    max_gap_frames: int = 3,
    dip_db: float = 6.0,
    min_voiced: float = 0.5,
) -> list[Segment]:
    """Notes from voiced stretches inside each word (clip-relative times).

    A word splits at unvoiced gaps longer than `max_gap_frames` and at level dips of
    `dip_db` (roughly syllables). Pieces shorter than `min_note_s` or less than
    `min_voiced` voiced are dropped. Without words, the whole track is one word.
    """
    n = len(track.midi)
    spans = words or [(None, 0.0, n * track.hop_s)]
    voiced = np.isfinite(track.midi)
    segments: list[Segment] = []
    for word, start_s, end_s in spans:
        lo = max(0, int(np.floor(start_s / track.hop_s)))
        hi = min(n, int(np.ceil(end_s / track.hop_s)))
        if hi - lo < 2:
            continue
        pieces = []
        piece_start, gap = None, 0
        for i in range(lo, hi):
            if voiced[i]:
                if piece_start is None:
                    piece_start = i
                gap = 0
            elif piece_start is not None:
                gap += 1
                if gap > max_gap_frames:
                    pieces.append((piece_start, i - gap + 1))
                    piece_start, gap = None, 0
        if piece_start is not None:
            pieces.append((piece_start, hi - gap))
        for a, b in pieces:
            bounds = [a, *_split_at_dips(track.level_db, a, b, dip_db), b]
            for s, e in itertools.pairwise(bounds):
                if (e - s) * track.hop_s < min_note_s:
                    continue
                frames = track.midi[s:e]
                if np.isfinite(frames).mean() < min_voiced:
                    continue
                segments.append(
                    Segment(
                        start=round(track.time(s), 4),
                        end=round(track.time(e), 4),
                        pitch=float(np.nanmedian(frames)),
                        level_db=float(track.level_db[s:e].max()),
                        word=word,
                    )
                )
    segments.sort(key=lambda seg: seg.start)
    return segments
