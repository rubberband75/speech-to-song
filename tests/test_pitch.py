"""Pitch tracking, contour clean-up and note segmentation."""

import numpy as np
import pytest

from speech2song.audio.pitch import (
    PitchTrack,
    fix_octave_jumps,
    hz_to_midi,
    median_smooth,
    midi_to_hz,
    segment_notes,
    track_pitch,
    voiced_runs,
)

from .fixtures.synth import SR, speechlike


def _harmonic(hz: np.ndarray) -> np.ndarray:
    phase = 2 * np.pi * np.cumsum(hz) / SR
    return (0.3 * sum(np.sin(h * phase) / h for h in range(1, 6))).astype(np.float32)


def test_hz_midi_round_trip() -> None:
    assert hz_to_midi(440.0) == pytest.approx(69.0)
    assert hz_to_midi(110.0) == pytest.approx(45.0)
    assert midi_to_hz(hz_to_midi(173.2)) == pytest.approx(173.2)


def test_steady_tone_is_tracked_accurately() -> None:
    track = track_pitch(_harmonic(np.full(SR, 110.0)), SR)
    voiced = np.isfinite(track.midi)
    assert voiced.mean() > 0.9
    assert np.median(track.midi[voiced]) == pytest.approx(45.0, abs=0.2)


def test_glide_is_tracked_as_a_rising_contour() -> None:
    track = track_pitch(_harmonic(np.linspace(100, 200, SR)), SR)
    contour = track.midi[np.isfinite(track.midi)]
    assert contour[-5:].mean() - contour[:5].mean() == pytest.approx(12.0, abs=1.5)
    assert np.mean(np.diff(contour) >= -0.05) > 0.95


def test_octave_jumps_are_folded_back() -> None:
    midi = np.full(60, 48.0)
    midi[20:23] = 60.0  # pYIN octave error
    midi[40] = 78.0  # wild outlier: neither close nor an octave off
    fixed = fix_octave_jumps(midi)
    assert np.all(fixed[20:23] == 48.0)
    assert np.isnan(fixed[40])
    assert np.all(fixed[:20] == 48.0)


def test_median_smoothing_stays_inside_voiced_runs() -> None:
    midi = np.array([50, 50, 70, 50, 50, np.nan, 60, 60, 60])
    smoothed = median_smooth(midi, kernel=3)
    assert smoothed[2] == 50  # the spike is removed
    assert np.isnan(smoothed[5])
    assert np.all(smoothed[6:] == 60)  # not pulled toward the earlier run
    assert voiced_runs(midi) == [(0, 5), (6, 9)]


def _track(midi: list[float], levels: list[float]) -> PitchTrack:
    return PitchTrack(np.array(midi, dtype=float), np.array(levels, dtype=float), 0.01)


def test_words_split_at_gaps_and_level_dips() -> None:
    #           word 1: two syllables split by a level dip | gap | word 2
    midi = [50.0] * 20 + [53.0] * 20 + [np.nan] * 10 + [57.0] * 20
    levels = [-20.0] * 18 + [-35.0] * 4 + [-20.0] * 18 + [-80.0] * 10 + [-22.0] * 20
    segments = segment_notes(_track(midi, levels), [("one", 0.0, 0.4), ("two", 0.5, 0.7)])
    assert [(s.word, round(s.pitch)) for s in segments] == [("one", 50), ("one", 53), ("two", 57)]
    assert segments[0].start == 0.0 and segments[2].end == pytest.approx(0.7)


def test_short_or_mostly_unvoiced_pieces_are_dropped() -> None:
    midi = [50.0] * 5 + [np.nan] * 10 + [52.0, np.nan, np.nan] * 10
    segments = segment_notes(_track(midi, [-20.0] * 45), [("w", 0.0, 0.45)])
    assert segments == []  # a 50 ms piece is too short; the rest is only a third voiced


def test_without_words_the_whole_track_is_used() -> None:
    segments = segment_notes(_track([55.0] * 30, [-20.0] * 30), [])
    assert len(segments) == 1 and segments[0].word is None


def test_speechlike_syllables_give_one_note_each() -> None:
    syllables = [(140, 150, 0.3), (180, 170, 0.3), (120, 125, 0.3)]
    audio, times = speechlike(syllables, gap_s=0.15)
    track = track_pitch(audio, SR)
    words = [(f"w{i}", a, b) for i, (a, b) in enumerate(times)]
    segments = segment_notes(track, words)
    assert len(segments) == 3
    for seg, (lo, hi, _) in zip(segments, syllables, strict=True):
        assert seg.pitch == pytest.approx(float(hz_to_midi((lo + hi) / 2)), abs=0.6)
