"""Keys, tuning, snapping, octaves, tempo search, quantizing and chords."""

import numpy as np
import pytest

from speech2song.audio.theory import (
    Key,
    TimedPitch,
    alignment_cost,
    choose_chords,
    diatonic_triads,
    estimate_tuning,
    grid_beats,
    parse_key,
    pitch_class_histogram,
    place_octaves,
    quantize,
    rank_keys,
    search_tempo,
    snap_to_scale,
)


@pytest.mark.parametrize(
    ("text", "name"),
    [("C", "C major"), ("Cm", "C minor"), ("c# minor", "C# minor"), ("Bb maj", "Bb major"),
     ("F#m", "F# minor"), ("A min", "A minor"), ("Ebmajor", "Eb major")],
)  # fmt: skip
def test_parse_key(text: str, name: str) -> None:
    assert parse_key(text).name == name


def test_parse_key_rejects_nonsense() -> None:
    with pytest.raises(ValueError, match="not a key"):
        parse_key("H minor")


@pytest.mark.parametrize(
    ("key", "scale", "chords"),
    [
        ("Eb minor", "Eb F Gb Ab Bb Cb Db", "Ebm Fdim Gb Abm Bbm Cb Db"),
        ("F# major", "F# G# A# B C# D# E#", "F# G#m A#m B C# D#m E#dim"),
        ("C major", "C D E F G A B", "C Dm Em F G Am Bdim"),
        ("G minor", "G A Bb C D Eb F", "Gm Adim Bb Cm Dm Eb F"),
    ],
)
def test_spelling_follows_the_key(key: str, scale: str, chords: str) -> None:
    k = parse_key(key)
    assert " ".join(k.spell(pc) for pc in k.scale) == scale
    assert " ".join(t.name for t in diatonic_triads(k)) == chords


def test_tuning_offset_is_a_circular_mean() -> None:
    assert estimate_tuning([60.3, 62.3, 64.28, 65.31], [1, 1, 1, 1]) == pytest.approx(0.3, abs=0.01)
    assert estimate_tuning([59.6, 61.62, 63.58], [1, 1, 1]) == pytest.approx(-0.4, abs=0.02)
    assert estimate_tuning([], []) == 0.0


def _hist(weights: dict[int, float]) -> np.ndarray:
    pitches = [60 + pc for pc in weights]
    return pitch_class_histogram(pitches, list(weights.values()))


def test_key_detection_finds_the_tonic_and_mode() -> None:
    c_major = _hist({0: 5, 2: 2, 4: 4, 5: 2, 7: 4, 9: 2, 11: 1})
    assert rank_keys(c_major)[0][1] == Key(0, "major")
    d_minor = _hist({2: 5, 4: 1, 5: 4, 7: 2, 9: 4, 10: 2, 0: 1})
    assert rank_keys(d_minor)[0][1] == Key(2, "minor")


def test_mode_prior_breaks_near_ties() -> None:
    relative = _hist({0: 3, 2: 2, 4: 3, 5: 2, 7: 3, 9: 3, 11: 1})  # C major / A minor blur
    assert rank_keys(relative, prefer_mode="minor", mode_bonus=0.5)[0][1].mode == "minor"
    assert rank_keys(np.zeros(12)) == []


def test_snapping_strength() -> None:
    c_major = Key(0, "major")
    assert snap_to_scale(61.4, c_major, 0.0) == 61.4
    assert snap_to_scale(61.4, c_major, 1.0) == 62  # C# -> D (nearer than C)
    assert snap_to_scale(61.4, c_major, 0.5) == pytest.approx(61.7)
    assert snap_to_scale(64.0, c_major, 1.0) == 64


def test_octave_placement() -> None:
    shift, placed = place_octaves([42.0, 44.0, 45.0, 30.0])
    assert shift == 24
    assert placed[:3] == [66.0, 68.0, 69.0]
    assert placed[3] == 66.0  # 54 is below the span, folded up an octave
    assert place_octaves([]) == (0, [])


def test_grid() -> None:
    assert grid_beats("1/8") == 0.5
    assert grid_beats("1/16") == 0.25
    assert grid_beats("1/4") == 1.0


def test_alignment_cost_is_zero_on_the_grid() -> None:
    beat = 60 / 100
    assert alignment_cost(100, [4 * beat, 6 * beat], [0.0, beat / 2, beat], 0.5) == 0.0
    assert alignment_cost(100, [4.5 * beat], [], 0.5) == pytest.approx(0.5)


def test_tempo_search_finds_the_fitting_tempo_and_prefers_the_preset() -> None:
    beat = 60 / 111
    durations = [7 * beat, 9 * beat, 12 * beat, 5 * beat]
    onsets = [i * beat / 2 for i in range(12)]
    bpm, cost = search_tempo(114, 6, durations, onsets, 0.5)
    assert bpm == pytest.approx(111, abs=0.25) and cost < 0.02
    assert search_tempo(114, 6, [], [], 0.5)[0] == 114  # nothing to fit: keep the preset


def test_quantize() -> None:
    notes = [  # at 120 BPM a beat is 0.5 s; the grid is an eighth note
        TimedPitch(0.02, 0.27, 60, 80),  # -> beat 0, 0.5 beats
        TimedPitch(0.24, 0.30, 62, 80),  # collides at 0.5? starts at 0.5 beats
        TimedPitch(0.26, 0.90, 64, 80),  # same start, longer: wins
        TimedPitch(0.95, 0.97, 65, 80),  # tiny: at least one grid step
    ]
    out = quantize(notes, 120, 0.5)
    assert [(n.start_beat, n.beats, n.pitch) for n in out] == [
        (0.0, 0.5, 60), (0.5, 1.5, 64), (2.0, 0.5, 65),
    ]  # fmt: skip


def test_overlaps_are_trimmed() -> None:
    out = quantize([TimedPitch(0.0, 1.0, 60, 80), TimedPitch(0.5, 1.0, 62, 80)], 120, 0.5)
    assert [(n.start_beat, n.beats) for n in out] == [(0.0, 1.0), (1.0, 1.0)]


def test_chords_follow_the_melody() -> None:
    c_major = Key(0, "major")
    bars = [_hist({0: 2, 4: 1, 7: 1}), _hist({5: 2, 9: 1, 0: 1}), _hist({7: 2, 11: 1, 2: 1}),
            _hist({0: 2, 4: 2})]  # fmt: skip
    assert [t.name for t in choose_chords(bars, c_major)] == ["C", "F", "G", "C"]


def test_empty_bars_get_a_plausible_progression() -> None:
    chords = choose_chords([np.zeros(12)] * 3, Key(9, "minor"))
    assert chords[0].name == "Am"  # starts on the tonic
    assert choose_chords([], Key(0, "major")) == []
