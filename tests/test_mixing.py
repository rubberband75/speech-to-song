"""Mixing on arrays: clip placement, the exact speech stem, tails, the music's shaping
and gaps, the melody layer."""

import numpy as np
import pytest

from speech2song.arrangement import ClipInfo, build_arrangement, default_parts
from speech2song.audio.mixing import (
    ENERGY_RAMP_S,
    TAIL_MAX_S,
    apply_entry_fix,
    audible_end,
    bar_levels_db,
    energy_gains,
    gain_curve,
    gap_lift,
    late_entries,
    melody_notes,
    place_clips,
    ring_out,
    section_spans,
    speech_bus,
    speech_stem,
    tail_limits,
)
from speech2song.config import load_preset
from speech2song.models import Arrangement, BarChord, ClipMelody, Melody, MelodyNote, PlacedClip

from .conftest import PRESETS_DIR
from .fixtures.synth import SR, noise_burst

PRESET = load_preset("cinematic_future_bass", PRESETS_DIR)
BPM = 120.0  # a beat is 0.5 s, a bar 2 s


def _phrase(cid: str, bars: int = 1) -> ClipMelody:
    notes = [
        MelodyNote(start_beat=b, beats=1.0, midi=60 + b, pitch=60 + b, speech_pitch=48,
                   start_s=b / 2, end_s=b / 2 + 0.4, velocity=80)
        for b in range(3)
    ]  # fmt: skip
    chord = BarChord(bar=0, degree=1, name="Am", pitch_classes=[9, 0, 4])
    return ClipMelody(clip_id=cid, file=f"clips/{cid}.wav", duration_s=1.5, bars=bars,
                      voiced_ratio=0.5, median_speech_pitch=48, notes=notes,
                      chords=[chord])  # fmt: skip


def _setup() -> tuple[Arrangement, Melody, dict[str, np.ndarray]]:
    clips = [ClipInfo("a", "hook", "a", 1.5, 1.2), ClipInfo("b", "build", "b", 1.5, 1.2),
             ClipInfo("c", "outro", "c", 1.5, 1.2)]  # fmt: skip
    melody = Melody(key="A minor", tonic=9, mode="minor", key_source="detected",
                    tuning_offset=0, bpm=BPM, tempo_cost=0, grid="1/8", snap_strength=0.8,
                    octave_shift=0, loop_phrase_count=3, instrument="soft_piano",
                    main_clip="a", clips=[_phrase(c.id) for c in clips])  # fmt: skip
    arrangement = build_arrangement(default_parts(PRESET, clips), clips, melody, PRESET)
    audio = {c.id: np.stack([noise_burst(1.5, seed=i), noise_burst(1.5, seed=i + 9)], axis=1)
             for i, c in enumerate(clips)}  # fmt: skip
    return arrangement, melody, audio


def _placed(arrangement: Arrangement, audio: dict[str, np.ndarray]) -> list[PlacedClip]:
    return place_clips(arrangement, {k: f"clips/{k}.wav" for k in audio},
                       {k: len(v) for k, v in audio.items()}, SR)  # fmt: skip


def test_clips_land_on_their_section_starts() -> None:
    arrangement, _, audio = _setup()
    placed = _placed(arrangement, audio)
    beds = [s for s in arrangement.sections if s.clip_id]
    assert [p.clip_id for p in placed] == ["a", "b", "c"]
    for p, bed in zip(placed, beds, strict=True):
        assert p.start_sample == round(bed.start_bar * 2.0 * SR)
        assert p.end_sample - p.start_sample == len(audio[p.clip_id])
    shifted = arrangement.model_copy(deep=True)
    next(s for s in shifted.sections if s.clip_id == "a").clip_offset_beats = 1.0
    assert _placed(shifted, audio)[0].start_sample == placed[0].start_sample + SR // 2


def test_speech_stem_is_the_clips_verbatim_on_silence() -> None:
    arrangement, _, audio = _setup()
    placed = _placed(arrangement, audio)
    frames = round((arrangement.total_seconds + 2) * SR)
    stem = speech_stem(placed, audio, frames, 2)
    covered = np.zeros(frames, dtype=bool)
    for p in placed:
        assert np.array_equal(stem[p.start_sample : p.end_sample], audio[p.clip_id])
        covered[p.start_sample : p.end_sample] = True
    assert not stem[~covered].any()
    with pytest.raises(ValueError, match="past the end"):
        speech_stem(placed, audio, placed[-1].end_sample - 1, 2)


def test_tails_stop_before_the_next_clip() -> None:
    arrangement, _, audio = _setup()
    placed = _placed(arrangement, audio)
    frames = round((arrangement.total_seconds + 2) * SR)
    limits = tail_limits(placed, frames, SR)
    assert limits[0] == min(placed[1].start_sample - placed[0].end_sample, TAIL_MAX_S * SR)
    bus = speech_bus(placed, audio, frames, SR, gain_db=0, highpass_hz=90, threshold_db=-20,
                     reverb_send=0.5, delay_send=0.5, delay_s=0.375)  # fmt: skip
    for p, limit in zip(placed, limits, strict=True):
        tail_end = p.end_sample + limit
        assert np.abs(bus.wet[p.end_sample : tail_end]).max() > 0  # tails ring...
        if limit < TAIL_MAX_S * SR:  # ...and are silent by the time the next clip starts
            assert np.abs(bus.wet[tail_end - 10 : tail_end]).max() < 1e-3
    outside = np.ones(frames, dtype=bool)
    for p in placed:
        outside[p.start_sample : p.end_sample] = False
    assert not bus.dry[outside].any()  # the processed speech stays inside the clips


def test_melody_layer_modes() -> None:
    arrangement, melody, _ = _setup()
    assert melody_notes(arrangement, melody, "off") == ([], [])
    replay, under = melody_notes(arrangement, melody, "replay")
    assert under == []
    drop = next(s for s in arrangement.sections if s.role == "drop")
    start, stop = drop.start_bar * 2.0, (drop.start_bar + drop.bars) * 2.0
    in_drop = [n for n in replay if start <= n.start_s < stop]
    assert len(in_drop) == 3 * drop.bars  # the 1-bar phrase loops through the section
    assert all(n.end_s <= stop + 1e-9 for n in in_drop)
    assert in_drop[0].start_s == pytest.approx(start)
    _, under = melody_notes(arrangement, melody, "all")
    bed = next(s for s in arrangement.sections if s.clip_id == "a")
    first = [n for n in under if bed.start_bar * 2.0 <= n.start_s < (bed.start_bar + bed.bars) * 2]
    assert len(first) == 3  # played once under its clip
    assert first[0].start_s == pytest.approx(bed.start_bar * 2.0)


# --- The music's shaping ------------------------------------------------------------------


def test_energy_gains_fix_only_large_deviations() -> None:
    levels = [-16.0, -13.0, -26.0, None, -11.0, -10.7, -12.0]  # a near-silent build, a gap
    energies = [0.15, 0.2, 0.8, 0.0, 1.0, 0.3, 0.2]
    gains, targets = energy_gains(levels, energies, range_db=10, tolerance_db=3, max_db=6)
    anchor = float(np.median([-17.5, -15.0, -34.0, -21.0, -13.7, -14.0]))
    assert targets[0] == pytest.approx(anchor + 1.5)
    assert gains[2] == 6.0  # 15 dB too quiet: capped
    assert gains[3] == 0.0 and targets[3] is None  # silent: left to the lift
    assert gains[4] == pytest.approx((anchor + 10) - (-11.0) - 3)  # beyond the tolerance only
    assert gains[0] == 0.0 and gains[1] == 0.0  # close enough: untouched
    assert energy_gains(levels, energies, range_db=10, tolerance_db=3, max_db=0)[0] == [0.0] * 7
    flat, _ = energy_gains([-12.0] * 3, [0.2, 0.6, 1.0], range_db=10, tolerance_db=1, max_db=6)
    assert flat[0] < 0 < flat[2] and flat[1] == 0.0  # an even take gets contrast


def test_gain_curve_ramps_into_each_section() -> None:
    sr = 1000
    curve = gain_curve([(0, 2000), (2000, 4000)], [0.0, 6.0], 4500, sr)
    ramp = round(ENERGY_RAMP_S * sr)
    assert curve[0] == 1.0 and curve[2000 - ramp - 1] == 1.0
    assert curve[2000] == pytest.approx(10 ** (6 / 20)) and curve[-1] == curve[2000]
    assert np.all(np.diff(curve[2000 - ramp : 2001]) >= 0)  # a smooth climb to the downbeat


def _rms(block: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(block))))


def test_a_gap_lifts_into_the_next_section_and_is_never_silent() -> None:
    sr = SR
    rng = np.random.default_rng(1)
    music = (0.1 * rng.standard_normal((6 * sr, 2))).astype(np.float32)
    start, end = 3 * sr, 4 * sr
    lifted = gap_lift(music, start, end, sr, swell=0.8, lift_db=3.0)
    assert lifted.shape == (sr, 2) and lifted.dtype == np.float32
    quarters = [_rms(lifted[i * sr // 4 : (i + 1) * sr // 4]) for i in range(4)]
    assert quarters[0] > 0.05  # the build's tail runs on: no hole
    assert quarters[3] > 1.4 * quarters[0]  # and rises into the downbeat
    swell = lifted - gap_lift(music, start, end, sr, swell=0.0, lift_db=3.0)  # the swell alone
    assert abs(swell[-1]).max() < 1e-3 < abs(swell[-sr // 10 : -sr // 100]).max()  # no click
    plain = gap_lift(music, start, end, sr, swell=0.0, lift_db=0.0)
    np.testing.assert_allclose(plain, music[start:end])  # no lift, no swell: as generated


def test_a_gap_at_the_very_start_or_end_is_safe() -> None:
    sr = 1000
    music = np.ones((3000, 2), dtype=np.float32)
    assert gap_lift(music, 0, 500, sr, swell=0.5, lift_db=3.0).shape == (500, 2)
    assert gap_lift(music, 2800, 3500, sr, swell=0.5, lift_db=3.0).shape == (200, 2)


def _fading_music(sr: int, seconds: float = 12.0, hold: float = 8.0) -> np.ndarray:
    """A chord-like tone that holds for `hold` s, then fades out quickly (-30 dB a second)."""
    t = np.arange(round(seconds * sr)) / sr
    tone = sum(np.sin(2 * np.pi * f * t) for f in (220.0, 277.2, 329.6))
    fade = np.where(t < hold, 1.0, 10 ** (-(t - hold) * 30 / 20))
    return (0.1 * tone * fade)[:, None].repeat(2, axis=1).astype(np.float32)


def test_audible_end_is_where_the_music_stops_being_loud_enough() -> None:
    sr = 8000
    music = _fading_music(sr)
    end = audible_end(music, sr, 4 * sr, within_db=12.0)
    assert end is not None and 8 * sr < end < 9 * sr  # about 12 dB into the fade
    assert audible_end(np.zeros((sr, 2), dtype=np.float32), sr, 0, 12.0) is None
    assert audible_end(music, sr, len(music), 12.0) is None  # nothing to look at


def test_the_last_chord_rings_out_instead_of_stopping() -> None:
    sr = 8000
    music = _fading_music(sr)
    out, at = ring_out(music, sr, 4 * sr, 6.0)
    assert at == audible_end(music, sr, 4 * sr, 12.0)
    assert len(out) >= at + 6 * sr and out.shape[1] == 2
    np.testing.assert_array_equal(out[: at - 1], music[: at - 1])  # untouched up to there
    plain = _rms(music[at : at + sr])  # the generated fade, a second after the stop point
    ring = [_rms(out[at + i * sr : at + (i + 1) * sr]) for i in range(6)]
    assert ring[1] > 2 * plain  # it rings on where the generated music has died away
    assert ring[0] > ring[2] > ring[4] > ring[5]  # and dies away
    assert ring[0] < 1.2 * _rms(music[at - 2 * sr : at]) and ring[-1] < ring[0] / 4
    assert abs(out[at + 6 * sr - 1]).max() < 1e-3  # ends at silence, never a cut
    same, where = ring_out(music, sr, 4 * sr, 0.0)  # off
    assert same is music and where == len(music)
    silent = np.zeros_like(music)
    assert ring_out(silent, sr, 0, 6.0)[1] == len(silent)


def test_a_ring_out_that_runs_past_the_end_extends_the_music() -> None:
    sr = 8000
    music = _fading_music(sr, seconds=9.0, hold=99.0)  # still at full level at the end
    out, at = ring_out(music, sr, 0, 4.0)
    assert at == len(music) and len(out) == len(music) + 4 * sr
    assert _rms(out[len(music) : len(music) + sr]) > 0.01


def test_shape_music_lifts_gaps_and_reports_levels() -> None:
    from speech2song.stages.mix import shape_music

    arrangement, _, _ = _setup()
    sr = 8000
    spans = section_spans(arrangement, sr)
    frames = spans[-1][2] + sr
    rng = np.random.default_rng(2)
    music = (0.1 * rng.standard_normal((frames, 2))).astype(np.float32)  # evenly loud
    shaped, levels, lifted, fixes = shape_music(music, arrangement, PRESET.mix, sr)
    assert fixes == []  # evenly loud: nothing comes in late
    gaps = [(a, b) for s, a, b in spans if s.silent]
    assert lifted == [s.id for s, _, _ in spans if s.silent] and gaps
    start, end = gaps[0]
    first, last = (_rms(shaped[a:b]) for a, b in ((start, start + (end - start) // 4),
                                                  (end - (end - start) // 4, end)))  # fmt: skip
    assert first > 0.5 * _rms(music[start:end])  # not muted: the build runs on...
    assert last > 1.3 * first  # ...and swells into the drop
    by_role = {lv.role: lv for lv in levels}
    assert by_role["drop"].gain_db > 3 and by_role["speech_bed"].gain_db <= 0  # contrast
    assert by_role["gap"].lufs is None and by_role["gap"].gain_db == 0.0


def _late_drop_music(arrangement: Arrangement, sr: int, late: dict[str, tuple[int, int]],
                     riser: bool = False):  # fmt: skip
    """Noise at each section's energy; for a section ID in `late`, (n, spill): its first
    n bars are near-silent (with `riser`, all but the first rise to 10 dB down) and its
    sound runs `spill` bars into the next section."""
    rng = np.random.default_rng(3)
    spans = section_spans(arrangement, sr)
    bar = 2.0 * sr  # 120 BPM
    music = np.zeros((spans[-1][2] + 4 * sr, 2), dtype=np.float32)
    for section, start, end in spans:
        level = 0.002 if section.silent else 0.02 + 0.3 * section.energy
        music[start:end] = level * rng.standard_normal((end - start, 2))
    for section, start, end in spans:
        if section.id in late:
            n, spill = late[section.id]
            music[start : start + round(n * bar)] *= 0.01
            if riser and n > 1:
                music[start + round(bar) : start + round(n * bar)] *= 30.0
            extra = round(spill * bar)
            music[end : end + extra] = 0.32 * rng.standard_normal((extra, 2))
    return music


def test_a_late_drop_after_a_gap_moves_onto_its_downbeat() -> None:
    arrangement, _, _ = _setup()
    sr = 4000
    drops = [s for s in arrangement.sections if s.role == "drop"]
    music = _late_drop_music(
        arrangement, sr, {drops[0].id: (2, 2), drops[1].id: (1, 0)}, riser=True
    )  # a silent bar, then a riser bar: 2 bars late
    fixes = late_entries(music, arrangement, sr, max_bars=4)
    assert [(f.section_id, f.bars, f.mode) for f in fixes] == [
        (drops[0].id, 2, "shift"),  # its spill into the breakdown moves back with it
        (drops[1].id, 1, "fill"),
    ]  # fmt: skip
    out = music
    for fix in fixes:
        out = apply_entry_fix(out, fix, sr)
    spans = {s.id: (a, b) for s, a, b in section_spans(arrangement, sr)}
    for drop in drops:
        levels = bar_levels_db(out, spans[drop.id][0], drop.bars, 2.0 * sr)
        assert max(levels) - min(levels) < 1.0  # full from the first bar
    breakdown_start = spans[drops[0].id][1]
    after = bar_levels_db(out, breakdown_start, 4, 2.0 * sr)
    assert max(after) - min(after) < 1.0  # no drop-level spill over the next section
    assert np.array_equal(out[: spans[drops[0].id][0]], music[: spans[drops[0].id][0]])


def test_soft_openings_and_disabled_fixes_are_left_alone() -> None:
    arrangement, _, _ = _setup()
    sr = 4000
    drop = next(s for s in arrangement.sections if s.role == "drop")
    music = _late_drop_music(arrangement, sr, {drop.id: (6, 0)})  # a deliberate soft start
    assert late_entries(music, arrangement, sr, max_bars=4) == []
    music = _late_drop_music(arrangement, sr, {drop.id: (2, 0)})
    assert late_entries(music, arrangement, sr, max_bars=0) == []
    spans = {s.id: (a, b) for s, a, b in section_spans(arrangement, sr)}
    soft = _late_drop_music(arrangement, sr, {})
    start = spans[drop.id][0]
    soft[start : start + 2 * sr] *= 0.3  # one bar 10 dB down: a soft start, not a late one
    assert late_entries(soft, arrangement, sr, max_bars=4) == []
