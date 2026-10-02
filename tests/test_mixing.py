"""Mixing on arrays: clip placement, the exact speech stem, tails, the music's shaping
and gaps, the melody layer."""

import numpy as np
import pytest

from speech2song.arrangement import ClipInfo, build_arrangement, default_parts
from speech2song.audio.dsp import loudness_lufs
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
    for p, bed in zip(placed, beds, strict=True):  # after the passage's lead-in bar
        assert p.start_sample == round((bed.start_bar * 2.0 + bed.clip_offset_beats / 2) * SR)
        assert p.end_sample - p.start_sample == len(audio[p.clip_id])
    assert beds[0].clip_offset_beats == 4
    shifted = arrangement.model_copy(deep=True)
    next(s for s in shifted.sections if s.clip_id == "a").clip_offset_beats = 5.0
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
    assert len(first) == 3  # played once under its clip, from where the clip starts
    assert first[0].start_s == pytest.approx(bed.start_bar * 2.0 + bed.clip_offset_beats / 2)


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


# --- M7: passages, stops, endings --------------------------------------------------------------


def _bed_song(treatment: str = "under") -> tuple[Arrangement, list]:
    """intro 4 bars, then a passage of two quotes (a lead-in bar, then the words), then a
    build: 120 BPM, so a bar is 2 s."""
    from speech2song.audio.mixing import WordSpan

    clips = [ClipInfo("a", "hook", "a", 3.3, 3.0), ClipInfo("b", "build", "b", 3.3, 3.0)]
    melody = Melody(key="A minor", tonic=9, mode="minor", key_source="detected",
                    tuning_offset=0, bpm=BPM, tempo_cost=0, grid="1/8", snap_strength=0.8,
                    octave_shift=0, loop_phrase_count=3, instrument="soft_piano",
                    main_clip="a", clips=[_phrase(c.id) for c in clips])  # fmt: skip
    from speech2song.models import ArcPart

    parts = [ArcPart(role="intro", bars=4),
             ArcPart(role="speech_bed", clips=["a", "b"] if treatment == "under" else ["a"],
                     treatment=treatment),
             ArcPart(role="build", bars=4)]  # fmt: skip
    if treatment != "under":
        parts.insert(2, ArcPart(role="speech_bed", clips=["b"]))
    arrangement = build_arrangement(parts, clips, melody, PRESET, ending="fade")
    words = []
    for section in arrangement.sections:
        if section.clip_id:
            start = round((section.start_bar * 2 + section.clip_offset_beats / 2) * SR)
            words += [WordSpan(section.clip_id, section.id, "w", start + i * SR // 2,
                               start + i * SR // 2 + SR // 3) for i in range(5)]  # fmt: skip
    return arrangement, words


def _tone(frames: int, level: float, sr: int = SR, freq: float = 440.0) -> np.ndarray:
    t = np.arange(frames) / sr
    tone = level * np.sin(2 * np.pi * freq * t) + level * 0.5 * np.sin(2 * np.pi * 1.5 * freq * t)
    return np.stack([tone, tone], axis=1).astype(np.float32)


def test_a_passage_is_set_once_at_its_bar_lines_and_only_as_far_as_needed() -> None:
    from speech2song.audio.mixing import PASSAGE_RAMP_S, passage_curve, passage_levels

    arrangement, words = _bed_song()
    spans = section_spans(arrangement, SR)
    frames = spans[-1][2]
    speech = np.zeros((frames, 2), dtype=np.float32)
    for w in words:
        speech[w.start : w.end] = _tone(w.end - w.start, 0.3, freq=300.0)
    loud = _tone(frames, 0.1, freq=500.0)  # about 10 dB under the words
    levels = passage_levels(loud, speech, arrangement, words, SR, margin_db=14, max_cut_db=-9)
    assert len(levels) == 1 and levels[0].sections == ["s2", "s3"]  # one passage, two quotes
    level = levels[0]
    assert level.start == spans[1][1] and level.end == spans[2][2]  # bar line to bar line
    assert -6 < level.gain_db < -2 and level.gain_db == pytest.approx(
        level.margin_db - 14, abs=0.01
    )
    quiet = loud * np.float32(0.1)  # a bed the model made quiet: left alone
    assert passage_levels(quiet, speech, arrangement, words, SR, margin_db=14,
                          max_cut_db=-9)[0].gain_db == 0  # fmt: skip
    curve = passage_curve(levels, frames, SR)
    db = 20 * np.log10(curve)
    first, last = words[0].start, max(w.end for w in words)
    assert db[: level.start].max() == 0  # nothing moves before the passage's bar line
    assert db[level.start + round(PASSAGE_RAMP_S * SR) : first].max() == pytest.approx(
        level.gain_db, abs=1e-3
    )  # settled during the lead-in bar, before the first word
    assert np.ptp(db[first:last]) < 1e-3  # steady under the whole passage: no pumping
    assert db[level.end :].max() == 0 and db[last + 1000] == pytest.approx(level.gain_db, abs=1e-3)


def test_a_passage_without_a_lead_in_settles_just_before_its_first_word() -> None:
    from speech2song.audio.mixing import PassageLevel, passage_curve

    level = PassageLevel(["s2"], 1000, 9000, (1200, 8000), 5.0, -9.0)
    db = 20 * np.log10(passage_curve([level], 10000, 1000))  # 1 kHz: 0.5 s ramps = 500
    assert db[1100] == pytest.approx(-9, abs=1e-3)  # in place 0.1 s before the first word
    assert db[600] == 0  # the ramp began in the section before (an M5 bed has no lead-in)
    assert db[8100] == pytest.approx(-9, abs=1e-3) and db[8700] > -9 and db[9000] == 0


def test_a_passage_played_alone_stops_the_music_and_swells_it_back() -> None:
    from speech2song.audio.mixing import alone_break

    sr = 8000
    arrangement, _ = _bed_song("alone")
    spans = section_spans(arrangement, sr)
    alone = next((a, b) for s, a, b in spans if s.rest)
    frames = spans[-1][2]
    music = _tone(frames, 0.2, sr)
    music[alone[0] : alone[1]] = 0  # the take stage spliced it open
    last_word = alone[1] - sr  # a second before the next downbeat
    out = alone_break(music, alone[0], alone[1], last_word, sr, ring_s=2.5, swell=0.8)
    np.testing.assert_array_equal(out[: alone[0]], music[: alone[0]])  # untouched before
    np.testing.assert_array_equal(out[alone[1] :], music[alone[1] :])  # and after
    ring = [_rms(out[alone[0] + i * sr // 4 : alone[0] + (i + 1) * sr // 4]) for i in range(10)]
    assert ring[0] > 0.02 and ring[0] > ring[3] > ring[6] and ring[9] < ring[0] / 10  # dies away
    assert _rms(out[alone[0] + round(2.6 * sr) : last_word]) < 1e-4  # the speaker alone
    swell = out[last_word : alone[1]]
    assert _rms(swell[-sr // 4 :]) > 4 * _rms(swell[: sr // 4])  # grows into the downbeat
    assert abs(swell[-1]).max() < 0.05  # and never clicks
    crowded = alone_break(music, alone[0], alone[1], alone[1] - sr // 10, sr, ring_s=2.5,
                          swell=0.8)  # fmt: skip
    assert _rms(crowded[alone[1] - sr // 2 : alone[1]]) < 1e-4  # no room after the words


def test_a_natural_ending_is_left_alone_and_an_abrupt_one_rings() -> None:
    from speech2song.audio.mixing import ends_abruptly, settled_at

    sr = 8000
    held = _fading_music(sr, seconds=12.0, hold=99.0)  # still sounding at the end
    assert ends_abruptly(held, sr, 4 * sr, len(held))
    decayed = _fading_music(sr, seconds=12.0, hold=8.0)  # -30 dB a second from 8 s
    assert not ends_abruptly(decayed, sr, 4 * sr, len(decayed))
    assert 9 * sr < settled_at(decayed, sr, 4 * sr) < 10.5 * sr  # 50 dB down
    assert not ends_abruptly(np.zeros((sr, 2), dtype=np.float32), sr, 0, sr)
    faded = decayed.copy()
    faded[10 * sr :] = 1e-4  # an ending section that is quiet throughout: not a "loud" frame
    assert settled_at(faded, sr, 10 * sr) == 10 * sr


def test_endings_ring_out_of_the_last_music_that_is_heard() -> None:
    from speech2song.audio.mixing import live_start

    sr = 8000
    music = np.concatenate(
        [_tone(4 * sr, 0.2, sr), _tone(4 * sr, 0.2, sr), _tone(4 * sr, 0.0005, sr)]
    )  # a closing section the model left dead
    spans = [(0, 4 * sr), (4 * sr, 8 * sr), (8 * sr, 12 * sr)]
    reference = loudness_lufs(music, sr)
    assert live_start(music, spans, sr, reference) == 4 * sr  # not the dead last section
    alive = np.concatenate([music[: 8 * sr], _tone(4 * sr, 0.05, sr)])  # quieter, but heard
    assert live_start(alive, spans, sr, loudness_lufs(alive, sr)) == 8 * sr
    assert live_start(music, [], sr, reference) == 0


def test_the_take_is_spliced_open_for_passages_played_alone() -> None:
    from speech2song.arrangement import music_bars
    from speech2song.stages.generate_music import conform, splice_rests

    sr = 8000
    arrangement, _ = _bed_song("alone")
    bar = 2 * sr
    music_frames = sum(s.bars for s in arrangement.sections if not s.rest) * bar
    music = np.stack([np.arange(music_frames)] * 2, axis=1).astype(np.float32) + 1  # traceable
    frames = (arrangement.total_bars + 1) * bar
    song = splice_rests(music, arrangement, frames, sr)
    starts = music_bars(arrangement)
    for section, first in zip(arrangement.sections, starts, strict=True):
        a, b = section.start_bar * bar, (section.start_bar + section.bars) * bar
        if section.rest:
            assert not song[a:b].any()
        else:  # the music of this section, exactly as generated (away from the edge fades)
            np.testing.assert_array_equal(
                song[a + 100 : b - 100],
                music[first * bar + 100 : (first + section.bars) * bar - 100],
            )
    rest = next(s for s in arrangement.sections if s.rest)
    stop = rest.start_bar * bar
    assert song[stop - 1, 0] < music[stop - 1, 0] * 0.5  # a 5 ms fade where the music stops
    plain, _ = _bed_song()
    np.testing.assert_array_equal(
        splice_rests(music, plain, frames, sr), conform(music, frames, sr)
    )  # nothing played alone
    held = plain.model_copy(update={"ending": "held_chord"})  # generated 20 s too long
    end = held.total_bars * bar
    longer = np.concatenate([music, music])
    song = splice_rests(longer, held, frames, sr)
    np.testing.assert_array_equal(song[: end - 100], longer[: end - 100])
    assert not song[end:].any()  # cut on the last bar line; the mix rings out the chord
