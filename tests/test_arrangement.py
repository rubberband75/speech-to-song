"""Arrangement logic: grouping clips into the arc's speech slots, bar math, chords, the
melody-layer plan, checks on Claude's arcs and on hand-edited arrangements."""

import pytest

from speech2song.arrangement import (
    ClipInfo,
    arrangement_warnings,
    bar_energies,
    bar_seconds,
    bed_bars,
    build_arrangement,
    check_parts,
    default_parts,
    energy_bounds,
    group_clips,
    timeline_rows,
    timeline_strip,
    used_slots,
    validate_arrangement,
)
from speech2song.config import load_preset
from speech2song.models import ArcPart, BarChord, ClipMelody, Melody

from .conftest import PRESETS_DIR

PRESET = load_preset("cinematic_future_bass", PRESETS_DIR)
BPM = 120.0  # a bar is 2 s


def _clip(cid: str, spoken: float, role: str = "build", duration: float | None = None):
    return ClipInfo(cid, role, f"text of {cid}", duration or spoken + 0.3, spoken)


def _phrase(cid: str, chords: list[str]) -> ClipMelody:
    return ClipMelody(
        clip_id=cid, file=f"clips/{cid}.wav", duration_s=4.0, bars=len(chords),
        voiced_ratio=0.5, median_speech_pitch=50.0, notes=[],
        chords=[BarChord(bar=i, degree=1, name=c, pitch_classes=[0, 3, 7])
                for i, c in enumerate(chords)],
    )  # fmt: skip


def _melody(clips: list[ClipInfo], main: str | None = None) -> Melody:
    phrases = [_phrase(c.id, ["Am", "F", "C", "G"] if i == 0 else ["Dm", "Am"])
               for i, c in enumerate(clips)]  # fmt: skip
    return Melody(key="A minor", tonic=9, mode="minor", key_source="detected", tuning_offset=0,
                  bpm=BPM, tempo_cost=0.1, grid="1/8", snap_strength=0.8, octave_shift=12,
                  loop_phrase_count=3, instrument="soft_piano", main_clip=main or clips[0].id,
                  clips=phrases)  # fmt: skip


def test_bed_fits_the_last_word_plus_the_tail_and_the_whole_file() -> None:
    assert bar_seconds(BPM) == 2.0
    assert bed_bars(_clip("a", 5.0), BPM, tail_beats=2) == 3  # 10 + 2 = 12 beats
    assert bed_bars(_clip("a", 5.1), BPM, tail_beats=2) == 4  # 10.2 + 2 = 12.2 beats
    assert bed_bars(_clip("a", 3.9, duration=3.95), BPM, tail_beats=0) == 2  # 7.9 beats
    assert bed_bars(_clip("a", 3.9, duration=4.2), BPM, tail_beats=0) == 3  # the file is longer
    assert bed_bars(_clip("a", 0.1), BPM, tail_beats=0) == 1


def test_grouping_balances_speech_and_keeps_the_order() -> None:
    clips = [_clip("c2", 6, "hook"), _clip("c1", 15), _clip("c3", 9), _clip("c4", 9),
             _clip("c5", 10, "outro")]  # fmt: skip
    groups = group_clips(clips, 3)
    assert [[c.id for c in g] for g in groups] == [["c2", "c1"], ["c3", "c4"], ["c5"]]
    groups = group_clips(clips, 2)
    assert [[c.id for c in g] for g in groups] == [["c2", "c1"], ["c3", "c4", "c5"]]


def _ids(groups: list[list[ClipInfo]]) -> list[list[str]]:
    return [[c.id for c in group] for group in groups]


def test_role_hints_pull_the_hook_first_and_the_outro_last() -> None:
    # Balance alone would put the middle clip with its neighbour on the other side.
    assert _ids(group_clips([_clip("a", 6), _clip("c", 4), _clip("b", 12)], 2)) == [
        ["a", "c"], ["b"]]  # fmt: skip
    assert _ids(group_clips([_clip("a", 6), _clip("c", 4, "outro"), _clip("b", 12)], 2)) == [
        ["a"], ["c", "b"]]  # fmt: skip
    assert _ids(group_clips([_clip("b", 12), _clip("a", 4), _clip("c", 6)], 2)) == [
        ["b"], ["a", "c"]]  # fmt: skip
    assert _ids(group_clips([_clip("b", 12), _clip("a", 4, "hook"), _clip("c", 6)], 2)) == [
        ["b", "a"], ["c"]]  # fmt: skip


def test_fewer_clips_than_slots_keep_the_first_slots_and_the_last() -> None:
    assert used_slots(3, 3) == [0, 1, 2]
    assert used_slots(2, 3) == [0, 2]
    assert used_slots(1, 3) == [0]
    assert used_slots(3, 5) == [0, 1, 4]
    parts = default_parts(PRESET, [_clip("a", 4), _clip("b", 4)])
    roles = [p.role for p in parts]
    assert roles.count("speech_bed") == 2
    assert roles[-2:] == ["speech_bed", "outro"]  # the last slot (before the outro) is kept
    assert [p.clips for p in parts if p.clips] == [["a"], ["b"]]


def test_arrangement_sections_bars_chords_and_melody_plan() -> None:
    clips = [_clip("c2", 6, "hook"), _clip("c1", 15), _clip("c3", 9), _clip("c4", 9),
             _clip("c5", 10, "outro")]  # fmt: skip
    arrangement = build_arrangement(default_parts(PRESET, clips), clips, _melody(clips), PRESET)
    sections = arrangement.sections
    assert [s.role for s in sections][:5] == ["intro", "speech_bed", "speech_bed", "build", "gap"]
    assert [s.clip_id for s in sections if s.clip_id] == ["c2", "c1", "c3", "c4", "c5"]
    bar = 0
    for section in sections:  # contiguous, with one chord per bar
        assert section.start_bar == bar
        assert len(section.chords) == section.bars
        assert section.start_s == pytest.approx(bar * 2.0)
        bar += section.bars
    assert arrangement.total_bars == bar
    assert arrangement.total_seconds == pytest.approx(bar * 2.0)
    intro, bed = sections[0], sections[1]
    assert intro.chords == ["Am", "F", "C", "G", "Am", "F", "C", "G"]  # the main phrase
    assert bed.chords == ["Am", "F", "C", "G"]  # c2 (4 bars) plays its own phrase's chords
    assert sections[2].chords[:2] == ["Dm", "Am"]  # c1's own phrase
    drops = [s for s in sections if s.role == "drop"]
    assert [d.melody_phrase for d in drops] == ["c1", "c4"]  # the line heard just before
    assert next(s for s in sections if s.role == "breakdown").melody_phrase == "c1"
    assert all(s.melody_phrase is None for s in sections if s.role in ("intro", "build"))
    assert next(s for s in sections if s.role == "build").shape == "rise"
    assert validate_arrangement(arrangement, {c.id: c for c in clips}) == []


def test_energy_rises_through_the_gap_and_falls_at_the_end() -> None:
    clips = [_clip("a", 4), _clip("b", 4), _clip("c", 4)]
    arrangement = build_arrangement(default_parts(PRESET, clips), clips, _melody(clips), PRESET)
    bounds = dict(zip([s.id for s in arrangement.sections], energy_bounds(arrangement.sections),
                      strict=True))  # fmt: skip
    build = next(s for s in arrangement.sections if s.role == "build")
    assert bounds[build.id] == (0.6, 1.0)  # rises into the drop after the gap
    outro = arrangement.sections[-1]
    assert bounds[outro.id] == (0.1, 0.0)
    energies = bar_energies(arrangement)
    assert len(energies) == arrangement.total_bars


def test_claude_parts_are_checked() -> None:
    clips = [_clip("a", 4), _clip("b", 4)]
    good = [ArcPart(role="intro", bars=4), ArcPart(role="speech_bed", clips=["a", "b"]),
            ArcPart(role="drop", bars=16), ArcPart(role="outro", bars=8)]  # fmt: skip
    assert check_parts(good, clips, PRESET) == []
    bad = [ArcPart(role="intro", bars=0), ArcPart(role="speech_bed", clips=["b", "a"]),
           ArcPart(role="choir", bars=4), ArcPart(role="drop", bars=40, clips=["a"]),
           ArcPart(role="speech_bed")]  # fmt: skip
    problems = check_parts(bad, clips, PRESET)
    text = "\n".join(problems)
    assert "part 1: intro has 0 bars" in text
    assert "unknown role 'choir'" in text
    assert "only speech_bed parts carry clips" in text
    assert "part 4: drop has 40 bars" in text
    assert "part 5: a speech_bed part needs at least one clip" in text
    assert "in this order: a, b" in text
    assert check_parts([], clips, PRESET) == ["the arc has no parts"]


def test_hand_edited_arrangements_are_validated() -> None:
    clips = [_clip("a", 4), _clip("b", 4)]
    infos = {c.id: c for c in clips}
    arrangement = build_arrangement(default_parts(PRESET, clips), clips, _melody(clips), PRESET)
    gap = arrangement.model_copy(deep=True)
    gap.sections[2].start_bar += 1
    assert any("starts at bar" in p for p in validate_arrangement(gap, infos))
    unknown = arrangement.model_copy(deep=True)
    unknown.sections[1].clip_id = "zz"
    assert any("unknown clip 'zz'" in p for p in validate_arrangement(unknown, infos))
    overlap = arrangement.model_copy(deep=True)
    beds = [s for s in overlap.sections if s.clip_id]
    beds[1].clip_offset_beats = -(beds[1].start_bar - beds[0].start_bar) * 4
    assert any("overlap" in p for p in validate_arrangement(overlap, infos))
    total = arrangement.model_copy(update={"total_bars": 3})
    assert any("total_bars" in p for p in validate_arrangement(total, infos))
    dropped = arrangement.model_copy(deep=True)
    for section in dropped.sections:
        if section.clip_id == "b":
            section.clip_id = None
    assert arrangement_warnings(dropped, ["a", "b"]) == ["the arrangement does not play b"]


def test_timeline_strip_and_rows() -> None:
    clips = [_clip("a", 4), _clip("b", 4)]
    arrangement = build_arrangement(default_parts(PRESET, clips), clips, _melody(clips), PRESET)
    energy, speech = timeline_strip(arrangement)
    assert len(energy) == len("energy │") + arrangement.total_bars + 1
    assert "━" in speech
    narrow = timeline_strip(arrangement, width=20)
    assert "one column = " in narrow[1] and "bars" in narrow[1]
    rows = timeline_rows(arrangement, {"a": "x" * 80})
    assert rows[0][0] == "s1" and rows[0][4] == "intro"
    bed = next(r for r in rows if r[6].startswith("a "))
    assert bed[6].endswith("…")
    assert any("(melody of" in r[6] for r in rows)


def _part(quote: str, n: int, of: int, spoken: float) -> ClipInfo:
    return ClipInfo(f"{quote}-{n}", "build", "x", spoken + 0.3, spoken, quote, n == of)


def test_a_long_quotes_parts_stay_together_with_music_between() -> None:
    parts = [_part("b", 1, 3, 6.0), _part("b", 2, 3, 6.0), _part("b", 3, 3, 6.0)]
    clips = [_clip("a", 3.0), *parts, _clip("c", 3.0)]
    for slots in (2, 3, 5):
        groups = _ids(group_clips(clips, slots))
        holding = [g for g in groups if "b-1" in g]
        assert len(holding) == 1 and "b-1,b-2,b-3" in ",".join(holding[0]), groups
    assert _ids(group_clips(clips, 5)) == [["a"], ["b-1", "b-2", "b-3"], ["c"]]
    # A part before more of its quote gets a bar of music after its last word.
    assert bed_bars(parts[0], BPM, tail_beats=2, part_pause_beats=4) == 4  # 12 + 4 = 16 beats
    assert bed_bars(parts[2], BPM, tail_beats=2, part_pause_beats=4) == 4  # 12 + 2 = 14 beats
    assert bed_bars(parts[0], BPM, tail_beats=2, part_pause_beats=5) == 5


def test_a_proposed_arc_must_keep_a_quotes_parts_together() -> None:
    preset = load_preset("cinematic_future_bass", PRESETS_DIR)
    clips = [_part("b", 1, 2, 6.0), _part("b", 2, 2, 6.0)]
    split = [ArcPart(role="speech_bed", clips=["b-1"]), ArcPart(role="drop", bars=4),
             ArcPart(role="speech_bed", clips=["b-2"])]  # fmt: skip
    assert any("parts of quote b" in p for p in check_parts(split, clips, preset))
    together = [ArcPart(role="speech_bed", clips=["b-1", "b-2"]), ArcPart(role="drop", bars=4)]
    assert check_parts(together, clips, preset) == []
