"""Arrangement logic: grouping clips into the arc's speech slots, bar math, chords, the
melody-layer plan, checks on Claude's arcs and on hand-edited arrangements."""

import pytest

from speech2song.arrangement import (
    ClipInfo,
    arc_schema,
    arrangement_warnings,
    bar_energies,
    bar_seconds,
    bed_bars,
    build_arrangement,
    check_parts,
    default_parts,
    energy_bounds,
    group_clips,
    music_bars,
    music_total_bars,
    passages,
    timeline_rows,
    timeline_strip,
    used_slots,
    validate_arrangement,
)
from speech2song.config import load_preset
from speech2song.models import (
    ArcPart,
    Arrangement,
    BarChord,
    ClipMelody,
    Melody,
    MelodyNote,
)

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
    assert intro.chords == ["Am", "Am", "F", "F", "Am", "Am", "F", "F"]  # i-VI, 2 bars each
    # c2 opens its passage: a lead-in bar (holding its first chord), then its own chords
    assert bed.clip_offset_beats == 4 and bed.bars == 5  # 4 + max(12.6, 12 + 2) = 18 beats
    assert bed.chords == ["Am", "Am", "F", "C", "G"]
    assert sections[2].clip_offset_beats == 0 and sections[2].chords[:2] == ["Dm", "Am"]
    drops = [s for s in sections if s.role == "drop"]
    assert [d.melody_phrase for d in drops] == ["c1", "c4"]  # the line heard just before
    assert next(s for s in sections if s.role == "breakdown").melody_phrase == "c1"
    assert all(s.melody_phrase is None for s in sections if s.role in ("intro", "build"))
    assert next(s for s in sections if s.role == "build").shape == "rise"
    assert arrangement.plan_version == 2 and arrangement.ending == "held_chord"
    outro = sections[-1]  # it holds steady and comes home: the mix rings that chord out
    assert outro.role == "outro" and outro.shape == "flat" and outro.chords[-1] == "Am"
    assert validate_arrangement(arrangement, {c.id: c for c in clips}) == []


def test_energy_rises_through_the_gap_and_falls_at_the_end() -> None:
    clips = [_clip("a", 4), _clip("b", 4), _clip("c", 4)]
    arrangement = build_arrangement(default_parts(PRESET, clips), clips, _melody(clips), PRESET)
    bounds = dict(zip([s.id for s in arrangement.sections], energy_bounds(arrangement.sections),
                      strict=True))  # fmt: skip
    build = next(s for s in arrangement.sections if s.role == "build")
    assert bounds[build.id] == (0.6, 1.0)  # rises into the drop after the gap
    fade = build_arrangement(default_parts(PRESET, clips), clips, _melody(clips), PRESET,
                             ending="fade")  # fmt: skip
    bounds = dict(zip([s.id for s in fade.sections], energy_bounds(fade.sections), strict=True))
    outro = fade.sections[-1]
    assert outro.role == "outro" and bounds[outro.id] == (0.1, 0.0)
    energies = bar_energies(arrangement)
    assert len(energies) == arrangement.total_bars


def _note(midi: int, start: float, beats: float = 1.0) -> MelodyNote:
    return MelodyNote(start_beat=start, beats=beats, midi=midi, pitch=midi, speech_pitch=midi,
                      start_s=0, end_s=0, velocity=80)  # fmt: skip


def _sung(melody: Melody) -> Melody:
    """Every phrase gets a few notes (so it can condition the music)."""
    clips = [p.model_copy(update={"notes": [_note(69, 0), _note(72, 1), _note(76, 2)]})
             for p in melody.clips]  # fmt: skip
    return melody.model_copy(update={"clips": clips})


def as_m7_alone(arrangement: Arrangement) -> Arrangement:
    """The arrangement as M7 stored passages played alone: no music at all ("alone")."""
    sections = [s.model_copy(update={"treatment": "alone", "energy": 0.0, "styles": []})
                if s.treatment == "break" else s for s in arrangement.sections]  # fmt: skip
    return arrangement.model_copy(update={"sections": sections})


ALONE_PARTS = [ArcPart(role="intro", bars=4), ArcPart(role="speech_bed", clips=["a"]),
               ArcPart(role="build", bars=8), ArcPart(role="speech_bed", clips=["b"],
                                                      treatment="alone"),
               ArcPart(role="drop", bars=16), ArcPart(role="speech_bed", clips=["c"],
                                                      treatment="alone")]  # fmt: skip


def test_alone_passages_keep_a_bed_until_their_last_phrase() -> None:
    clips = [_clip("a", 5), _clip("b", 4), _clip("c", 3, "outro")]
    arrangement = build_arrangement(ALONE_PARTS, clips, _melody(clips), PRESET)
    roles = [(s.role, s.clip_id, s.treatment) for s in arrangement.sections]
    assert roles == [
        ("intro", None, "under"),
        ("speech_bed", "a", "under"),
        ("build", None, "under"),
        ("speech_bed", "b", "break"),
        ("drop", None, "under"),
        ("speech_bed", "c", "break"),
    ]
    a, b, c = (s for s in arrangement.sections if s.clip_id)
    assert a.clip_offset_beats == b.clip_offset_beats == 4  # the music leads into both
    assert b.bars == bed_bars(clips[1], BPM, 2, lead_in_beats=4) == 4  # 4 + 8 + 2 = 14 beats
    assert b.energy == a.energy == 0.2 and b.styles == a.styles and not b.rest
    assert c.chords[-1] == "Am"  # the music runs on under the closing line and comes home
    assert music_bars(arrangement) == [s.start_bar for s in arrangement.sections]  # no splices
    assert music_total_bars(arrangement) == arrangement.total_bars
    assert [[s.clip_id for s in run] for run in passages(arrangement)] == [["a"], ["b"], ["c"]]
    rows = timeline_rows(arrangement, {"b": "the key line"})
    assert rows[3][-1] == "b  the key line  (last phrase alone)"


def test_an_outro_runs_on_into_a_closing_line() -> None:
    clips = [_clip("a", 5), _clip("b", 4, "outro")]
    parts = [ArcPart(role="intro", bars=4), ArcPart(role="speech_bed", clips=["a"]),
             ArcPart(role="outro", bars=8),
             ArcPart(role="speech_bed", clips=["b"], treatment="alone")]  # fmt: skip
    arrangement = build_arrangement(parts, clips, _melody(clips), PRESET, ending="fade")
    outro, closing = arrangement.sections[-2:]
    assert outro.shape == "flat" and closing.shape == "fall"  # it fades under the last line
    held = build_arrangement(parts, clips, _melody(clips), PRESET, ending="held_chord")
    assert held.sections[-2].shape == "flat" and held.sections[-1].chords[-1] == "Am"
    ending_on_outro = build_arrangement(parts[:3], clips[:1], _melody(clips[:1]), PRESET,
                                        ending="fade")  # fmt: skip
    assert ending_on_outro.sections[-1].shape == "fall"


def test_m7_alone_passages_take_no_music_time() -> None:
    clips = [_clip("a", 5), _clip("b", 4), _clip("c", 3, "outro")]
    arrangement = as_m7_alone(build_arrangement(ALONE_PARTS, clips, _melody(clips), PRESET))
    a, b, c = (s for s in arrangement.sections if s.clip_id)
    assert b.rest and c.rest and not a.rest
    build = next(s for s in arrangement.sections if s.role == "build")
    assert energy_bounds(arrangement.sections)[2] == (0.6, 1.0)  # rises through the stop
    assert build.bars == 8
    # Music time leaves the alone passages out: the build runs straight into the drop.
    starts = music_bars(arrangement)
    assert starts[4] == starts[2] + 8 and starts[3] == starts[4]
    assert music_total_bars(arrangement) == arrangement.total_bars - b.bars - c.bars
    rows = timeline_rows(arrangement, {"b": "the key line"})
    assert rows[3][-1] == "b  the key line  (alone)"


def test_sections_develop_and_their_chords_vary() -> None:
    clips = [_clip("a", 4), _clip("b", 4), _clip("c", 4)]
    arrangement = build_arrangement(default_parts(PRESET, clips), clips, _melody(clips), PRESET)
    drops = [s for s in arrangement.sections if s.role == "drop"]
    builds = [s for s in arrangement.sections if s.role == "build"]
    assert "restrained first drop" in drops[0].styles
    assert "final drop and climax of the song" in drops[1].styles
    assert "restrained first drop" not in drops[1].styles
    assert "the biggest build of the song" in builds[1].styles
    assert "the biggest build of the song" not in builds[0].styles
    assert drops[0].chords[:4] != drops[1].chords[:4]  # the second drop is not a repeat
    outro = next(s for s in arrangement.sections if s.role == "outro")
    assert outro.chords[-1] == "Am"  # comes home
    claude = [
        ArcPart(role="intro", bars=4, styles=["a lone music box"]),
        ArcPart(role="speech_bed", clips=["a", "b", "c"], styles=["warm tape hiss"]),
    ]
    parts = build_arrangement(claude, clips, _melody(clips), PRESET).sections
    assert parts[0].styles[-1] == "a lone music box"  # Claude's words come last
    assert all(s.styles[-1] == "warm tape hiss" for s in parts if s.clip_id)


def test_endings() -> None:
    clips = [_clip("a", 4), _clip("b", 4, "outro")]
    parts = [ArcPart(role="intro", bars=4), ArcPart(role="speech_bed", clips=["a", "b"])]
    held = build_arrangement(parts, clips, _melody(clips), PRESET)
    last = held.sections[-1]  # the closing line's bed: it comes home to the tonic
    assert held.ending == "held_chord" and last.clip_id == "b" and last.chords[-1] == "Am"
    stop = build_arrangement(parts, clips, _melody(clips), PRESET, ending="stop")
    assert stop.ending == "stop" and stop.sections[-1].shape == "flat"
    fade = build_arrangement(parts, clips, _melody(clips), PRESET, ending="fade")
    assert fade.ending == "fade" and fade.sections[-1].role == "speech_bed"
    assert fade.sections[-1].shape == "fall"  # the last music fades under the last line


def test_melody_references_answer_each_passage() -> None:
    clips = [_clip("a", 4, "hook"), _clip("b", 4), _clip("c", 4)]
    melody = _sung(_melody(clips, main="b"))
    arrangement = build_arrangement(default_parts(PRESET, clips), clips, melody, PRESET)
    refs = [(s.role, s.melody_ref) for s in arrangement.sections if s.melody_ref]
    # the opening carries the main phrase; the music after a passage answers its last quote
    assert refs == [("intro", "b"), ("build", "a"), ("build", "b"), ("outro", "c")]
    plain = build_arrangement(default_parts(PRESET, clips), clips, _melody(clips), PRESET)
    assert all(s.melody_ref is None for s in plain.sections)  # phrases without notes


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


def test_treatments_and_styles_in_claude_parts_are_checked() -> None:
    clips = [_clip("a", 4), _clip("b", 4), _clip("c", 4), _clip("d", 4)]
    alone = [ArcPart(role="speech_bed", clips=["a", "b"], treatment="alone"),
             ArcPart(role="drop", bars=8, styles=["soaring lead", "angelic choir"]),
             ArcPart(role="speech_bed", clips=["c"], treatment="alone"),
             ArcPart(role="speech_bed", clips=["d"], treatment="alone",
                     styles=["one", "two", "three", "four", "five"])]  # fmt: skip
    text = "\n".join(check_parts(alone, clips, PRESET))
    assert "part 1: a passage played alone holds one quote (it has a, b)" in text
    assert "3 passages are played alone; use at most 2" in text
    assert "part 2: the style 'angelic choir' mentions 'choir'" in text
    assert "part 4: 5 styles; give at most 4" in text
    parted = [
        ClipInfo("q-1", "build", "x", 4.3, 4.0, "q", False),
        ClipInfo("q-2", "build", "y", 4.3, 4.0, "q", True),
    ]
    whole = [ArcPart(role="speech_bed", clips=["q-1", "q-2"], treatment="alone")]
    assert check_parts(whole, parted, PRESET) == []  # one quote in parts may play alone
    schema = arc_schema(["intro", "speech_bed"], ["a"])
    assert schema["required"] == ["parts", "ending", "notes"]
    assert schema["properties"]["parts"]["items"]["required"][-2:] == ["treatment", "styles"]


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
