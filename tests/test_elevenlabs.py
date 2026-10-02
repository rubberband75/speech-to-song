"""Eleven Music: the composition plan, the paid generate stage (with a fake client), and
take analysis and choice. No network."""

import itertools
import json
from collections.abc import Callable
from pathlib import Path

import pytest
import soundfile as sf
import yaml
from typer.testing import Result

from speech2song.arrangement import ClipInfo, build_arrangement, default_parts
from speech2song.backends import music_elevenlabs
from speech2song.backends.music_base import MelodyFiles
from speech2song.config import load_preset
from speech2song.costs import CostLog
from speech2song.manifest import Run
from speech2song.models import ArcPart, Arrangement, TakeChoice, TakeMeta
from speech2song.music_plan import (
    ENDING_TAIL_MS,
    MAX_CHUNK_MS,
    MELODY_SONG,
    MIN_CHUNK_MS,
    build_plan,
    check_plan,
    expand_sections,
    grid_ms,
    inpaint_plan,
    layout_chunks,
    plan_references,
    section_bounds_ms,
    tail_ms,
    total_ms,
)
from speech2song.stages.generate_music import choose_take

from .conftest import PRESETS_DIR, Talk
from .fixtures.fake_claude import clip_answer, response
from .fixtures.fake_elevenlabs import FakeElevenLabs
from .test_arrangement import ALONE_PARTS, _melody, as_m7_alone

Cli = Callable[..., Result]
PRESET = load_preset("cinematic_future_bass", PRESETS_DIR)


def _arrangement(
    parts: list[ArcPart] | None = None, bpm: float = 120.0, ending: str | None = None
) -> Arrangement:
    clips = [ClipInfo("a", "hook", "a", 6.3, 6.0), ClipInfo("b", "build", "b", 9.3, 9.0),
             ClipInfo("c", "outro", "c", 5.3, 5.0)]  # fmt: skip
    melody = _melody(clips).model_copy(update={"bpm": bpm})
    return build_arrangement(
        parts or default_parts(PRESET, clips), clips, melody, PRESET, ending=ending
    )  # type: ignore[arg-type]


# --- The composition plan -----------------------------------------------------------------


def test_plan_tiles_the_song_within_the_api_limits() -> None:
    arrangement = _arrangement()
    plan, layout = build_plan(arrangement, PRESET)
    assert check_plan(plan) == []
    total = section_bounds_ms(arrangement)[-1][2]
    tail = plan["chunks"][-1]  # music past the end, for the model to fade out in
    assert tail["duration_ms"] == ENDING_TAIL_MS == tail_ms(arrangement)
    assert "slowly fading out" in tail["positive_styles"]
    assert sum(c["duration_ms"] for c in plan["chunks"]) == total + ENDING_TAIL_MS
    assert grid_ms(arrangement) == [c["duration_ms"] for c in plan["chunks"]]
    assert tail_ms(_arrangement(ending="fade")) == 0  # a fade is the model's own ending
    assert layout[0].start_ms == 0 and layout[-1].end_ms == total
    for a, b in itertools.pairwise(layout):
        assert a.end_ms == b.start_ms
    assert all(MIN_CHUNK_MS <= c["duration_ms"] <= MAX_CHUNK_MS for c in plan["chunks"])


def test_gaps_fold_silently_into_the_build_and_beds_merge() -> None:
    arrangement = _arrangement()
    plan, layout = build_plan(arrangement, PRESET)
    roles = {s.id: s.role for s in arrangement.sections}
    build = next(i for i, c in enumerate(layout) if c.roles[0] == "build")
    assert [roles[s] for s in layout[build].sections] == ["build", "gap"]
    chunk = plan["chunks"][build]
    assert chunk["text"] == "[Build]"  # a "{near silence}" cue once silenced a whole build
    assert not any("silence" in style for style in chunk["positive_styles"])
    assert "steadily building" in chunk["positive_styles"]
    assert not any(c.roles[0] == "gap" for c in layout)
    adjacent = [c for c in layout if len(c.sections) > 1 and c.roles[0] == "speech_bed"]
    assert all(roles[s] == "speech_bed" for c in adjacent for s in c.sections)


def test_long_silent_sections_fold_too() -> None:
    parts = [ArcPart(role="gap", bars=4), ArcPart(role="intro", bars=4),
             ArcPart(role="gap", bars=4), ArcPart(role="speech_bed", clips=["a", "b", "c"]),
             ArcPart(role="outro", bars=4)]  # fmt: skip
    arrangement = _arrangement(parts)
    plan, layout = build_plan(arrangement, PRESET)
    assert check_plan(plan) == []
    assert [c["text"] for c in plan["chunks"]] == ["[Intro]", "[Ambient Bed]", "[Outro]",
                                                   "[Outro]"]  # fmt: skip
    assert layout[0].sections == ["s1", "s2", "s3"]  # 8 s gaps: still nothing generated
    assert layout[0].start_ms == 0 and layout[0].end_ms == 24_000
    assert grid_ms(arrangement) == [c["duration_ms"] for c in plan["chunks"]]


def test_styles_set_tempo_key_and_keep_it_instrumental() -> None:
    arrangement = _arrangement()
    plan, _ = build_plan(arrangement, PRESET)
    first, later = plan["chunks"][0], plan["chunks"][1]
    for chunk in (first, later):
        assert chunk["positive_styles"][:4] == ["120 BPM", "A minor", "half-time feel",
                                                "instrumental only"]  # fmt: skip
        assert {"vocals", "voiceover", "spoken word"} <= set(chunk["negative_styles"])
        assert chunk["context_adherence"] == "high"
        assert "conditioning_ref" not in chunk
    assert "melodic future bass" in first["positive_styles"]  # global styles open the song
    assert "melodic future bass" not in later["positive_styles"]
    assert later["text"] == "[Ambient Bed]"  # never "speech": no generated voices
    outro, tail = plan["chunks"][-2:]
    assert outro["text"] == tail["text"] == "[Outro]"  # held to the end; the tail is cut away
    assert "fading out to silence at the end" not in outro["positive_styles"]
    assert outro["positive_styles"][-1].endswith("- Am")  # it comes home to the tonic
    faded, _ = build_plan(_arrangement(ending="fade"), PRESET)
    assert "fading out to silence at the end" in faded["chunks"][-1]["positive_styles"]


def _m5() -> tuple[Arrangement, dict]:
    """A plan-version-1 arrangement and the plan the M5 code made for it (fixtures made
    with the code as of commit 13c08f1)."""
    fixtures = Path(__file__).parent / "fixtures"
    arrangement = Arrangement.model_validate_json((fixtures / "m5_arrangement.json").read_text())
    return arrangement, json.loads((fixtures / "m5_plan.json").read_text())


def test_m5_arrangements_keep_their_plan_and_timing() -> None:
    arrangement, expected = _m5()
    assert arrangement.plan_version == 1 and arrangement.ending is None
    plan, _ = build_plan(arrangement, PRESET)
    assert grid_ms(arrangement) == expected["grid_ms"]  # its takes still fit
    for chunk in plan["chunks"]:  # a global negative added to the preset for M7.1
        chunk["negative_styles"].remove("long silences")
    assert plan["chunks"][1:] == expected["plan"]["chunks"][1:]  # no chords, no negatives...
    first, old = plan["chunks"][0], expected["plan"]["chunks"][0]
    assert {k: v for k, v in first.items() if k != "positive_styles"} == \
        {k: v for k, v in old.items() if k != "positive_styles"}  # fmt: skip
    assert first["positive_styles"][:4] == old["positive_styles"][:4]
    assert first["positive_styles"][-5:] == old["positive_styles"][-5:]  # only the preset's
    # global styles (edited for M7) differ


def test_reference_conditions_only_the_first_chunk() -> None:
    arrangement, _ = _m5()  # the M5 melody reference
    plan, _ = build_plan(arrangement, PRESET, reference_song_id="ref_9", reference_ms=45_000,
                         condition_strength="medium")  # fmt: skip
    first = plan["chunks"][0]
    assert first["conditioning_ref"] == {"song_id": "ref_9",
                                         "range": {"start_ms": 0, "end_ms": 30_000}}  # fmt: skip
    assert first["condition_strength"] == "medium"
    assert all("conditioning_ref" not in c for c in plan["chunks"][1:])


def test_m7_chunks_carry_chords_negatives_and_the_development() -> None:
    clips = [ClipInfo("a", "hook", "a", 6.3, 6.0), ClipInfo("b", "build", "b", 9.3, 9.0),
             ClipInfo("c", "outro", "c", 5.3, 5.0)]  # fmt: skip
    melody = _melody(clips).model_copy(update={"bpm": 120.0})
    arrangement = build_arrangement(default_parts(PRESET, clips), clips, melody, PRESET)
    plan, _ = build_plan(arrangement, PRESET)
    assert check_plan(plan) == []
    by_text: dict[str, list[dict]] = {}
    for chunk in plan["chunks"]:
        by_text.setdefault(chunk["text"], []).append(chunk)
    bed = by_text["[Ambient Bed]"][0]
    assert {"drums", "bassline", "lead melody"} <= set(bed["negative_styles"])
    assert "warm sustained pads" in bed["positive_styles"]
    assert any(s.startswith("slow chord changes") for s in bed["positive_styles"])
    first_drop, last_drop = by_text["[Drop]"]
    assert "restrained first drop" in first_drop["positive_styles"]
    assert "final drop and climax of the song" in last_drop["positive_styles"]
    assert "drums" not in first_drop["negative_styles"]
    progressions = [next(s for s in d["positive_styles"] if s.startswith("chord progression"))
                    for d in (first_drop, last_drop)]  # fmt: skip
    assert progressions[0] != progressions[1]
    assert by_text["[Intro]"][0]["positive_styles"].count("chord progression Am - F") == 1


def test_alone_passages_keep_their_bed_in_the_plan() -> None:
    arrangement = _arrangement(ALONE_PARTS)
    plan, _ = build_plan(arrangement, PRESET)
    texts = [c["text"] for c in plan["chunks"]]
    assert texts == [
        "[Intro]",
        "[Ambient Bed]",
        "[Build]",
        "[Ambient Bed]",
        "[Drop]",
        "[Ambient Bed]",
        "[Ambient Bed]",
    ]  # fmt: skip (the last: the ending's tail)
    assert total_ms(plan) == arrangement.total_bars * 2000 + ENDING_TAIL_MS


def test_m7_alone_passages_take_no_music_time() -> None:
    arrangement = as_m7_alone(_arrangement(ALONE_PARTS))
    plan, layout = build_plan(arrangement, PRESET)
    alone = {s.id for s in arrangement.sections if s.rest}
    assert not alone & {s for chunk in layout for s in chunk.sections}
    assert [c["text"] for c in plan["chunks"]] == [
        "[Intro]",
        "[Ambient Bed]",
        "[Build]",
        "[Drop]",
        "[Drop]",
    ]  # fmt: skip (+ tail)
    music = sum(s.bars for s in arrangement.sections if not s.rest) * 2000
    assert total_ms(plan) == music + ENDING_TAIL_MS
    assert section_bounds_ms(arrangement)[-1][2] == music
    build = next(c for c in layout if c.roles[0] == "build")
    drop = next(c for c in layout if c.roles[0] == "drop")
    assert build.end_ms == drop.start_ms  # the build runs straight into its drop
    with pytest.raises(ValueError, match="no music of its own"):
        inpaint_plan(arrangement, PRESET, "song_1", sorted(alone)[:1], None)


def test_quotes_condition_the_music_around_them() -> None:
    from .test_arrangement import _sung

    clips = [ClipInfo("a", "hook", "a", 6.3, 6.0), ClipInfo("b", "build", "b", 9.3, 9.0),
             ClipInfo("c", "outro", "c", 5.3, 5.0)]  # fmt: skip
    melody = _sung(_melody(clips)).model_copy(update={"bpm": 120.0})
    arrangement = build_arrangement(default_parts(PRESET, clips), clips, melody, PRESET)
    assert plan_references(arrangement) == ["a", "b", "c"]
    ranges = {"a": (0, 9000), "b": (9500, 21500), "c": (22000, 30000)}
    plan, _ = build_plan(arrangement, PRESET, melody_ranges=ranges,
                              condition_strength="medium")  # fmt: skip
    conditioned = [(c["text"], c["conditioning_ref"]["range"]) for c in plan["chunks"]
                   if "conditioning_ref" in c]  # fmt: skip
    assert conditioned == [("[Intro]", {"start_ms": 0, "end_ms": 9000}),
                           ("[Build]", {"start_ms": 0, "end_ms": 9000}),
                           ("[Build]", {"start_ms": 9500, "end_ms": 21500}),
                           ("[Outro]", {"start_ms": 22000, "end_ms": 30000})]  # fmt: skip
    assert all(c.get("condition_strength") == "medium" for c in plan["chunks"]
               if "conditioning_ref" in c)  # fmt: skip
    assert all(c["conditioning_ref"]["song_id"] == MELODY_SONG for c in plan["chunks"]
               if "conditioning_ref" in c)  # fmt: skip
    long = {**ranges, "a": (0, 45_000)}
    capped, _ = build_plan(arrangement, PRESET, melody_ranges=long)
    assert capped["chunks"][0]["conditioning_ref"]["range"]["end_ms"] == 30_000
    assert check_plan(plan) == []
    broken = json.loads(json.dumps(plan))
    broken["chunks"][0]["conditioning_ref"]["range"]["end_ms"] = 40_000
    assert any("references must be up to 30000 ms" in p for p in check_plan(broken))


def test_the_upload_holds_only_the_referenced_quote_melodies(tmp_path: Path) -> None:
    from speech2song.backends.music_base import MelodyFiles
    from speech2song.backends.music_elevenlabs import melody_upload
    from speech2song.models import QuoteMelodies

    wav = tmp_path / "quotes.wav"
    wav.write_bytes(b"not read here, only hashed")
    index = QuoteMelodies(
        file="quotes.wav",
        sample_rate=44100,
        instrument="soft_piano",
        clips={"a": (0, 9000), "b": (9500, 50000), "c": (50500, 60000)},
    )
    upload = melody_upload(MelodyFiles(quotes=wav, index=index), ["c", "b", "zz"])
    assert upload is not None
    assert upload["source"] == {"c": [50500, 60000], "b": [9500, 39500]}  # at most 30 s each
    assert upload["ranges"] == {"c": [0, 9500], "b": [10000, 40000]}  # back to back
    assert upload["ms"] == 40000
    assert melody_upload(MelodyFiles(), ["a"]) is None  # a run from before M7
    assert melody_upload(MelodyFiles(quotes=wav, index=index), ["zz"]) is None


def test_long_sections_split_and_a_short_opening_merges_forward() -> None:
    parts = [ArcPart(role="gap", bars=1), ArcPart(role="intro", bars=4),
             ArcPart(role="speech_bed", clips=["a", "b", "c"]),
             ArcPart(role="drop", bars=64), ArcPart(role="outro", bars=4)]  # fmt: skip
    arrangement = _arrangement(parts)
    plan, layout = build_plan(arrangement, PRESET)
    assert check_plan(plan) == []
    assert layout[0].sections[0] == "s1" and layout[0].roles[0] == "intro"  # gap merged forward
    drops = [c for c in layout if c.roles[0] == "drop"]
    assert len(drops) == 2  # 128 s of drop in two chunks
    assert sum(c.duration_ms for c in drops) == 128_000


def test_check_plan_catches_api_violations() -> None:
    bad = {"chunks": [{"duration_ms": 2000, "positive_styles": ["x"] * 51}]}
    problems = check_plan(bad)
    assert any("3,000-120,000" in p for p in problems)
    assert any("more than 50" in p for p in problems)
    assert any("31 chunks" in p for p in check_plan({"chunks": [{"duration_ms": 5000}] * 31}))


def test_layout_covers_every_section_once() -> None:
    arrangement = _arrangement()
    seen = [s for chunk in layout_chunks(arrangement) for s in chunk.sections]
    assert seen == [s.id for s in arrangement.sections]


def test_inpainting_a_range_follows_the_plan_chunks() -> None:
    arrangement = _arrangement()
    ids = [s.id for s in arrangement.sections]
    roles = [s.role for s in arrangement.sections]
    build = roles.index("build")
    drop = roles.index("drop")
    assert expand_sections(arrangement, [f"{ids[build]}-{ids[drop]}"]) == ids[build : drop + 1]
    plan, (start, end) = inpaint_plan(arrangement, PRESET, "song_1", ids[build : drop + 1], None)
    bounds = {s.id: (a, b) for s, a, b in section_bounds_ms(arrangement)}
    assert (start, end) == (bounds[ids[build]][0], bounds[ids[drop]][1])
    generated = [c for c in plan["chunks"] if "song_id" not in c]
    assert [c["text"] for c in generated] == ["[Build]", "[Drop]"]  # the gap rides with the build
    assert generated[0]["duration_ms"] == bounds[ids[build + 1]][1] - bounds[ids[build]][0]
    assert {"steadily building", "rising energy throughout, from moderate to intense"} <= set(
        generated[0]["positive_styles"])  # fmt: skip
    kept = [c["range"] for c in plan["chunks"] if "song_id" in c]
    assert kept[0] == {"start_ms": 0, "end_ms": start}
    assert kept[-1]["end_ms"] == section_bounds_ms(arrangement)[-1][2]
    assert total_ms(plan) == section_bounds_ms(arrangement)[-1][2]


def test_inpainting_refuses_silence_and_scattered_sections() -> None:
    arrangement = _arrangement()
    ids = [s.id for s in arrangement.sections]
    gap = ids[[s.role for s in arrangement.sections].index("gap")]
    with pytest.raises(ValueError, match="no music of its own"):
        inpaint_plan(arrangement, PRESET, "song_1", [gap], None)
    with pytest.raises(ValueError, match="next to each other"):
        expand_sections(arrangement, [ids[0], ids[3]])
    with pytest.raises(ValueError, match="no section 's99'"):
        expand_sections(arrangement, ["s1-s99"])


# --- Take choice ----------------------------------------------------------------------------


def test_take_choice() -> None:
    from speech2song.models import TakeAnalysis

    def analysis(take: int, score: float) -> TakeAnalysis:
        return TakeAnalysis(take=take, seconds=1, tempo_bpm=None, tempo_ratio=None,
                            tempo_error=None, key=None, key_relation=None,
                            energy_correlation=None, score=score)  # fmt: skip

    takes = [analysis(1, 0.4), analysis(2, 0.9)]
    assert choose_take(takes, None) == (2, "best score")
    assert choose_take(takes, 1) == (1, "requested")
    assert choose_take(takes[:1], None) == (1, "only take")


def test_take_analysis_reads_tempo_key_and_energy() -> None:
    from speech2song.audio.analysis import analyze_take
    from speech2song.backends.music_stub import StubBackend, synthesize

    arrangement = _arrangement(ending="fade")
    audio = synthesize(StubBackend().request(arrangement, PRESET, MelodyFiles()), seed=3)
    bar = 2.0
    spans = [(s.start_bar * bar, (s.start_bar + s.bars) * bar) for s in arrangement.sections]
    result = analyze_take(audio, 44100, take=1, bpm=120.0, key="A minor", spans_s=spans,
                          energies=[s.energy for s in arrangement.sections],
                          tolerance_bpm=6, expected_s=arrangement.total_seconds + 2.0)  # fmt: skip
    assert result.tempo_ratio in (0.5, 1.0, 2.0)
    assert result.tempo_error is not None and result.tempo_error < 0.005  # 120 BPM, read exactly
    assert result.key_relation in ("same", "relative")
    assert (result.energy_correlation or 0) > 0.7  # quiet beds, loud drops
    assert result.score > 0.7 and not result.flags
    wrong = analyze_take(audio, 44100, take=2, bpm=97.0, key="F# major", spans_s=spans,
                         energies=[s.energy for s in arrangement.sections],
                         tolerance_bpm=6, expected_s=10.0)  # fmt: skip
    assert wrong.score < result.score
    assert any("off the 97 BPM target" in f for f in wrong.flags)
    assert any("long; the arrangement is 10.0 s" in f for f in wrong.flags)
    ids = [s.id for s in arrangement.sections]
    dead = audio.copy()
    bed = next(i for i, s in enumerate(arrangement.sections) if s.clip_id)
    dead[round(spans[bed][0] * 44100) : round(spans[bed][1] * 44100)] *= 0.001  # -60 dB
    empty = analyze_take(dead, 44100, take=3, bpm=120.0, key="A minor", spans_s=spans,
                         energies=[s.energy for s in arrangement.sections], tolerance_bpm=6,
                         expected_s=arrangement.total_seconds + 2.0, section_ids=ids)  # fmt: skip
    assert any(f.startswith(f"near-silent where music was asked for: {ids[bed]} ")
               for f in empty.flags)  # fmt: skip
    assert empty.score < result.score


# --- The paid generate stage, with a fake client ---------------------------------------------


@pytest.fixture
def eleven(
    cli: Cli, talk: Talk, install: Callable, short_song: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[..., FakeElevenLabs]:
    """A run with clips, melody and arrangement ready, two music takes configured, an
    ElevenLabs key, and a way to install a fake client."""
    install(response(clip_answer(talk[1], [(1, 1), (8, 8)])))
    assert cli("select", "--yes").exit_code == 0
    assert cli("melody").exit_code == 0
    assert cli("arrange").exit_code == 0
    config = Path("config.yaml")
    config.write_text(config.read_text().replace("music_takes: 1", "music_takes: 2"))
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")

    def install_fake(*responses: object) -> FakeElevenLabs:
        fake = FakeElevenLabs(*responses)
        monkeypatch.setattr(music_elevenlabs, "client_factory", lambda key: fake)
        return fake

    return install_fake


def _music(run_dir: Path) -> list[str]:
    return sorted(p.name for p in (run_dir / "06_music").iterdir())


def test_generate_with_elevenlabs(cli: Cli, talk: Talk, eleven: Callable) -> None:
    run = talk[0]
    fake = eleven()
    dry = cli("generate", "--music-backend", "elevenlabs", "--dry-run")
    assert dry.exit_code == 0, dry.output
    assert "music_v2_5" in dry.output and "take 2" in dry.output and fake.compose_calls == []

    result = cli("generate", "--yes")
    assert result.exit_code == 0, result.output
    assert len(fake.compose_calls) == 2
    call = fake.compose_calls[0]
    assert call["model_id"] == "music_v2_5" and call["store_for_inpainting"] is True
    assert call["request_options"]["max_retries"] == 0
    assert "output_format" not in call  # "auto": the API's choice
    assert check_plan(call["composition_plan"]) == []
    meta = TakeMeta.model_validate_json(run.path("06_music/take_001.meta.json").read_text())
    assert (meta.backend, meta.song_id, meta.file) == ("elevenlabs", "song_1",
                                                       "06_music/take_001.mp3")  # fmt: skip
    assert meta.sample_rate == 48000 and meta.channels == 2
    assert meta.grid_ms == [c["duration_ms"] for c in call["composition_plan"]["chunks"]]
    assert json.loads(run.path("06_music/take_001.response.json").read_text())["song_metadata"]
    entries = [e for e in CostLog(run.costs_path).read() if e.service == "elevenlabs"]
    minutes = sum(c["duration_ms"] for c in call["composition_plan"]["chunks"]) / 60_000
    # nothing is uploaded: melody conditioning is off unless the config turns it on
    assert [e.operation for e in entries] == ["music.compose_detailed"] * 2
    assert fake.upload_calls == []
    assert all("conditioning_ref" not in c for c in call["composition_plan"]["chunks"])
    assert entries[0].usd == pytest.approx(minutes * 0.15, abs=1e-6)
    assert entries[0].units == {"minutes": pytest.approx(minutes, abs=1e-4)}
    assert "generate: cached" in cli("generate", "--yes").output  # no second payment
    assert len(fake.compose_calls) == 2

    result = cli("mix")  # picks a take (free), then mixes
    assert result.exit_code == 0, result.output
    choice = TakeChoice.model_validate_json(run.path("06_music/analysis.json").read_text())
    assert choice.reason == "best score" and len(choice.takes) == 2
    assert all(a.current for a in choice.takes)
    selected = sf.info(str(run.path("06_music/selected.wav")))
    arrangement = Arrangement.model_validate_json(run.path("05_arrangement.json").read_text())
    assert selected.samplerate == 44100
    assert selected.duration == pytest.approx(arrangement.total_seconds + 2.0, abs=1e-3)
    gaps = {s.id for s in arrangement.sections if s.silent}
    assert gaps and not gaps & set(choice.takes[0].sections)  # lifted, so not measured


def test_a_failed_take_does_not_repeat_the_paid_one(cli: Cli, talk: Talk, eleven: Callable) -> None:
    from elevenlabs.core.api_error import ApiError

    run = talk[0]
    fake = eleven(None, ApiError(status_code=500, body={"detail": {"message": "boom"}}))
    result = cli("generate", "--music-backend", "elevenlabs", "--yes")
    assert result.exit_code == 1
    assert "ElevenLabs error 500" in result.output and "boom" in result.output
    paid = [e.operation for e in CostLog(run.costs_path).read() if e.service == "elevenlabs"]
    assert paid == ["music.compose_detailed"]
    fake.responses = []
    result = cli("generate", "--yes")
    assert result.exit_code == 0, result.output
    assert "take 1: reusing the paid take" in result.output
    assert len(fake.compose_calls) == 3  # take 1 once, take 2 failed once, then made


def _edit_styles(run_dir_path: Path) -> None:
    data = json.loads(run_dir_path.read_text())
    data["sections"][0]["styles"] = ["a different intro"]
    run_dir_path.write_text(json.dumps(data))


def test_takes_of_an_older_request_stay_while_they_fit(
    cli: Cli, talk: Talk, eleven: Callable
) -> None:
    run = talk[0]
    fake = eleven()
    assert cli("generate", "--music-backend", "elevenlabs", "--yes").exit_code == 0
    _edit_styles(run.path("05_arrangement.json"))  # new music, same timing
    result = cli("generate", "--yes")
    assert result.exit_code == 0, result.output
    assert len(fake.compose_calls) == 4
    assert "take 3: composing" in result.output and "take 4: composing" in result.output
    assert {m.take for m in _metas(run)} == {1, 2, 3, 4}
    assert not (run.path("06_music") / "archive").exists()

    assert cli("mix").exit_code == 0
    choice = TakeChoice.model_validate_json(run.path("06_music/analysis.json").read_text())
    assert [(a.take, a.current) for a in choice.takes] == [
        (1, False), (2, False), (3, True), (4, True)]  # fmt: skip
    assert choice.chosen in (3, 4)  # the newest music by default

    result = cli("mix", "--take", "1")  # free: an older take that still fits
    assert result.exit_code == 0, result.output
    assert "take 1 (older request)" in result.output and "using take 1 (requested)" in result.output
    assert len(fake.compose_calls) == 4
    result = cli("generate", "--yes")
    assert "generate: cached" in result.output and "take 1 is still the one mixed" not in (
        result.output)  # fmt: skip
    assert "using take" in cli("mix", "--take", "auto").output
    choice = TakeChoice.model_validate_json(run.path("06_music/analysis.json").read_text())
    assert choice.chosen in (3, 4) and choice.reason == "best score"
    assert cli("mix", "--take", "7").exit_code == 1
    assert "take 7 is not available" in cli("mix", "--take", "7").output
    assert "give a take number" in cli("mix", "--take", "x").output


def test_takes_that_no_longer_fit_are_archived(
    cli: Cli, talk: Talk, eleven: Callable, short_song: Path
) -> None:
    run = talk[0]
    fake = eleven()
    assert cli("generate", "--music-backend", "elevenlabs", "--yes").exit_code == 0
    preset = short_song / "cinematic_future_bass.yaml"
    data = yaml.safe_load(preset.read_text())
    data["section_roles"]["drop"]["bars"] = 3  # the timing changes
    preset.write_text(yaml.safe_dump(data))
    assert cli("arrange").exit_code == 0
    result = cli("generate", "--yes")
    assert result.exit_code == 0, result.output
    assert len(fake.compose_calls) == 4
    assert {m.take for m in _metas(run)} == {3, 4}  # numbers are never reused
    archived = list((run.path("06_music") / "archive").iterdir())
    assert len(archived) == 1
    assert sorted(p.name for p in archived[0].iterdir()) == [
        "take_001.meta.json", "take_001.mp3", "take_001.response.json",
        "take_002.meta.json", "take_002.mp3", "take_002.response.json"]  # fmt: skip
    assert cli("mix").exit_code == 0


def _metas(run: Run) -> list[TakeMeta]:
    return [TakeMeta.model_validate_json(p.read_text())
            for p in sorted(run.path("06_music").glob("take_*.meta.json"))]  # fmt: skip


def test_copyright_rejection_is_explained(cli: Cli, talk: Talk, eleven: Callable) -> None:
    from elevenlabs.core.api_error import ApiError

    run = talk[0]
    suggestion = {"chunks": [{"text": "[Intro]", "duration_ms": 5000}]}
    error = ApiError(status_code=400, body={"detail": {
        "status": "bad_composition_plan", "message": "copyrighted style",
        "data": {"composition_plan_suggestion": suggestion}}})  # fmt: skip
    eleven(error)
    result = cli("generate", "--music-backend", "elevenlabs", "--yes")
    assert result.exit_code == 1
    assert "rejected the plan (bad_composition_plan" in result.output
    saved = json.loads(run.path("06_music/plan_suggestion.json").read_text())
    assert saved["suggestion"] == suggestion
    paid = [e.operation for e in CostLog(run.costs_path).read() if e.service == "elevenlabs"]
    assert paid == []  # no take


def test_missing_key_and_confirmation(
    cli: Cli, talk: Talk, eleven: Callable, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = eleven()
    result = cli("generate", "--music-backend", "elevenlabs")  # no --yes, no terminal
    assert result.exit_code == 1 and "--yes" in result.output
    monkeypatch.delenv("ELEVENLABS_API_KEY")
    result = cli("generate", "--yes")
    assert result.exit_code == 1 and "ELEVENLABS_API_KEY" in result.output
    assert fake.compose_calls == []


def test_quote_melodies_are_uploaded_once_when_conditioning_is_on(
    cli: Cli, talk: Talk, eleven: Callable, short_song: Path
) -> None:
    run = talk[0]
    config = Path("config.yaml")
    config.write_text(config.read_text() + "elevenlabs:\n  melody_conditioning: true\n")
    fake = eleven()
    result = cli("generate", "--music-backend", "elevenlabs", "--yes")
    assert result.exit_code == 0, result.output
    assert len(fake.upload_calls) == 1
    first = fake.compose_calls[0]["composition_plan"]["chunks"][0]
    assert first["conditioning_ref"]["song_id"] == "ref_1"
    assert first["condition_strength"] == "low"
    entries = [e for e in CostLog(run.costs_path).read() if e.service == "elevenlabs"]
    assert [e.operation for e in entries] == ["music.upload", "music.compose_detailed",
                                              "music.compose_detailed"]  # fmt: skip
    assert entries[0].note.endswith("c2")  # the quotes the plan is conditioned on
    preset = short_song / "cinematic_future_bass.yaml"
    data = yaml.safe_load(preset.read_text())
    data["section_roles"]["drop"]["styles"] = ["bigger drop"]
    preset.write_text(yaml.safe_dump(data))
    assert cli("generate", "--yes").exit_code == 0
    assert len(fake.upload_calls) == 1  # the same reference audio is not uploaded again


def test_force_adds_fresh_takes(cli: Cli, talk: Talk, eleven: Callable) -> None:
    run = talk[0]
    fake = eleven()
    assert cli("generate", "--music-backend", "elevenlabs", "--yes").exit_code == 0
    result = cli("generate", "--force", "--yes")
    assert result.exit_code == 0, result.output
    assert "making new takes" in result.output
    assert len(fake.compose_calls) == 4
    assert [(m.take, m.song_id) for m in _metas(run)] == [
        (1, "song_1"), (2, "song_2"), (3, "song_3"), (4, "song_4")]  # fmt: skip
    assert cli("mix").exit_code == 0
    choice = TakeChoice.model_validate_json(run.path("06_music/analysis.json").read_text())
    assert len(choice.takes) == 4 and all(a.current for a in choice.takes)  # same request


def test_regenerate_one_section(cli: Cli, talk: Talk, eleven: Callable) -> None:
    run = talk[0]
    fake = eleven()
    assert cli("generate", "--music-backend", "elevenlabs", "--yes").exit_code == 0
    assert cli("mix").exit_code == 0
    chosen = TakeChoice.model_validate_json(run.path("06_music/analysis.json").read_text())
    arrangement = Arrangement.model_validate_json(run.path("05_arrangement.json").read_text())
    drop = next(s for s in arrangement.sections if s.role == "drop")

    dry = cli("regenerate", "--section", drop.id, "--dry-run")
    assert dry.exit_code == 0, dry.output
    assert "priced as the whole song" in dry.output and len(fake.compose_calls) == 2

    result = cli("regenerate", "--section", drop.id, "--note", "less busy", "--yes")
    assert result.exit_code == 0, result.output
    plan = fake.compose_calls[-1]["composition_plan"]
    original = f"song_{chosen.chosen}"
    kept = [c for c in plan["chunks"] if "song_id" in c]
    generated = [c for c in plan["chunks"] if "song_id" not in c]
    assert len(generated) == 1 and generated[0]["positive_styles"][0] == "less busy"
    assert generated[0]["text"] == "[Drop]"
    assert all(c["song_id"] == original for c in kept)
    assert kept[0]["range"]["start_ms"] == 0
    meta = TakeMeta.model_validate_json(
        run.path(f"06_music/take_{chosen.chosen:03d}.meta.json").read_text())  # fmt: skip
    assert meta.file == f"06_music/take_{chosen.chosen:03d}_v2.mp3"
    assert meta.song_id == "song_3"
    assert meta.params["history"][0]["song_id"] == original
    assert meta.params["edits"][0]["section"] == drop.id
    assert "▶ take" in result.output and "▶ mix" in result.output  # remixed with the new take
    assert "▶ generate" not in result.output
    stages = [e.stage for e in CostLog(run.costs_path).read() if e.service == "elevenlabs"]
    assert stages == ["generate", "generate", "regenerate"]  # two takes

    bad = cli("regenerate", "--section", "s99", "--yes")
    assert bad.exit_code == 1 and "no section 's99'" in bad.output
    assert "not both" in cli("regenerate", "--yes").output

    result = cli("regenerate", "--undo")  # free: back to the first version
    assert result.exit_code == 0, result.output
    path = run.path(f"06_music/take_{chosen.chosen:03d}.meta.json")
    meta = TakeMeta.model_validate_json(path.read_text())
    assert meta.file == f"06_music/take_{chosen.chosen:03d}.mp3" and meta.song_id == original
    assert meta.params["history"] == [] and meta.params["undone"][0]["section"] == drop.id
    assert "▶ mix" in result.output and len(fake.compose_calls) == 3
    assert "no earlier version" in cli("regenerate", "--undo").output
    assert cli("regenerate", "--section", drop.id, "--yes").exit_code == 0
    meta = TakeMeta.model_validate_json(path.read_text())
    assert meta.file == f"06_music/take_{chosen.chosen:03d}_v3.mp3"  # v2 stays on disk
    assert run.path(f"06_music/take_{chosen.chosen:03d}_v2.mp3").exists()


def test_regenerate_a_range_of_an_older_take(cli: Cli, talk: Talk, eleven: Callable) -> None:
    """The live case: the plan's wording changed since the take was made (so `generate`
    would compose new takes), but its timing still fits; regenerating build to drop
    must not touch `generate`."""
    run = talk[0]
    fake = eleven()
    assert cli("generate", "--music-backend", "elevenlabs", "--yes").exit_code == 0
    assert cli("mix", "--take", "1").exit_code == 0
    _edit_styles(run.path("05_arrangement.json"))
    arrangement = Arrangement.model_validate_json(run.path("05_arrangement.json").read_text())
    roles = [s.role for s in arrangement.sections]
    ids = [s.id for s in arrangement.sections]
    first, last = ids[roles.index("build")], ids[roles.index("drop")]
    result = cli("regenerate", "--section", f"{first}-{last}", "--adherence", "medium", "--yes")
    assert result.exit_code == 0, result.output
    assert len(fake.compose_calls) == 3 and "▶ generate" not in result.output
    generated = [c for c in fake.compose_calls[-1]["composition_plan"]["chunks"]
                 if "song_id" not in c]  # fmt: skip
    assert [c["text"] for c in generated] == ["[Build]", "[Drop]"]
    assert {c["context_adherence"] for c in generated} == {"medium"}
    meta = TakeMeta.model_validate_json(run.path("06_music/take_001.meta.json").read_text())
    assert meta.params["edits"][0]["section"] == f"{first}-{last}"
    assert "using take 1 (requested)" in result.output


def test_regenerate_needs_an_elevenlabs_take(cli: Cli, talk: Talk, eleven: Callable) -> None:
    eleven()
    assert cli("generate").exit_code == 0  # the stub
    assert cli("mix").exit_code == 0
    result = cli("regenerate", "--section", "s2", "--yes")
    assert result.exit_code == 1
    assert "was not generated with ElevenLabs" in result.output


def test_takes_from_before_grids_read_their_timing_from_the_response(tmp_path: Path) -> None:
    from speech2song.backends.music_base import take_fits, take_grid

    music = tmp_path / "06_music"
    music.mkdir()
    plan = {"chunks": [{"duration_ms": 4000}, {"duration_ms": 6000}]}
    (music / "take_001.response.json").write_text(json.dumps({"composition_plan": plan}))
    meta = TakeMeta(take=1, backend="elevenlabs", file="06_music/take_001.mp3", sample_rate=1,
                    channels=2, seconds=10, request_sha256="old")  # fmt: skip
    assert take_grid(meta, tmp_path) == [4000, 6000]
    assert take_fits(meta, tmp_path, [4000, 6000], "new")
    assert not take_fits(meta, tmp_path, [5000, 5000], "new")
    edited = meta.model_copy(update={"file": "06_music/take_001_v2.mp3",
                                     "params": {"history": [{"file": meta.file}]}})  # fmt: skip
    assert take_grid(edited, tmp_path) == [4000, 6000]  # the first version's plan
    bare = meta.model_copy(update={"file": "06_music/elsewhere.mp3"})
    assert take_fits(bare, tmp_path, [1], "old") and not take_fits(bare, tmp_path, [1], "new")
