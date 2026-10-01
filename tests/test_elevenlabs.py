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
from speech2song.config import load_preset
from speech2song.costs import CostLog
from speech2song.models import ArcPart, Arrangement, TakeChoice, TakeMeta
from speech2song.music_plan import (
    MAX_CHUNK_MS,
    MIN_CHUNK_MS,
    build_plan,
    check_plan,
    layout_chunks,
    section_bounds_ms,
)
from speech2song.stages.generate_music import choose_take

from .conftest import PRESETS_DIR, Talk
from .fixtures.fake_claude import clip_answer, response
from .fixtures.fake_elevenlabs import FakeElevenLabs
from .test_arrangement import _melody

Cli = Callable[..., Result]
PRESET = load_preset("cinematic_future_bass", PRESETS_DIR)


def _arrangement(parts: list[ArcPart] | None = None, bpm: float = 120.0) -> Arrangement:
    clips = [ClipInfo("a", "hook", "a", 6.3, 6.0), ClipInfo("b", "build", "b", 9.3, 9.0),
             ClipInfo("c", "outro", "c", 5.3, 5.0)]  # fmt: skip
    melody = _melody(clips).model_copy(update={"bpm": bpm})
    return build_arrangement(parts or default_parts(PRESET, clips), clips, melody, PRESET)


# --- The composition plan -----------------------------------------------------------------


def test_plan_tiles_the_song_within_the_api_limits() -> None:
    arrangement = _arrangement()
    plan, layout = build_plan(arrangement, PRESET)
    assert check_plan(plan) == []
    total = section_bounds_ms(arrangement)[-1][2]
    assert sum(c["duration_ms"] for c in plan["chunks"]) == total
    assert layout[0].start_ms == 0 and layout[-1].end_ms == total
    for a, b in itertools.pairwise(layout):
        assert a.end_ms == b.start_ms
    assert all(MIN_CHUNK_MS <= c["duration_ms"] <= MAX_CHUNK_MS for c in plan["chunks"])


def test_gaps_fold_into_the_build_and_beds_merge() -> None:
    arrangement = _arrangement()
    plan, layout = build_plan(arrangement, PRESET)
    roles = {s.id: s.role for s in arrangement.sections}
    build = next(i for i, c in enumerate(layout) if c.roles[0] == "build")
    assert [roles[s] for s in layout[build].sections] == ["build", "gap"]
    assert plan["chunks"][build]["text"] == "[Build]\n{near silence}"
    assert any(s.startswith("ends with 2.0 seconds of near silence")
               for s in plan["chunks"][build]["positive_styles"])  # fmt: skip
    assert "steadily building" in plan["chunks"][build]["positive_styles"]
    assert not any(c.roles[0] == "gap" for c in layout)
    adjacent = [c for c in layout if len(c.sections) > 1 and c.roles[0] == "speech_bed"]
    assert all(roles[s] == "speech_bed" for c in adjacent for s in c.sections)


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
    assert plan["chunks"][-1]["text"] == "[Outro]"
    assert "fading out to silence at the end" in plan["chunks"][-1]["positive_styles"]


def test_reference_conditions_only_the_first_chunk() -> None:
    plan, _ = build_plan(_arrangement(), PRESET, reference_song_id="ref_9", reference_ms=45_000,
                         condition_strength="medium")  # fmt: skip
    first = plan["chunks"][0]
    assert first["conditioning_ref"] == {"song_id": "ref_9",
                                         "range": {"start_ms": 0, "end_ms": 30_000}}  # fmt: skip
    assert first["condition_strength"] == "medium"
    assert all("conditioning_ref" not in c for c in plan["chunks"][1:])


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

    arrangement = _arrangement()
    audio = synthesize(StubBackend().request(arrangement, PRESET, None), seed=3)
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
    assert json.loads(run.path("06_music/take_001.response.json").read_text())["song_metadata"]
    entries = [e for e in CostLog(run.costs_path).read() if e.service == "elevenlabs"]
    minutes = sum(c["duration_ms"] for c in call["composition_plan"]["chunks"]) / 60_000
    assert [e.operation for e in entries] == ["music.compose_detailed"] * 2
    assert entries[0].usd == pytest.approx(minutes * 0.15, abs=1e-6)
    assert entries[0].units == {"minutes": pytest.approx(minutes, abs=1e-4)}
    choice = TakeChoice.model_validate_json(run.path("06_music/analysis.json").read_text())
    assert choice.reason == "best score" and len(choice.takes) == 2
    selected = sf.info(str(run.path("06_music/selected.wav")))
    arrangement = Arrangement.model_validate_json(run.path("05_arrangement.json").read_text())
    assert selected.samplerate == 44100
    assert selected.duration == pytest.approx(arrangement.total_seconds + 2.0, abs=1e-3)

    assert "generate: cached" in cli("generate", "--yes").output  # no second payment
    assert len(fake.compose_calls) == 2
    assert cli("mix").exit_code == 0


def test_a_failed_take_does_not_repeat_the_paid_one(cli: Cli, talk: Talk, eleven: Callable) -> None:
    from elevenlabs.core.api_error import ApiError

    run = talk[0]
    fake = eleven(None, ApiError(status_code=500, body={"detail": {"message": "boom"}}))
    result = cli("generate", "--music-backend", "elevenlabs", "--yes")
    assert result.exit_code == 1
    assert "ElevenLabs error 500" in result.output and "boom" in result.output
    assert len([e for e in CostLog(run.costs_path).read() if e.service == "elevenlabs"]) == 1
    fake.responses = []
    result = cli("generate", "--yes")
    assert result.exit_code == 0, result.output
    assert "take 1: reusing the paid take" in result.output
    assert len(fake.compose_calls) == 3  # take 1 once, take 2 failed once, then made


def test_takes_of_an_old_request_are_archived(cli: Cli, talk: Talk, eleven: Callable) -> None:
    run = talk[0]
    fake = eleven()
    assert cli("generate", "--music-backend", "elevenlabs", "--yes").exit_code == 0
    path = run.path("05_arrangement.json")
    data = json.loads(path.read_text())
    data["sections"][0]["styles"] = ["a different intro"]
    path.write_text(json.dumps(data))
    result = cli("generate", "--yes")
    assert result.exit_code == 0, result.output
    assert len(fake.compose_calls) == 4
    archived = list((run.path("06_music") / "archive").iterdir())
    assert len(archived) == 1
    assert sorted(p.name for p in archived[0].iterdir()) == [
        "take_001.meta.json", "take_001.mp3", "take_002.meta.json", "take_002.mp3"]  # fmt: skip


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
    assert not [e for e in CostLog(run.costs_path).read() if e.service == "elevenlabs"]


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


def test_melody_reference_is_uploaded_once(
    cli: Cli, talk: Talk, eleven: Callable, short_song: Path
) -> None:
    run = talk[0]
    config = Path("config.yaml")
    config.write_text(config.read_text() + "elevenlabs:\n  melody_reference: true\n")
    fake = eleven()
    result = cli("generate", "--music-backend", "elevenlabs", "--yes")
    assert result.exit_code == 0, result.output
    assert len(fake.upload_calls) == 1
    first = fake.compose_calls[0]["composition_plan"]["chunks"][0]
    assert first["conditioning_ref"]["song_id"] == "ref_1"
    assert first["condition_strength"] == "low"
    operations = [e.operation for e in CostLog(run.costs_path).read() if e.service == "elevenlabs"]
    assert operations == ["music.upload", "music.compose_detailed", "music.compose_detailed"]
    preset = short_song / "cinematic_future_bass.yaml"
    data = yaml.safe_load(preset.read_text())
    data["section_roles"]["drop"]["styles"] = ["bigger drop"]
    preset.write_text(yaml.safe_dump(data))
    assert cli("generate", "--yes").exit_code == 0
    assert len(fake.upload_calls) == 1  # the same reference audio is not uploaded again


def test_force_makes_fresh_takes_and_archives_the_paid_ones(
    cli: Cli, talk: Talk, eleven: Callable
) -> None:
    run = talk[0]
    fake = eleven()
    assert cli("generate", "--music-backend", "elevenlabs", "--yes").exit_code == 0
    result = cli("generate", "--force", "--yes")
    assert result.exit_code == 0, result.output
    assert "making new takes" in result.output
    assert len(fake.compose_calls) == 4
    meta = TakeMeta.model_validate_json(run.path("06_music/take_001.meta.json").read_text())
    assert meta.song_id == "song_3"
    archived = next((run.path("06_music") / "archive").iterdir())
    assert "take_001.mp3" in {p.name for p in archived.iterdir()}


def test_regenerate_one_section(cli: Cli, talk: Talk, eleven: Callable) -> None:
    run = talk[0]
    fake = eleven()
    assert cli("generate", "--music-backend", "elevenlabs", "--yes").exit_code == 0
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
    stages = [e.stage for e in CostLog(run.costs_path).read() if e.service == "elevenlabs"]
    assert stages == ["generate", "generate", "regenerate"]

    bad = cli("regenerate", "--section", "s99", "--yes")
    assert bad.exit_code == 1 and "no section 's99'" in bad.output


def test_regenerate_needs_an_elevenlabs_take(cli: Cli, talk: Talk, eleven: Callable) -> None:
    eleven()
    assert cli("generate").exit_code == 0  # the stub
    result = cli("regenerate", "--section", "s2", "--yes")
    assert result.exit_code == 1
    assert "was not generated with ElevenLabs" in result.output
