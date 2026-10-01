"""The arrange, generate and mix steps through the CLI, on the synthetic talk: caching,
melody-layer modes, hand edits, and Claude's optional arc refinement (mocked)."""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml
from typer.testing import Result

from speech2song.audio.io import measure_loudness
from speech2song.costs import CostLog
from speech2song.models import ArcResult, Arrangement, ClipSet, MixReport

from .conftest import Talk
from .fixtures.fake_claude import clip_answer, response

Cli = Callable[..., Result]
ORDER = ["c5", "c4", "c3", "c2", "c1"]  # clip_answer plays its clips in reverse


TWO_CLIPS = [(1, 1), (8, 8)]  # a short song for the mixing tests: plays c2, then c1


def _prepare(cli: Cli, talk: Talk, install: Callable, picks: list | None = None) -> None:
    answer = clip_answer(talk[1]) if picks is None else clip_answer(talk[1], picks)
    install(response(answer))
    assert cli("select", "--yes").exit_code == 0
    assert cli("melody").exit_code == 0


def _arrangement(talk: Talk) -> Arrangement:
    return Arrangement.model_validate_json(talk[0].path("05_arrangement.json").read_text())


def _ran(output: str) -> list[str]:
    return [line.split()[1] for line in output.splitlines() if line.startswith("▶ ")]


def _m4(cli: Cli, *mix_flags: str) -> list[str]:
    """Run arrange, generate and mix; return the stages that actually ran."""
    ran = []
    for args in (("arrange", "--yes"), ("generate", "--yes"), ("mix", *mix_flags)):
        result = cli(*args)
        assert result.exit_code == 0, result.output
        ran += _ran(result.output)
    return ran


def test_arrange_generate_mix(cli: Cli, talk: Talk, install: Callable, short_song: Path) -> None:
    run = talk[0]
    _prepare(cli, talk, install, TWO_CLIPS)

    result = cli("arrange")
    assert result.exit_code == 0, result.output
    assert "Arrangement ·" in result.output and "energy │" in result.output
    arrangement = _arrangement(talk)
    assert [s.clip_id for s in arrangement.sections if s.clip_id] == ["c2", "c1"]
    assert [s.role for s in arrangement.sections][-2:] == ["speech_bed", "outro"]
    assert arrangement.arc_source == "preset"

    result = cli("generate")
    assert result.exit_code == 0, result.output
    assert run.path("06_music/take_001.wav").exists()
    assert not run.path("06_music/take_002.wav").exists()  # music_takes: 1

    result = cli("mix")
    assert result.exit_code == 0, result.output
    assert "Listen to:" in result.output
    selected = sf.info(str(run.path("06_music/selected.wav")))
    assert selected.duration == pytest.approx(arrangement.total_seconds + 2.0, abs=1e-3)
    report = MixReport.model_validate_json(run.path("07_mix/mix.json").read_text())
    assert report.melody_layer == "off"  # the preset's default: no MIDI in the song
    assert not run.path("07_mix/stems/melody_layer.wav").exists()
    gaps = [s.id for s in arrangement.sections if s.silent]
    assert gaps and report.silenced == gaps
    assert [level.section_id for level in report.section_levels] == [
        s.id for s in arrangement.sections]  # fmt: skip
    assert report.integrated_lufs == pytest.approx(-14.0, abs=0.2)
    assert report.true_peak_dbtp <= -1.0
    ffmpeg = measure_loudness(run.path("07_mix/master.wav"))
    assert ffmpeg["integrated_lufs"] == pytest.approx(-14.0, abs=0.3)
    assert ffmpeg["true_peak_dbtp"] <= -0.9  # ffmpeg's own true-peak meter agrees
    assert sf.info(str(run.path("07_mix/master.wav"))).subtype == "PCM_24"
    assert run.path("07_mix/master.mp3").stat().st_size > 10_000
    speech, _ = sf.read(str(run.path("07_mix/stems/speech.wav")), dtype="float32", always_2d=True)
    clip_set = ClipSet.model_validate_json(run.path("03_clips.json").read_text())
    files = {c.id: c.file for c in clip_set.clips}
    for placed in report.clips:
        clip, _ = sf.read(str(run.path(files[placed.clip_id])), dtype="float32", always_2d=True)
        assert np.array_equal(speech[placed.start_sample : placed.end_sample], clip)

    # Everything is cached now; the melody layer is a mix setting only.
    assert _m4(cli) == []
    result = cli("mix", "--melody-layer", "replay")
    assert _ran(result.output) == ["mix"]
    layer, _ = sf.read(str(run.path("07_mix/stems/melody_layer.wav")), dtype="float32")
    assert np.abs(layer).max() > 0
    assert _m4(cli, "--melody-layer", "all") == ["mix"]
    report = MixReport.model_validate_json(run.path("07_mix/mix.json").read_text())
    assert report.melody_layer == "all"
    result = cli("mix", "--melody-layer", "off")
    assert not run.path("07_mix/stems/melody_layer.wav").exists()
    assert "melody layer off" in result.output


def test_generate_reruns_only_when_the_music_request_changes(
    cli: Cli, talk: Talk, install: Callable, short_song: Path
) -> None:
    run = talk[0]
    _prepare(cli, talk, install, TWO_CLIPS)
    assert _m4(cli) == ["arrange", "generate", "take", "mix"]
    path = run.path("05_arrangement.json")

    data = json.loads(path.read_text())
    data["sections"][0]["melody_phrase"] = "c2"  # a melody-layer choice: not in the request
    path.write_text(json.dumps(data))
    assert _m4(cli) == ["mix"]

    data["sections"][0]["energy"] = 0.5  # the music itself changes
    path.write_text(json.dumps(data))
    assert _m4(cli) == ["generate", "take", "mix"]

    data["sections"][1]["start_bar"] += 1  # broken by hand: caught before mixing
    path.write_text(json.dumps(data))
    result = cli("mix")
    assert result.exit_code == 1
    assert "can't be used" in result.output and "starts at bar" in result.output
    assert "can't be used" in cli("generate").output  # nothing is generated for it either


def _arc_answer(parts: list[tuple[str, int, list[str]]]) -> dict:
    return {"parts": [{"role": r, "bars": b, "clips": c} for r, b, c in parts],
            "notes": "front-load the hook"}  # fmt: skip


GOOD_ARC = [("intro", 2, []), ("speech_bed", 0, ["c5", "c4"]), ("build", 2, []),
            ("drop", 4, []), ("speech_bed", 0, ["c3", "c2", "c1"]), ("outro", 2, [])]  # fmt: skip


def test_refine_arc_with_claude(cli: Cli, talk: Talk, install: Callable, short_song: Path) -> None:
    run = talk[0]
    _prepare(cli, talk, install)
    dry = cli("arrange", "--refine-arc", "--dry-run")
    assert dry.exit_code == 0, dry.output
    assert "refine the arc" in dry.output and "Nothing was executed" in dry.output
    assert not run.path("05_arc.json").exists()

    fake = install(response(_arc_answer(GOOD_ARC), input_tokens=2000, output_tokens=900))
    result = cli("arrange", "--yes")  # --refine-arc stuck to the run
    assert result.exit_code == 0, result.output
    request = fake.requests[0]
    schema = request["output_config"]["format"]["schema"]
    assert schema["properties"]["parts"]["items"]["properties"]["clips"]["items"]["enum"] == [
        "c5", "c4", "c3", "c2", "c1"]  # fmt: skip
    assert "- c5 (hook" in request["messages"][0]["content"]
    arrangement = _arrangement(talk)
    assert arrangement.arc_source == "claude"
    assert arrangement.notes == "front-load the hook"
    assert [s.role for s in arrangement.sections] == [
        "intro", "speech_bed", "speech_bed", "build", "drop", "speech_bed", "speech_bed",
        "speech_bed", "outro"]  # fmt: skip
    assert [e.stage for e in CostLog(run.costs_path).read()] == ["select", "arc"]

    # Cached: no scripted response is left, so another call would fail the test.
    assert _ran(cli("arrange", "--yes").output) == []
    # Turning refinement off goes back to the preset's arc without calling Claude.
    result = cli("arrange", "--no-refine-arc")
    assert _ran(result.output) == ["arrange"]
    assert _arrangement(talk).arc_source == "preset"


def test_refine_arc_retries_then_falls_back_to_the_preset(
    cli: Cli, talk: Talk, install: Callable, short_song: Path
) -> None:
    run = talk[0]
    _prepare(cli, talk, install)
    wrong_order = [("intro", 2, []), ("speech_bed", 0, ["c1", "c2", "c3", "c4", "c5"])]
    too_long = [("intro", 99, []), ("speech_bed", 0, ORDER)]
    fake = install(response(_arc_answer(wrong_order)), response(_arc_answer(too_long)))
    result = cli("arrange", "--refine-arc", "--yes")
    assert result.exit_code == 0, result.output
    assert "asking once more" in result.output
    assert "Your previous answer had these problems" in fake.requests[1]["messages"][0]["content"]
    saved = ArcResult.model_validate_json(run.path("05_arc.json").read_text())
    assert len(saved.attempts) == 2 and all(a.problems for a in saved.attempts)
    arrangement = _arrangement(talk)
    assert arrangement.arc_source == "preset"
    assert "not usable" in arrangement.warnings[0]


def test_refine_arc_needs_confirmation(
    cli: Cli, talk: Talk, install: Callable, short_song: Path
) -> None:
    _prepare(cli, talk, install)
    install()  # no responses: any call fails the test
    result = cli("arrange", "--refine-arc")
    assert result.exit_code == 1
    assert "--yes" in result.output
    assert not talk[0].path("05_arc.json").exists()


def test_arc_edits_keep_the_paid_selection(
    cli: Cli, talk: Talk, install: Callable, short_song: Path
) -> None:
    _prepare(cli, talk, install)  # the only scripted response is used up here
    path = short_song / "cinematic_future_bass.yaml"
    data = yaml.safe_load(path.read_text())
    data["arc"] = ["intro", "speech_bed", "drop", "outro"]
    path.write_text(yaml.safe_dump(data))
    result = cli("select", "--yes")
    assert result.exit_code == 0, result.output
    assert "select: cached" in result.output
    assert "Arrangement" in cli("arrange").output


def test_take_must_exist(cli: Cli, talk: Talk, install: Callable, short_song: Path) -> None:
    _prepare(cli, talk, install, TWO_CLIPS)
    assert _m4(cli) == ["arrange", "generate", "take", "mix"]
    result = cli("mix", "--take", "3")
    assert result.exit_code == 1
    assert "take 3 is not available; takes that fit the arrangement: 1" in result.output
    assert "give a take number" in cli("mix", "--take", "0").output
    result = cli("mix", "--take", "auto")
    assert result.exit_code == 0, result.output
    assert "using take 1 (only take)" in result.output
