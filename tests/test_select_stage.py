"""The select step end to end: fake Claude, synthetic talk, real cutting."""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml
from typer.testing import Result

from speech2song.manifest import Run
from speech2song.models import ClipReview, ClipSet, SelectionResult

from .conftest import PRESETS_DIR
from .fixtures.fake_claude import PICKS, clip_answer, response
from .fixtures.synth import SR

Cli = Callable[..., Result]
FADE = round(0.015 * SR)


def _clip_set(run: Run) -> ClipSet:
    return ClipSet.model_validate_json(run.path("03_clips.json").read_text())


def test_dry_run_estimates_without_calling(cli: Cli, talk, install) -> None:
    run = talk[0]
    fake = install()
    result = cli("select", "--dry-run")
    assert result.exit_code == 0, result.output
    assert "select up to 10 clips (8 sentences)" in result.output
    assert "Dry run total, worst case" in result.output
    assert fake.requests == []
    assert not run.path("03_selection.json").exists()
    assert json.loads(run.costs_path.read_text()) == []


def test_paid_call_needs_yes_without_a_terminal(cli: Cli, talk, install) -> None:
    fake = install()
    result = cli("select")
    assert result.exit_code == 1
    assert "pass --yes" in result.output
    assert fake.requests == []


def test_select_cuts_sample_exact_clips_in_the_pauses(cli: Cli, talk, install) -> None:
    run, transcript, silences, stereo = talk
    fake = install(response(clip_answer(transcript), input_tokens=1000, output_tokens=500))
    result = cli("select", "--yes")
    assert result.exit_code == 0, result.output
    assert "Choose 6 to 10 (about 8) clips" in fake.requests[0]["messages"][0]["content"]

    [entry] = json.loads(run.costs_path.read_text())
    assert entry["usd"] == pytest.approx(0.007)
    assert "Spend this command: $0.0070" in result.output

    selection = SelectionResult.model_validate_json(run.path("03_selection.json").read_text())
    assert (len(selection.attempts), selection.chosen) == (1, 0)
    clip_set = _clip_set(run)
    assert [c.id for c in clip_set.clips] == ["c1", "c2", "c3", "c4", "c5"]
    assert clip_set.order == ["c5", "c4", "c3", "c2", "c1"]
    assert (clip_set.source, clip_set.channels, clip_set.isolated) == ("01_clean.wav", 2, False)
    for clip, (first, last) in zip(clip_set.clips, PICKS, strict=True):
        before, after = silences[first - 1], silences[last]
        assert before[0] <= clip.start_s <= before[1], (clip.id, clip.start_s, before)
        assert after[0] <= clip.end_s <= after[1], (clip.id, clip.end_s, after)
        assert max(clip.cut_level_db) < -80
        # Fades land in the pause: no speech within the fade at either edge.
        assert np.abs(stereo[clip.start_sample : clip.start_sample + FADE]).max() < 1e-4
        assert np.abs(stereo[clip.end_sample - FADE : clip.end_sample]).max() < 1e-4
        data, rate = sf.read(str(run.path(clip.file)), dtype="float32", always_2d=True)
        assert rate == SR and len(data) == clip.end_sample - clip.start_sample
        source = stereo[clip.start_sample : clip.end_sample]
        np.testing.assert_array_equal(data[FADE:-FADE], source[FADE:-FADE])  # sample-exact


def test_second_run_is_cached(cli: Cli, talk, install) -> None:
    install(response(clip_answer(talk[1])))
    cli("select", "--yes")
    result = cli("select", "--yes")  # the fake has no responses left: any call would fail
    assert "select: cached" in result.output and "clips: cached" in result.output


def test_changing_fades_recuts_without_calling_claude(
    cli: Cli, talk, install, tmp_path: Path
) -> None:
    install(response(clip_answer(talk[1])))
    cli("select", "--yes")
    presets = tmp_path / "presets"
    presets.mkdir()
    data = yaml.safe_load((PRESETS_DIR / "cinematic_future_bass.yaml").read_text())
    data["clips"] = {"fade_ms": 5}
    (presets / "cinematic_future_bass.yaml").write_text(yaml.safe_dump(data))
    (tmp_path / "config.yaml").write_text(f"presets_dir: {presets}\nruns_dir: runs\n")
    result = cli("select", "--yes")
    assert "select: cached" in result.output
    assert "clips (stale)" in result.output
    assert _clip_set(talk[0]).fade_ms == 5


def test_invalid_answer_gets_one_retry_with_feedback(cli: Cli, talk, install) -> None:
    run, transcript = talk[0], talk[1]
    bad = clip_answer(transcript)
    bad["clips"][2]["text"] = transcript.sentences[5].text  # quotes the wrong sentence
    fake = install(response(bad), response(clip_answer(transcript)))
    result = cli("select", "--yes")
    assert result.exit_code == 0, result.output
    assert "asking once more with feedback" in result.output
    retry_prompt = fake.requests[1]["messages"][0]["content"]
    assert "Your previous answer had these problems" in retry_prompt
    assert "which read" in retry_prompt
    selection = SelectionResult.model_validate_json(run.path("03_selection.json").read_text())
    assert len(selection.attempts) == 2 and selection.attempts[0].problems
    assert selection.chosen == 1
    assert len(_clip_set(run).clips) == 5
    assert len(json.loads(run.costs_path.read_text())) == 2


def test_refusal_fails_the_stage_but_logs_the_cost(cli: Cli, talk, install) -> None:
    run = talk[0]
    install(response({}, stop_reason="refusal", category="cyber"))
    result = cli("select", "--yes")
    assert result.exit_code == 1
    assert "declined" in result.output
    assert Run.open(run.root.parent, run.id).manifest.stages["select"].status == "failed"
    assert len(json.loads(run.costs_path.read_text())) == 1


def test_interactive_review_is_saved_and_survives_reruns(cli: Cli, talk, install) -> None:
    run = talk[0]
    install(response(clip_answer(talk[1])))
    result = cli("select", "--yes", "--interactive-review", input="drop c2\nmove c1 1\nsave\n")
    assert result.exit_code == 0, result.output
    review = ClipReview.model_validate_json(run.path("03_review.json").read_text())
    assert review.order == ["c1", "c5", "c4", "c3"] and review.dropped == ["c2"]
    clip_set = _clip_set(run)
    assert clip_set.order == review.order and clip_set.dropped == ["c2"]
    again = cli("select", "--yes")
    assert "clips: cached" in again.output
    assert _clip_set(run).order == review.order


def test_review_quit_saves_nothing(cli: Cli, talk, install) -> None:
    install(response(clip_answer(talk[1])))
    result = cli("select", "--yes", "--interactive-review", input="drop c1\nquit\n")
    assert result.exit_code == 0, result.output
    assert "without saving" in result.output
    assert not talk[0].path("03_review.json").exists()


def test_costs_command_lists_the_call(cli: Cli, talk, install) -> None:
    install(response(clip_answer(talk[1])))
    cli("select", "--yes")
    output = cli("costs").output
    assert "claude-sonnet-5-5" in output and "$0.0070" in output


def _preset_with(tmp_path: Path, **clips: float) -> None:
    presets = tmp_path / "presets"
    presets.mkdir(exist_ok=True)
    data = yaml.safe_load((PRESETS_DIR / "cinematic_future_bass.yaml").read_text())
    data["clips"] = {**data.get("clips", {}), **clips}
    (presets / "cinematic_future_bass.yaml").write_text(yaml.safe_dump(data))
    (tmp_path / "config.yaml").write_text(f"presets_dir: {presets}\nruns_dir: runs\n")


def test_long_quotes_are_cut_into_parts_at_their_pauses(
    cli: Cli, talk, install, tmp_path: Path
) -> None:
    run, _, silences, stereo = talk
    _preset_with(tmp_path, part_seconds=4.5)  # c2 (sentences 2-3) has more speech than that
    install(response(clip_answer(talk[1])))
    result = cli("select", "--yes")
    assert result.exit_code == 0, result.output
    clip_set = _clip_set(run)
    assert [c.id for c in clip_set.clips] == ["c1", "c2-1", "c2-2", "c3", "c4", "c5"]
    assert clip_set.order == ["c5", "c4", "c3", "c2-1", "c2-2", "c1"]
    first, second = clip_set.clips[1:3]
    assert (first.quote, first.part, first.parts, second.part) == ("c2", 1, 2, 2)
    assert (first.start_sentence, first.end_sentence, second.start_sentence) == (2, 2, 3)
    pause = silences[2]  # between sentences 2 and 3
    assert pause[0] <= first.end_s <= second.start_s <= pause[1]
    for clip in (first, second):
        data, _ = sf.read(str(run.path(clip.file)), dtype="float32", always_2d=True)
        source = stereo[clip.start_sample : clip.end_sample]
        np.testing.assert_array_equal(data[FADE:-FADE], source[FADE:-FADE])  # sample-exact
    assert "(part 1 of 2)" in result.output


def test_required_quotes_are_matched_and_kept(cli: Cli, talk, install, tmp_path: Path) -> None:
    run, transcript, _, _ = talk
    quotes = tmp_path / "quotes.txt"
    quotes.write_text(transcript.sentences[5].text + "\n")  # sentence 6: Claude skips it
    fake = install(response(clip_answer(transcript)), response(clip_answer(transcript)))
    result = cli("select", "--quotes", str(quotes), "--yes")
    assert result.exit_code == 0, result.output
    assert "required q1: sentences 6-6 (100% match)" in result.output
    prompt = fake.requests[0]["messages"][0]["content"]
    assert "The producer requires these passages" in prompt and "sentences 6-6" in prompt
    assert len(fake.requests) == 2  # the answer missed it: one retry with feedback
    clip_set = _clip_set(run)
    q1 = next(c for c in clip_set.clips if c.id == "q1")
    assert (q1.start_sentence, q1.end_sentence, q1.reason) == (6, 6, "required by the producer")
    assert "q1" in clip_set.order
    assert "select: cached" in cli("select", "--yes").output  # the quotes file sticks

    quotes.write_text("words nobody said in this talk at all\n")
    result = cli("select", "--yes")
    assert result.exit_code == 1 and "Quote 1 was not found" in result.output
    assert len(fake.requests) == 2  # refused before any call
    result = cli("select", "--no-quotes", "--dry-run")  # the request changes: a paid re-run
    assert "select: would run" in result.output and len(fake.requests) == 2
    assert Run.open(run.root.parent, run.id).manifest.quotes is None
