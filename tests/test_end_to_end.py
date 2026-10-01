"""The whole pipeline offline (spec section 10): ingest, a fake Whisper, alignment, a fake
Claude, clip cutting, melody, arrangement, the stub music backend and the mix. The clips
inside the speech stem must be bit-identical to the cut clips, which are the source
times the fade envelope."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
from typer.testing import Result

from speech2song.audio.dsp import fade_edges
from speech2song.backends import transcribe_base
from speech2song.costs import CostLog
from speech2song.manifest import Run
from speech2song.models import AsrResult, AsrWord, ClipSet, MixReport, Transcript

from .fixtures.fake_claude import clip_answer, response
from .fixtures.synth import asr_result, synthetic_talk, write_wav

Cli = Callable[..., Result]


class TalkTranscriber:
    """Returns the synthetic talk's own word times, as Whisper would (roughly)."""

    name = "fake"

    def __init__(self, words: list[AsrWord]) -> None:
        self.words = words

    def params(self) -> dict[str, Any]:
        return {"backend": "fake"}

    def transcribe(self, audio: Path, *, progress: Any = None) -> AsrResult:
        return asr_result(self.words)


def test_whole_pipeline_keeps_speech_exact(
    cli: Cli, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, install: Callable,
    short_song: Path,
) -> None:  # fmt: skip
    audio, truth, _ = synthetic_talk()
    source = write_wav(tmp_path / "talk.wav", np.stack([audio, 0.8 * audio], axis=1))
    words = [AsrWord(w=w.w.capitalize(), start=w.start, end=w.end, conf=0.9) for w in truth.words]
    monkeypatch.setattr(transcribe_base, "make_transcriber",
                        lambda config, options: TalkTranscriber(words))  # fmt: skip

    first = cli("run", str(source), "--stop-after", "transcribe")
    assert first.exit_code == 0, first.output
    run = Run.open(tmp_path / "runs")
    transcript = Transcript.model_validate_json(run.path("02_transcript.json").read_text())
    assert len(transcript.sentences) == 8
    install(response(clip_answer(transcript, [(1, 1), (5, 5), (8, 8)])))

    result = cli("run", "--run", "latest", "--yes")
    assert result.exit_code == 0, result.output
    assert "Listen to:" in result.output

    clean, rate = sf.read(str(run.path("01_clean.wav")), dtype="float32", always_2d=True)
    clip_set = ClipSet.model_validate_json(run.path("03_clips.json").read_text())
    fade = round(clip_set.fade_ms / 1000 * rate)
    clips = {}
    for clip in clip_set.clips:  # the cut clips: the source times the fade envelope
        data, _ = sf.read(str(run.path(clip.file)), dtype="float32", always_2d=True)
        expected = fade_edges(clean[clip.start_sample : clip.end_sample], fade)
        assert np.array_equal(data, expected)
        clips[clip.id] = data

    report = MixReport.model_validate_json(run.path("07_mix/mix.json").read_text())
    speech, _ = sf.read(str(run.path("07_mix/stems/speech.wav")), dtype="float32", always_2d=True)
    assert sorted(p.clip_id for p in report.clips) == sorted(clips)
    for placed in report.clips:  # ...and the speech stem holds exactly those samples
        assert np.array_equal(speech[placed.start_sample : placed.end_sample],
                              clips[placed.clip_id])  # fmt: skip
    assert report.integrated_lufs == pytest.approx(-14.0, abs=0.2)
    assert report.true_peak_dbtp <= -1.0

    entries = CostLog(run.costs_path).read()
    assert [(e.service, e.stage) for e in entries] == [("anthropic", "select")]  # no music spend

    again = cli("run", "--run", "latest", "--yes")
    assert again.exit_code == 0
    assert not [line for line in again.output.splitlines() if line.startswith("▶ ")]
    status = cli("status")
    for stage in ("arrange", "generate", "take", "mix"):
        assert any(stage in line and "complete" in line for line in status.output.splitlines())
