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
from speech2song.models import Arrangement, AsrResult, AsrWord, ClipSet, MixReport, Transcript

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

    # A passage played alone (a hand edit): its bed runs until its last phrase, which
    # lands in silence; the take still fits (the plan doesn't change), so only the mix runs.
    path = run.path("05_arrangement.json")
    original = path.read_text()
    arrangement = Arrangement.model_validate_json(original)
    bed = [s for s in arrangement.sections if s.clip_id][1]
    bed.treatment = "break"
    path.write_text(arrangement.model_dump_json(indent=2))
    result = cli("mix", "--run", "latest")
    assert result.exit_code == 0, result.output
    assert "the music drops out at" in result.output
    report = MixReport.model_validate_json(run.path("07_mix/mix.json").read_text())
    (item,) = report.breaks
    assert item["sections"] == [bed.id] and item["back_s"] > item["stop_s"]
    speech, _ = sf.read(str(run.path("07_mix/stems/speech.wav")), dtype="float32", always_2d=True)
    for placed in report.clips:
        assert np.array_equal(speech[placed.start_sample : placed.end_sample],
                              clips[placed.clip_id])  # fmt: skip
    music, sr = sf.read(str(run.path("07_mix/stems/music.wav")), dtype="float32")
    start = round(bed.start_bar * 4 * 60 / arrangement.bpm * sr)
    stop, back = round(item["stop_s"] * sr), round(item["back_s"] * sr)
    assert np.abs(music[start : stop - sr // 10]).max() > 1e-3  # the bed under the words
    lone = music[
        stop + round(1.6 * sr) : back - round(1.5 * sr)
    ]  # after the ring, before the swell
    assert len(lone) and np.abs(lone).max() < 1e-4  # the last phrase alone
    assert np.abs(music[back : back + sr // 4]).max() > 1e-3  # and the music is back

    # As M7 stored it ("alone"): the music stops for the whole passage.
    arrangement = Arrangement.model_validate_json(original)
    bed = [s for s in arrangement.sections if s.clip_id][1]
    bed.treatment = "alone"
    path.write_text(arrangement.model_dump_json(indent=2))
    result = cli("mix", "--run", "latest")  # the free stub's take is made again for it
    assert result.exit_code == 1 and "generate" in result.output  # (the old take no longer fits)
    result = cli("run", "--run", "latest", "--yes")
    assert result.exit_code == 0, result.output
    assert "silence spliced in" in result.output and "stops for the passages" in result.output
    report = MixReport.model_validate_json(run.path("07_mix/mix.json").read_text())
    assert report.alone == [bed.id]
    speech, _ = sf.read(str(run.path("07_mix/stems/speech.wav")), dtype="float32", always_2d=True)
    for placed in report.clips:
        assert np.array_equal(speech[placed.start_sample : placed.end_sample],
                              clips[placed.clip_id])  # fmt: skip
    music, sr = sf.read(str(run.path("07_mix/stems/music.wav")), dtype="float32")
    start = round(bed.start_bar * 4 * 60 / arrangement.bpm * sr)
    words = next(p for p in report.clips if p.section_id == bed.id)
    alone = music[start + round(2.6 * sr) : words.end_sample - sr // 2]  # after the ring
    assert len(alone) and np.abs(alone).max() < 1e-4  # the speaker alone
