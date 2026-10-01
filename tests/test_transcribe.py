"""The transcribe step (asr + align stages), with a fake transcriber."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import Result

from speech2song.backends import transcribe_base
from speech2song.backends.transcribe_whisper import (
    WhisperTranscriber,
    apply_repairs,
    find_holes,
    repair_windows,
    segments_to_asr,
    shift_segments,
)
from speech2song.config import AppConfig, WhisperConfig
from speech2song.manifest import Run
from speech2song.models import AsrResult, AsrSegment, AsrWord, RunOptions, Transcript

from .fixtures.synth import FIXTURES, asr_result, fake_asr_words, speechlike, write_wav

Cli = Callable[..., Result]
SPOKEN = (FIXTURES / "mini_talk_spoken.txt").read_text(encoding="utf-8").strip()
OFFICIAL = FIXTURES / "mini_talk.txt"


@dataclass
class FakeWord:
    start: float
    end: float
    word: str
    probability: float


@dataclass
class FakeSegment:
    id: int
    start: float
    end: float
    text: str
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    words: list[FakeWord] | None


def test_segments_to_asr_cleans_word_timings() -> None:
    segments = [
        FakeSegment(1, 0.0, 2.0, " Hello there. ", -0.2, 1.1, 0.01, [
            FakeWord(0.1234, 0.5, " Hello", 0.98),
            FakeWord(0.45, 0.4, " there.", 0.91),  # overlaps the previous word, ends early
            FakeWord(0.6, 0.7, "  ", 0.5),  # empty
        ]),
        FakeSegment(2, 2.0, 3.0, " Bye.", -0.3, 1.0, 0.02, [FakeWord(2.1, 2.4, " Bye.", 0.8)]),
        FakeSegment(3, 3.0, 3.5, "", -1.0, 1.0, 0.9, None),
    ]  # fmt: skip
    result = segments_to_asr(
        segments,
        language="en",
        language_probability=0.99,
        duration_s=3.5,
        backend="faster-whisper",
        model="m",
        params={"model": "m"},
        audio="01_clean.wav",
    )
    words = result.words()
    assert [(w.w, w.start, w.end, w.conf) for w in words] == [
        ("Hello", 0.123, 0.5, 0.98),
        ("there.", 0.5, 0.5, 0.91),
        ("Bye.", 2.1, 2.4, 0.8),
    ]
    assert [s.text for s in result.segments] == ["Hello there.", "Bye.", ""]
    assert result.duration_s == 3.5


def _words(*spans: tuple[float, float], prefix: str = "w") -> list[AsrWord]:
    return [AsrWord(w=f"{prefix}{i}", start=a, end=b, conf=0.9) for i, (a, b) in enumerate(spans)]


def _segment(number: int, *spans: tuple[float, float], prefix: str = "w") -> AsrSegment:
    words = _words(*spans, prefix=prefix)
    return AsrSegment(id=number, start=spans[0][0], end=spans[-1][1], text="", words=words)


def test_find_holes_needs_a_long_gap_with_speech() -> None:
    words = _words((0.5, 1.0), (1.2, 1.5), (31.0, 31.4), (40.0, 40.5))
    speech = [(0.4, 1.6), (2.0, 30.5), (31.0, 31.5), (35.0, 36.0), (40.0, 40.6), (44.0, 49.0)]
    holes = find_holes(words, speech, 50.0, min_gap_s=3.0, min_speech_s=1.5)
    # 1.5-31.0 holds 28.6 s of speech; 31.4-40.0 only 1 s (a pause); 40.5-50 holds 5.1 s.
    assert holes == [(1.5, 31.0, 28.6), (40.5, 50.0, 5.1)]


def test_find_holes_without_any_words() -> None:
    assert find_holes([], [(1.0, 9.0)], 10.0, min_gap_s=3.0, min_speech_s=1.5) == [(0.0, 10.0, 8.0)]


def test_repair_windows_take_in_the_touching_segments() -> None:
    segments = [
        _segment(1, (0.0, 1.0), (1.2, 2.0)),
        _segment(2, (10.0, 11.0), (11.5, 12.0)),  # the gap 2.0-10.0 sits between 1 and 2
        _segment(3, (13.0, 14.0), (20.0, 21.0)),  # gap 14.0-20.0 is inside segment 3
        # Segment 4 starts 41 s before its gap: too far to take in on that side.
        _segment(4, (100.0, 101.0), (140.0, 141.0), (180.0, 181.0)),
    ]
    holes = [(2.0, 10.0, 7.0), (14.0, 20.0, 5.0), (141.0, 180.0, 30.0)]
    windows = repair_windows(segments, holes, max_extend_s=30.5)
    assert [(w.start, w.end) for w in windows] == [(0.0, 12.0), (13.0, 21.0), (141.0, 181.0)]
    merged = repair_windows(segments, holes[:2], max_extend_s=30.5)
    assert len(merged) == 2
    overlapping = repair_windows(segments[:2], [(2.0, 10.0, 7.0), (11.0, 11.5, 0.5)],
                                 max_extend_s=30.5)  # fmt: skip
    assert [(w.start, w.end, w.speech_s) for w in overlapping] == [(0.0, 12.0, 7.5)]


def test_apply_repairs_replaces_the_words_in_the_window() -> None:
    before = _segment(1, (0.5, 1.0), (1.2, 1.5))
    damaged = _segment(2, (20.0, 21.4), (21.4, 22.8), (24.2, 24.6), prefix="bad")
    after = _segment(3, (25.0, 25.4))
    result = asr_result([]).model_copy(update={"segments": [before, damaged, after]})
    window = repair_windows([before, damaged, after], [(1.5, 20.0, 15.0)], max_extend_s=30.5)[0]
    assert (window.start, window.end) == (0.5, 24.6)
    redo = AsrSegment(id=1, start=0.0, end=24.2, text="", words=[
        AsrWord(w="a", start=0.0, end=0.5, conf=0.9),
        AsrWord(w="b", start=0.7, end=1.4, conf=0.9),
        AsrWord(w="c", start=12.0, end=12.5, conf=0.9),
        AsrWord(w="d", start=23.6, end=24.4, conf=0.9),
        AsrWord(w="x", start=24.4, end=25.4, conf=0.9),  # midpoint after the window: dropped
    ])  # fmt: skip
    merged = apply_repairs(result, [(window, shift_segments([redo], window.start))])
    assert [w.w for w in merged.words()] == ["a", "b", "c", "d", "w0"]
    assert [(s.id, s.repaired) for s in merged.segments] == [(1, True), (2, False)]
    assert merged.repairs[0].words_before == 5
    assert merged.repairs[0].words_after == 4
    starts = [w.start for w in merged.words()]
    assert starts == sorted(starts)


def test_whisper_params_leave_out_machine_settings() -> None:
    settings = WhisperConfig(cpu_threads=3, device="cpu")
    params = WhisperTranscriber("large-v3-turbo", "en", settings).params()
    assert params["model"] == "large-v3-turbo"
    assert params["language"] == "en"
    assert params["batch_size"] == settings.batch_size
    assert "cpu_threads" not in params and "device" not in params


def test_run_options_override_config_model() -> None:
    config = AppConfig()
    default = transcribe_base.make_transcriber(config, RunOptions())
    override = transcribe_base.make_transcriber(config, RunOptions(whisper_model="small"))
    assert default.params()["model"] == "large-v3-turbo"
    assert override.params()["model"] == "small"


class FakeTranscriber:
    name = "fake"
    calls = 0

    def __init__(self, model: str) -> None:
        self.model = model

    def params(self) -> dict[str, Any]:
        return {"backend": "fake", "model": self.model}

    def transcribe(self, audio: Path, *, progress: Any = None) -> AsrResult:
        FakeTranscriber.calls += 1
        if progress is not None:
            progress(1.0)
        return asr_result(fake_asr_words(SPOKEN)).model_copy(update={"model": self.model})


@pytest.fixture
def fake_asr(monkeypatch: pytest.MonkeyPatch) -> type[FakeTranscriber]:
    FakeTranscriber.calls = 0
    monkeypatch.setattr(
        transcribe_base,
        "make_transcriber",
        lambda config, options: FakeTranscriber(options.whisper_model or config.whisper_model),
    )
    return FakeTranscriber


@pytest.fixture
def ingested(cli: Cli, tmp_path: Path) -> Run:
    voice, _ = speechlike([(180, 220, 0.3), (220, 160, 0.25)] * 4)
    talk = write_wav(tmp_path / "talk.wav", voice)
    assert cli("ingest", str(talk), "--transcript", str(OFFICIAL)).exit_code == 0
    return Run.open(tmp_path / "runs", "latest")


def _transcript(run: Run) -> Transcript:
    return Transcript.model_validate_json(run.path("02_transcript.json").read_text())


def test_transcribe_aligns_to_the_official_text(
    cli: Cli, ingested: Run, fake_asr: type[FakeTranscriber]
) -> None:
    result = cli("transcribe")
    assert result.exit_code == 0, result.output
    asr = json.loads(ingested.path("02_asr.json").read_text())
    assert asr["model"] == "large-v3-turbo"
    transcript = _transcript(ingested)
    assert transcript.official_transcript_used
    assert transcript.source == str((ingested.root.parent.parent / "talk.wav").resolve())
    assert 'unspoken (line 14): "Conclusion"' in result.output
    assert "ASR-only" in result.output


def test_transcribe_caches_and_realigns_without_rerunning_asr(
    cli: Cli, ingested: Run, fake_asr: type[FakeTranscriber]
) -> None:
    cli("transcribe")
    again = cli("transcribe")
    assert "asr: cached" in again.output and "align: cached" in again.output
    no_official = cli("transcribe", "--no-transcript")
    assert "asr: cached" in no_official.output
    assert "align (stale)" in no_official.output
    assert fake_asr.calls == 1
    run = Run.open(ingested.root.parent, ingested.id)
    assert not _transcript(run).official_transcript_used
    assert run.manifest.transcript is None


def test_whisper_model_flag_sticks_to_the_run(
    cli: Cli, ingested: Run, fake_asr: type[FakeTranscriber]
) -> None:
    cli("transcribe")
    result = cli("transcribe", "--whisper-model", "small")
    assert "asr (stale)" in result.output
    assert fake_asr.calls == 2
    assert "asr: cached" in cli("transcribe").output  # no flag: keeps using "small"
    run = Run.open(ingested.root.parent, ingested.id)
    assert run.manifest.options.whisper_model == "small"
    assert json.loads(run.path("02_asr.json").read_text())["model"] == "small"


def test_status_after_transcribe(cli: Cli, ingested: Run, fake_asr: type[FakeTranscriber]) -> None:
    cli("transcribe")
    output = cli("status").output
    for stage in ("ingest", "isolate", "asr", "align"):
        assert f"{stage} " in output
    assert output.count("complete") == 4


def test_run_command_needs_consent_before_the_paid_step(
    cli: Cli, tmp_path: Path, fake_asr: type[FakeTranscriber]
) -> None:
    talk = write_wav(tmp_path / "talk.wav", np.zeros(4410, dtype=np.float32))
    result = cli("run", str(talk), "--transcript", str(OFFICIAL), "--stop-after", "transcribe")
    assert result.exit_code == 0, result.output
    assert "align" in result.output and "select" not in result.output
    paid = cli("run", "--run", "latest")  # no TTY and no --yes: must not spend
    assert paid.exit_code == 1
    assert "pass --yes" in paid.output
    run = Run.open(tmp_path / "runs", "latest")
    assert json.loads(run.costs_path.read_text()) == []
