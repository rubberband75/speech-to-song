"""The ingest step (ingest + isolate stages) through the CLI, with real ffmpeg."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from typer.testing import Result

from speech2song.backends import isolate_demucs
from speech2song.manifest import Run

from .fixtures.synth import SR, speechlike, sweep, write_wav

Cli = Callable[..., Result]


@pytest.fixture
def talk(tmp_path: Path) -> Path:
    """A 48 kHz stereo 'talk': speech-like syllables, as a 16-bit WAV."""
    voice, _ = speechlike([(180, 220, 0.3), (220, 160, 0.25), (150, 200, 0.4)] * 3)
    return write_wav(tmp_path / "talk.wav", np.stack([voice, voice], axis=1), sr=48000,
                     subtype="PCM_16")  # fmt: skip


def _only_run(tmp_path: Path) -> Run:
    return Run.open(tmp_path / "runs", "latest")


def test_ingest_creates_source_and_clean_link(cli: Cli, talk: Path, tmp_path: Path) -> None:
    result = cli("ingest", str(talk))
    assert result.exit_code == 0, result.output
    run = _only_run(tmp_path)
    info = sf.info(str(run.path("00_source.wav")))
    assert (info.samplerate, info.channels, info.subtype) == (SR, 2, "FLOAT")
    clean = run.path("01_clean.wav")
    assert clean.is_symlink() and clean.readlink() == Path("00_source.wav")
    manifest = run.manifest
    assert manifest.source_probe is not None and manifest.source_probe.sample_rate == 48000
    assert manifest.source is not None and manifest.source.duration_s == pytest.approx(
        info.frames / SR
    )
    assert manifest.clean is None
    assert {name: rec.status for name, rec in manifest.stages.items()} == {
        "ingest": "complete",
        "isolate": "complete",
    }
    assert "wrote 00_source.wav" in result.output
    assert "Spend this command: $0.0000" in result.output


def test_ingest_again_is_cached(cli: Cli, talk: Path, tmp_path: Path) -> None:
    cli("ingest", str(talk))
    result = cli("ingest")
    assert "ingest: cached" in result.output
    assert "isolate: cached" in result.output


def test_changed_input_reruns_ingest(cli: Cli, talk: Path, tmp_path: Path) -> None:
    cli("ingest", str(talk))
    write_wav(talk, np.stack([sweep(100, 900, 1.0, sr=48000)] * 2, axis=1), sr=48000)
    result = cli("ingest")
    assert "ingest (stale)" in result.output
    # Different source bytes make 01_clean.wav's input stale too.
    assert "isolate (stale)" in result.output


def test_force_reruns_with_identical_output(cli: Cli, talk: Path, tmp_path: Path) -> None:
    cli("ingest", str(talk))
    before = _only_run(tmp_path).manifest.stages["ingest"].outputs["00_source.wav"].sha256
    result = cli("ingest", "--force")
    assert "ingest (forced)" in result.output
    after = _only_run(tmp_path).manifest.stages["ingest"].outputs["00_source.wav"].sha256
    assert before == after


def test_missing_transcript_is_rejected(cli: Cli, talk: Path) -> None:
    result = cli("ingest", str(talk), "--transcript", "nope.txt")
    assert result.exit_code == 1
    assert "Transcript file not found" in result.output


class _HalfGain:
    """Stands in for demucs: a deterministic 'separation' that halves the signal."""

    def __init__(self, config: object) -> None:
        pass

    def __call__(self, block: np.ndarray) -> np.ndarray:
        return 0.5 * block


def test_isolate_voice_toggle_reruns_only_isolate(
    cli: Cli, talk: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(isolate_demucs, "DemucsSeparator", _HalfGain)
    cli("ingest", str(talk))
    result = cli("ingest", "--isolate-voice")
    assert "ingest: cached" in result.output
    assert "isolate (stale)" in result.output
    run = _only_run(tmp_path)
    clean_path = run.path("01_clean.wav")
    assert not clean_path.is_symlink()
    source, _ = sf.read(str(run.path("00_source.wav")), dtype="float32")
    clean, _ = sf.read(str(clean_path), dtype="float32")
    np.testing.assert_allclose(clean, 0.5 * source, atol=1e-6)
    assert run.manifest.options.isolate_voice
    assert run.manifest.clean is not None

    assert "isolate: cached" in cli("ingest").output  # the option sticks to the run
    cli("ingest", "--no-isolate-voice")
    assert _only_run(tmp_path).path("01_clean.wav").is_symlink()


def test_isolation_without_the_extra_explains_how_to_install(
    cli: Cli, talk: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import builtins

    real_import = builtins.__import__

    def no_demucs(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("demucs"):
            raise ImportError("No module named 'demucs'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_demucs)
    result = cli("ingest", str(talk), "--isolate-voice")
    assert result.exit_code == 1
    assert "uv sync --extra isolate" in result.output
