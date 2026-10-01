import io
import socket
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from rich.console import Console
from typer.testing import CliRunner, Result

from speech2song.cli import app
from speech2song.config import AppConfig
from speech2song.llm import claude
from speech2song.manifest import Run, write_json
from speech2song.models import Transcript
from speech2song.pipeline import Context

from .fixtures.fake_claude import FakeAnthropic
from .fixtures.synth import synthetic_talk, write_wav

REPO_ROOT = Path(__file__).resolve().parents[1]
PRESETS_DIR = REPO_ROOT / "presets"
KEY_VARS = ("ANTHROPIC_API_KEY", "ELEVENLABS_API_KEY")


@pytest.fixture(autouse=True)
def _isolated_environment(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No API keys, no network, and a cwd where the real .env/config.yaml can't be found."""
    for var in KEY_VARS:
        monkeypatch.setenv(var, "")  # records the original value so teardown restores it
        monkeypatch.delenv(var)
    monkeypatch.chdir(tmp_path)
    if "integration" not in request.keywords:

        def no_network(*args: object, **kwargs: object) -> None:
            raise RuntimeError("Network access is disabled in unit tests")

        monkeypatch.setattr(socket.socket, "connect", no_network)
        monkeypatch.setattr(socket, "create_connection", no_network)


@pytest.fixture
def config(tmp_path: Path) -> AppConfig:
    return AppConfig(runs_dir=tmp_path / "runs", presets_dir=PRESETS_DIR)


@pytest.fixture
def input_file(tmp_path: Path) -> Path:
    path = tmp_path / "My Talk.mp3"
    path.write_bytes(b"not really audio")
    return path


@pytest.fixture
def run(config: AppConfig, input_file: Path) -> Run:
    return Run.create(config.runs_dir, input_file, preset="cinematic_future_bass")


@pytest.fixture
def console_output() -> io.StringIO:
    return io.StringIO()


@pytest.fixture
def make_context(config: AppConfig, console_output: io.StringIO) -> Callable[..., Context]:
    def make(run: Run, **flags: bool) -> Context:
        console = Console(file=console_output, width=200, highlight=False)
        return Context(run, config, console, **flags)

    return make


@pytest.fixture
def cli(tmp_path: Path) -> Callable[..., Result]:
    """Invoke the CLI with a config.yaml (in the tmp cwd) pointing at the repo presets."""
    (tmp_path / "config.yaml").write_text(f"presets_dir: {PRESETS_DIR}\nruns_dir: runs\n")
    runner = CliRunner()

    def invoke(*args: str, input: str | None = None) -> Result:
        return runner.invoke(app, list(args), input=input, catch_exceptions=False)

    return invoke


Talk = tuple[Run, Transcript, list[tuple[float, float]], np.ndarray]


@pytest.fixture
def talk(cli: Callable[..., Result], tmp_path: Path) -> Talk:
    """A run (in the CLI's runs dir) holding a synthetic talk: 01_clean.wav (stereo) and
    02_transcript.json with jittered word times, plus the true pauses."""
    audio, transcript, silences = synthetic_talk()
    stereo = np.stack([audio, 0.8 * audio], axis=1)
    source = write_wav(tmp_path / "talk.wav", stereo)
    run = Run.create(tmp_path / "runs", source, preset="cinematic_future_bass")
    write_wav(run.path("01_clean.wav"), stereo)
    write_json(run.path("02_transcript.json"), transcript)
    return run, transcript, silences, stereo


@pytest.fixture
def install(monkeypatch: pytest.MonkeyPatch) -> Callable[..., FakeAnthropic]:
    """Install a fake Anthropic client with scripted responses."""

    def install_fake(*responses: object) -> FakeAnthropic:
        fake = FakeAnthropic(*responses)
        monkeypatch.setattr(claude, "client_factory", lambda: fake)
        return fake

    return install_fake
