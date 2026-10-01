from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import pytest
import yaml
from typer.testing import Result

from speech2song.costs import CostLog
from speech2song.manifest import Run
from speech2song.models import CostEntry

from .conftest import PRESETS_DIR

Cli = Callable[..., Result]


@pytest.fixture
def cli_run(cli: Cli, tmp_path: Path, input_file: Path) -> Run:
    """A run inside the CLI's runs dir (tmp cwd/runs)."""
    return Run.create(tmp_path / "runs", input_file, preset="cinematic_future_bass")


def test_help_lists_commands(cli: Cli) -> None:
    result = cli("--help")
    assert result.exit_code == 0
    for command in ("run", "ingest", "transcribe", "select", "costs", "status", "presets"):
        assert command in result.output


def test_version(cli: Cli) -> None:
    result = cli("--version")
    assert result.exit_code == 0
    assert result.output.startswith("speech2song ")


def test_presets_list(cli: Cli) -> None:
    result = cli("presets", "list")
    assert result.exit_code == 0
    assert "cinematic_future_bass" in result.output
    assert "ok" in result.output


def test_presets_list_flags_invalid_preset(cli: Cli, tmp_path: Path) -> None:
    presets = tmp_path / "my_presets"
    presets.mkdir()
    data = yaml.safe_load((PRESETS_DIR / "cinematic_future_bass.yaml").read_text())
    data.update(name="broken", arc=["nope"])
    (presets / "broken.yaml").write_text(yaml.safe_dump(data))
    (tmp_path / "config.yaml").write_text(f"presets_dir: {presets}\n")
    result = cli("presets", "list")
    assert "broken" in result.output
    assert "invalid" in result.output


@pytest.mark.parametrize(("command", "milestone"), [("melody", "M3"), ("mix", "M4")])
def test_unimplemented_steps_exit_2(cli: Cli, cli_run: Run, command: str, milestone: str) -> None:
    result = cli(command)
    assert result.exit_code == 2
    assert f"planned for {milestone}" in result.output


def test_regenerate_is_planned(cli: Cli) -> None:
    result = cli("regenerate", "--section", "s5")
    assert result.exit_code == 2
    assert "M5" in result.output


def test_costs_empty_and_with_entries(cli: Cli, cli_run: Run) -> None:
    result = cli("costs")
    assert result.exit_code == 0
    assert "No paid calls" in result.output
    CostLog(cli_run.costs_path).append(
        CostEntry(
            ts=datetime(2026, 10, 1, 9, 30).astimezone(),
            run_id=cli_run.id,
            stage="select",
            service="anthropic",
            operation="messages.create",
            model="claude-sonnet-5-5",
            units={"input_tokens": 41000, "output_tokens": 8000},
            usd=0.243,
        )
    )
    result = cli("costs", "--run", cli_run.id)
    assert result.exit_code == 0
    assert "$0.2430" in result.output
    assert "Total" in result.output


def test_status_shows_steps(cli: Cli, cli_run: Run) -> None:
    result = cli("status")
    assert result.exit_code == 0
    assert cli_run.id in result.output
    assert "select" in result.output


def test_unknown_run_is_a_clean_error(cli: Cli, cli_run: Run) -> None:
    result = cli("status", "--run", "nope")
    assert result.exit_code == 1
    assert "Error: No run" in result.output


def test_bad_config_is_a_clean_error(cli: Cli, tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text("not_a_setting: 1\n")
    result = cli("presets", "list")
    assert result.exit_code == 1
    assert "not_a_setting" in result.output


def test_run_dry_run_creates_nothing(cli: Cli, tmp_path: Path, input_file: Path) -> None:
    result = cli("run", str(input_file), "--dry-run")
    assert result.exit_code == 0, result.output
    assert "assuming a 15-minute talk" in result.output  # the fixture is not real audio
    assert "select 5 clips" in result.output
    assert "Nothing was executed" in result.output
    assert not (tmp_path / "runs").exists()


def test_input_and_run_are_exclusive(cli: Cli, cli_run: Run, input_file: Path) -> None:
    result = cli("run", str(input_file), "--run", cli_run.id)
    assert result.exit_code == 1
    assert "not both" in result.output


def test_missing_input_is_a_clean_error(cli: Cli) -> None:
    result = cli("ingest", "nope.mp3")
    assert result.exit_code == 1
    assert "Input file not found" in result.output


def test_secrets_never_reach_the_run_dir(
    cli: Cli, cli_run: Run, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sentinel = "sk-ant-SENTINEL-DO-NOT-LOG"
    monkeypatch.setenv("ANTHROPIC_API_KEY", sentinel)
    (tmp_path / ".env").write_text(f"ELEVENLABS_API_KEY={sentinel}\n")
    cli("status")
    cli("run", "--run", cli_run.id, "--stop-after", "transcribe")
    for path in cli_run.root.rglob("*"):
        if path.is_file():
            assert sentinel not in path.read_text(errors="ignore"), path
