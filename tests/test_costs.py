import io
from datetime import datetime
from pathlib import Path

import pytest
from rich.console import Console

from speech2song.config import PricingConfig, TokenPrice, UnitPrice
from speech2song.costs import CostLog, SpendEstimate, claude_usd, confirm_spend, unit_usd
from speech2song.errors import SpendDeclinedError
from speech2song.models import CostEntry

PRICING = PricingConfig(
    anthropic={"claude-x": TokenPrice(input_usd_per_mtok=3.0, output_usd_per_mtok=15.0)},
    elevenlabs={"music": UnitPrice(unit="minute", usd_per_unit=0.5)},
)


def _entry(service: str, usd: float) -> CostEntry:
    return CostEntry(
        ts=datetime(2026, 10, 1, 12, 0).astimezone(),
        run_id="r",
        stage="select",
        service=service,  # type: ignore[arg-type]
        operation="op",
        model="m",
        units={"input_tokens": 1000},
        usd=usd,
    )


def test_cost_log_appends_and_totals(tmp_path: Path) -> None:
    log = CostLog(tmp_path / "costs.json")
    assert log.read() == []
    assert log.totals() == {"total": 0.0}
    log.append(_entry("anthropic", 0.25))
    log.append(_entry("elevenlabs", 1.0))
    log.append(_entry("anthropic", 0.5))
    entries = log.read()
    assert [e.usd for e in entries] == [0.25, 1.0, 0.5]
    assert log.totals() == {"anthropic": 0.75, "elevenlabs": 1.0, "total": 1.75}


def test_price_lookups() -> None:
    assert claude_usd(PRICING, "claude-x", 1_000_000, 100_000) == pytest.approx(4.5)
    assert claude_usd(PRICING, "unknown", 10, 10) is None
    assert unit_usd(PRICING, "music", 3.0) == pytest.approx(1.5)
    assert unit_usd(PRICING, "credits", 3.0) is None


def _estimates(usd: float | None = 0.12) -> list[SpendEstimate]:
    return [SpendEstimate(service="anthropic", model="m", description="d", units={}, usd=usd)]


def _console() -> tuple[Console, io.StringIO]:
    buffer = io.StringIO()
    return Console(file=buffer, width=200), buffer


def test_confirm_spend_dry_run_never_proceeds() -> None:
    console, buffer = _console()
    assert confirm_spend(_estimates(), console=console, yes=True, dry_run=True) is False
    assert "Dry run" in buffer.getvalue()
    assert "$0.1200" in buffer.getvalue()


def test_confirm_spend_yes_skips_prompt() -> None:
    console, _ = _console()

    def never_ask(question: str) -> bool:
        raise AssertionError("should not prompt")

    assert confirm_spend(_estimates(), console=console, yes=True, dry_run=False, ask=never_ask)


def test_confirm_spend_refuses_without_tty() -> None:
    console, _ = _console()
    with pytest.raises(SpendDeclinedError, match="--yes"):
        confirm_spend(
            _estimates(), console=console, yes=False, dry_run=False, is_interactive=lambda: False
        )


@pytest.mark.parametrize("answer", [True, False])
def test_confirm_spend_asks_interactively(answer: bool) -> None:
    console, _ = _console()
    kwargs = {"console": console, "yes": False, "dry_run": False}
    if answer:
        assert confirm_spend(
            _estimates(), **kwargs, is_interactive=lambda: True, ask=lambda q: True
        )
    else:
        with pytest.raises(SpendDeclinedError, match="declined"):
            confirm_spend(_estimates(), **kwargs, is_interactive=lambda: True, ask=lambda q: False)


def test_unknown_price_points_to_config() -> None:
    console, buffer = _console()
    confirm_spend(_estimates(usd=None), console=console, yes=False, dry_run=True)
    assert "pricing:" in buffer.getvalue()
