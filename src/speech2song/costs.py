"""Paid-call bookkeeping: the costs.json log, price lookups, and spend confirmation."""

import json
import sys
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, TypeAdapter
from rich.console import Console
from rich.table import Table

from speech2song.config import PricingConfig
from speech2song.errors import SpendDeclinedError
from speech2song.manifest import write_json
from speech2song.models import CostEntry

_ENTRIES = TypeAdapter(list[CostEntry])


class CostLog:
    """Append-only list of paid calls, stored as a JSON array and rewritten atomically."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def read(self) -> list[CostEntry]:
        if not self.path.is_file():
            return []
        return _ENTRIES.validate_json(self.path.read_bytes())

    def append(self, entry: CostEntry) -> None:
        entries = self.read()
        entries.append(entry)
        write_json(self.path, json.loads(_ENTRIES.dump_json(entries)))

    def totals(self) -> dict[str, float]:
        """USD per service, plus 'total'."""
        totals: dict[str, float] = {}
        for entry in self.read():
            totals[entry.service] = totals.get(entry.service, 0.0) + entry.usd
        totals["total"] = sum(totals.values())
        return totals


def claude_usd(
    pricing: PricingConfig, model: str, input_tokens: int, output_tokens: int
) -> float | None:
    """USD for a Claude call, or None when the model has no configured price."""
    price = pricing.anthropic.get(model)
    if price is None:
        return None
    return (
        input_tokens * price.input_usd_per_mtok + output_tokens * price.output_usd_per_mtok
    ) / 1_000_000


def unit_usd(pricing: PricingConfig, item: str, units: float) -> float | None:
    """USD for a unit-priced ElevenLabs item (e.g. music minutes), or None if unpriced."""
    price = pricing.elevenlabs.get(item)
    if price is None:
        return None
    return units * price.usd_per_unit


class SpendEstimate(BaseModel):
    service: str
    model: str
    description: str
    units: dict[str, float]
    usd: float | None  # None = no price configured


def render_estimates(console: Console, estimates: list[SpendEstimate]) -> float | None:
    """Print a table of estimates. Returns the total, or None if any price is unknown."""
    table = Table(title="Estimated spend", show_lines=False)
    for column in ("Service", "Model", "What", "Units", "USD"):
        table.add_column(column)
    total: float | None = 0.0
    for est in estimates:
        units = ", ".join(f"{k}={v:,.0f}" for k, v in est.units.items())
        usd = "unknown" if est.usd is None else f"${est.usd:.4f}"
        table.add_row(est.service, est.model, est.description, units, usd)
        total = None if total is None or est.usd is None else total + est.usd
    console.print(table)
    if total is None:
        console.print(
            "[yellow]Some prices are not configured; set them under `pricing:` in config.yaml.[/]"
        )
    else:
        console.print(f"Estimated total: [bold]${total:.4f}[/]")
    return total


def confirm_spend(
    estimates: list[SpendEstimate],
    *,
    console: Console,
    yes: bool,
    dry_run: bool,
    is_interactive: Callable[[], bool] = lambda: sys.stdin.isatty(),
    ask: Callable[[str], bool] | None = None,
) -> bool:
    """Show the estimate, then decide whether paid calls may proceed.

    Dry runs never proceed. `--yes` proceeds without asking. Otherwise the user is
    asked; without a TTY to ask on, this refuses rather than spending implicitly.
    """
    render_estimates(console, estimates)
    if dry_run:
        console.print("[cyan]Dry run: no paid calls made.[/]")
        return False
    if yes:
        return True
    if not is_interactive():
        raise SpendDeclinedError("Paid calls need confirmation: pass --yes in non-interactive use.")
    if ask is None:
        from rich.prompt import Confirm

        def ask(question: str) -> bool:
            return Confirm.ask(question, console=console, default=False)

    if not ask("Proceed with these paid calls?"):
        raise SpendDeclinedError("Paid calls declined.")
    return True
