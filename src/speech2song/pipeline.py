"""Stage framework: fingerprints, the cache rule, and execution with read/wrote/spend reports.

A stage is skipped when its last run completed with the same fingerprint (stage name,
stage version, input content hashes, params) and its recorded outputs still exist.
A stage's own outputs are not re-hashed for that decision, so hand edits to an output
are kept; downstream stages see the edit through their input hashes.

A stage can also wait for files it does not hash (`StagePlan.waits_for`). It then puts a
digest of what it takes from them into its params instead, so edits elsewhere in those
files don't invalidate it. Paid stages use this to repeat a call only when the request
itself changes.
"""

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar, Literal

from rich.console import Console
from rich.markup import escape
from rich.text import Text

from speech2song.config import AppConfig
from speech2song.costs import CostLog, SpendEstimate, confirm_spend, render_estimates
from speech2song.errors import StageError
from speech2song.manifest import Run
from speech2song.models import FileRef, StageRecord

log = logging.getLogger(__name__)


@dataclass
class StagePlan:
    """What a stage will read and write. Built without touching input contents."""

    inputs: dict[str, Path]
    params: dict[str, Any]
    outputs: list[str]  # run-relative paths
    waits_for: dict[str, Path] = field(default_factory=dict)  # must exist; not hashed


@dataclass
class StageResult:
    outputs: list[str] | None = None  # overrides plan.outputs when only known after running
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class Context:
    run: Run
    config: AppConfig
    console: Console
    force: bool = False
    dry_run: bool = False
    yes: bool = False
    estimates: list[SpendEstimate] = field(default_factory=list)  # collected by dry runs

    def say(self, message: str) -> None:
        """Print for the user and mirror into the run log."""
        self.console.print(message)
        log.info(Text.from_markup(message).plain)


class Stage(ABC):
    name: ClassVar[str]
    version: ClassVar[int] = 1  # bump when output-affecting logic changes
    paid: ClassVar[bool] = False

    @abstractmethod
    def plan(self, ctx: Context) -> StagePlan: ...

    @abstractmethod
    def run(self, ctx: Context, plan: StagePlan) -> StageResult: ...

    def estimate(self, ctx: Context, plan: StagePlan) -> list[SpendEstimate]:
        """Paid stages return their expected spend; free stages spend nothing."""
        return []

    def is_paid(self, ctx: Context) -> bool:
        """Whether running this stage for this run calls a paid API (may depend on options)."""
        return self.paid


def fingerprint(
    name: str, version: int, input_hashes: dict[str, str], params: dict[str, Any]
) -> str:
    payload = {"stage": name, "version": version, "inputs": input_hashes, "params": params}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


class State(StrEnum):
    COMPLETE = "complete"
    STALE = "stale"
    MISSING_OUTPUTS = "missing outputs"
    NOT_RUN = "not run"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    BLOCKED = "waiting for inputs"


@dataclass
class Check:
    state: State
    plan: StagePlan
    fingerprint: str | None = None
    input_refs: dict[str, FileRef] = field(default_factory=dict)
    missing_inputs: list[str] = field(default_factory=list)


def check(stage: Stage, ctx: Context) -> Check:
    """Evaluate a stage's cache state (hashes its inputs, memoized)."""
    plan = stage.plan(ctx)
    missing = [str(p) for p in {**plan.inputs, **plan.waits_for}.values() if not p.exists()]
    if missing:
        return Check(State.BLOCKED, plan, missing_inputs=missing)
    refs = {key: ctx.run.file_ref(path) for key, path in plan.inputs.items()}
    fp = fingerprint(stage.name, stage.version, {k: r.sha256 for k, r in refs.items()}, plan.params)
    record = ctx.run.manifest.stages.get(stage.name)
    if record is None:
        state = State.NOT_RUN
    elif record.status == "failed":
        state = State.FAILED
    elif record.status == "running":
        state = State.INTERRUPTED
    elif record.fingerprint != fp:
        state = State.STALE
    elif not all(ctx.run.path(out).exists() for out in record.outputs):
        state = State.MISSING_OUTPUTS
    else:
        state = State.COMPLETE
    return Check(state, plan, fp, refs)


@dataclass
class Outcome:
    stage: str
    action: Literal["ran", "cached", "would run"]
    record: StageRecord | None = None


def fmt_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1000 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1000
    raise AssertionError("unreachable")


def _dry_run_estimate(stage: Stage, ctx: Context, plan: StagePlan) -> None:
    if stage.is_paid(ctx):
        estimates = stage.estimate(ctx, plan)
        render_estimates(ctx.console, estimates)
        ctx.estimates.extend(estimates)


def execute(stage: Stage, ctx: Context) -> Outcome:
    """Run one stage unless its cached outputs are still valid."""
    run = ctx.run
    chk = check(stage, ctx)
    if chk.state is State.BLOCKED:
        if ctx.dry_run:
            ctx.say(f"  {stage.name}: would run (after earlier stages produce its inputs)")
            _dry_run_estimate(stage, ctx, chk.plan)
            return Outcome(stage.name, "would run")
        raise StageError(f"{stage.name}: missing inputs: {', '.join(chk.missing_inputs)}")
    if chk.state is State.COMPLETE and not ctx.force:
        ctx.say(f"[green]✓[/] {stage.name}: cached (inputs unchanged)")
        return Outcome(stage.name, "cached", run.manifest.stages[stage.name])

    reason = "forced" if chk.state is State.COMPLETE else chk.state.value
    if ctx.dry_run:
        ctx.say(f"  {stage.name}: would run ({reason})")
        _dry_run_estimate(stage, ctx, chk.plan)
        return Outcome(stage.name, "would run")
    if stage.is_paid(ctx):
        confirm_spend(
            stage.estimate(ctx, chk.plan), console=ctx.console, yes=ctx.yes, dry_run=False
        )

    assert chk.fingerprint is not None
    ctx.say(f"[bold]▶ {stage.name}[/] ({reason})")
    record = StageRecord(
        status="running",
        version=stage.version,
        fingerprint=chk.fingerprint,
        inputs=chk.input_refs,
        params=chk.plan.params,
        started_at=datetime.now().astimezone(),
    )
    run.manifest.stages[stage.name] = record
    run.save()
    costs = CostLog(run.costs_path)
    spent_before = costs.totals()["total"]
    t0 = time.monotonic()
    try:
        result = stage.run(ctx, chk.plan)
        outputs = result.outputs if result.outputs is not None else chk.plan.outputs
        missing = [out for out in outputs if not run.path(out).exists()]
        if missing:
            raise StageError(f"{stage.name} did not produce: {', '.join(missing)}")
        record.outputs = {out: run.file_ref(run.path(out)) for out in outputs}
    except BaseException as exc:
        record.status = "failed"
        record.error = f"{type(exc).__name__}: {exc}"
        record.finished_at = datetime.now().astimezone()
        record.elapsed_s = round(time.monotonic() - t0, 3)
        run.save()
        raise
    record.status = "complete"
    record.summary = result.summary
    record.finished_at = datetime.now().astimezone()
    record.elapsed_s = round(time.monotonic() - t0, 3)
    run.save()

    for ref in chk.input_refs.values():
        ctx.say(f"  read  {escape(ref.path)} ({fmt_bytes(ref.size)}, sha256 {ref.sha256[:12]})")
    for ref in record.outputs.values():
        ctx.say(f"  wrote {escape(ref.path)} ({fmt_bytes(ref.size)})")
    spent = costs.totals()["total"] - spent_before
    ctx.say(f"[green]✓[/] {stage.name} done in {record.elapsed_s:.1f} s · spend ${spent:.4f}")
    return Outcome(stage.name, "ran", record)
