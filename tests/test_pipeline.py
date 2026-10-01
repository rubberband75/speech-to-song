import io
from collections.abc import Callable
from typing import ClassVar

import pytest

from speech2song.costs import SpendEstimate
from speech2song.errors import SpendDeclinedError, StageError
from speech2song.manifest import Run, atomic_write_text
from speech2song.pipeline import (
    Context,
    Stage,
    StagePlan,
    StageResult,
    State,
    check,
    execute,
    fingerprint,
)

MakeContext = Callable[..., Context]


class Upper(Stage):
    """Toy stage: uppercases one file into another."""

    name: ClassVar[str] = "upper"

    def __init__(self, src: str = "in.txt", dst: str = "a.txt", suffix: str = "") -> None:
        self.src, self.dst, self.suffix = src, dst, suffix
        self.calls = 0

    def plan(self, ctx: Context) -> StagePlan:
        return StagePlan({"src": ctx.run.path(self.src)}, {"suffix": self.suffix}, [self.dst])

    def run(self, ctx: Context, plan: StagePlan) -> StageResult:
        self.calls += 1
        text = plan.inputs["src"].read_text().upper() + self.suffix
        atomic_write_text(ctx.run.path(self.dst), text)
        return StageResult(summary={"chars": len(text)})


class UpperV2(Upper):
    version: ClassVar[int] = 2


class Downstream(Upper):
    name: ClassVar[str] = "downstream"

    def __init__(self) -> None:
        super().__init__(src="a.txt", dst="b.txt", suffix="!")


class Exploding(Upper):
    name: ClassVar[str] = "exploding"

    def run(self, ctx: Context, plan: StagePlan) -> StageResult:
        raise ValueError("boom")


class Lazy(Upper):
    name: ClassVar[str] = "lazy"

    def run(self, ctx: Context, plan: StagePlan) -> StageResult:
        return StageResult()  # forgets to write its output


class Paid(Upper):
    name: ClassVar[str] = "paid"
    paid: ClassVar[bool] = True

    def estimate(self, ctx: Context, plan: StagePlan) -> list[SpendEstimate]:
        return [SpendEstimate(service="anthropic", model="m", description="d", units={}, usd=0.5)]


@pytest.fixture
def source(run: Run) -> Run:
    atomic_write_text(run.path("in.txt"), "hello")
    return run


def test_runs_once_then_caches(source: Run, make_context: MakeContext) -> None:
    stage = Upper()
    first = execute(stage, make_context(source))
    assert first.action == "ran"
    assert source.path("a.txt").read_text() == "HELLO"
    record = source.manifest.stages["upper"]
    assert record.status == "complete"
    assert record.summary == {"chars": 5}
    assert set(record.outputs) == {"a.txt"}
    reopened = Run.open(source.root.parent, source.id)
    assert execute(stage, make_context(reopened)).action == "cached"
    assert stage.calls == 1


@pytest.mark.parametrize(
    "change",
    ["input", "params", "version", "missing_output", "force"],
)
def test_reruns_when_something_changed(source: Run, make_context: MakeContext, change: str) -> None:
    execute(Upper(), make_context(source))
    stage: Upper = Upper()
    flags = {}
    if change == "input":
        atomic_write_text(source.path("in.txt"), "changed")
    elif change == "params":
        stage = Upper(suffix="?")
    elif change == "version":
        stage = UpperV2()
    elif change == "missing_output":
        source.path("a.txt").unlink()
    else:
        flags = {"force": True}
    assert execute(stage, make_context(source, **flags)).action == "ran"
    assert stage.calls == 1


def test_failure_is_recorded_and_retried(source: Run, make_context: MakeContext) -> None:
    with pytest.raises(ValueError, match="boom"):
        execute(Exploding(), make_context(source))
    record = source.manifest.stages["exploding"]
    assert record.status == "failed"
    assert record.error == "ValueError: boom"
    assert check(Upper(), make_context(source)).state is State.NOT_RUN
    assert check(Exploding(), make_context(source)).state is State.FAILED


def test_missing_declared_output_fails_the_stage(source: Run, make_context: MakeContext) -> None:
    with pytest.raises(StageError, match=r"did not produce: a\.txt"):
        execute(Lazy(), make_context(source))
    assert source.manifest.stages["lazy"].status == "failed"


def test_interrupted_stage_reruns(source: Run, make_context: MakeContext) -> None:
    execute(Upper(), make_context(source))
    source.manifest.stages["upper"].status = "running"
    assert check(Upper(), make_context(source)).state is State.INTERRUPTED
    assert execute(Upper(), make_context(source)).action == "ran"


def test_downstream_reruns_only_when_upstream_bytes_change(
    source: Run, make_context: MakeContext
) -> None:
    ctx = make_context(source)
    execute(Upper(), ctx)
    execute(Downstream(), ctx)
    assert source.path("b.txt").read_text() == "HELLO!"

    execute(Upper(), make_context(source, force=True))  # identical bytes again
    assert execute(Downstream(), ctx).action == "cached"

    atomic_write_text(source.path("in.txt"), "bye")
    assert execute(Upper(), ctx).action == "ran"
    assert execute(Downstream(), ctx).action == "ran"
    assert source.path("b.txt").read_text() == "BYE!"


def test_hand_edited_output_is_kept_and_propagates(source: Run, make_context: MakeContext) -> None:
    ctx = make_context(source)
    execute(Upper(), ctx)
    execute(Downstream(), ctx)
    atomic_write_text(source.path("a.txt"), "EDITED BY HAND")
    assert execute(Upper(), ctx).action == "cached"
    assert source.path("a.txt").read_text() == "EDITED BY HAND"
    assert execute(Downstream(), ctx).action == "ran"
    assert source.path("b.txt").read_text() == "EDITED BY HAND!"


def test_missing_input_blocks_unless_dry_run(run: Run, make_context: MakeContext) -> None:
    with pytest.raises(StageError, match="missing inputs"):
        execute(Upper(), make_context(run))
    assert execute(Upper(), make_context(run, dry_run=True)).action == "would run"


def test_dry_run_executes_nothing(
    source: Run, make_context: MakeContext, console_output: io.StringIO
) -> None:
    stage = Upper()
    outcome = execute(stage, make_context(source, dry_run=True))
    assert outcome.action == "would run"
    assert stage.calls == 0
    assert "upper" not in source.manifest.stages
    assert "would run (not run)" in console_output.getvalue()


def test_paid_stage_dry_run_shows_estimate(
    source: Run, make_context: MakeContext, console_output: io.StringIO
) -> None:
    stage = Paid()
    execute(stage, make_context(source, dry_run=True))
    assert stage.calls == 0
    assert "$0.5000" in console_output.getvalue()


def test_paid_stage_needs_yes_without_tty(source: Run, make_context: MakeContext) -> None:
    stage = Paid()
    with pytest.raises(SpendDeclinedError, match="--yes"):
        execute(stage, make_context(source))
    assert stage.calls == 0
    assert "paid" not in source.manifest.stages
    assert execute(stage, make_context(source, yes=True)).action == "ran"


def test_reports_read_wrote_and_spend(
    source: Run, make_context: MakeContext, console_output: io.StringIO
) -> None:
    execute(Upper(), make_context(source))
    out = console_output.getvalue()
    assert "read  in.txt" in out
    assert "wrote a.txt (5 B)" in out
    assert "spend $0.0000" in out


def test_fingerprint_ignores_key_order() -> None:
    a = fingerprint("s", 1, {"x": "1", "y": "2"}, {"p": 1, "q": [1, 2]})
    b = fingerprint("s", 1, {"y": "2", "x": "1"}, {"q": [1, 2], "p": 1})
    assert a == b
    assert a != fingerprint("s", 1, {"x": "1", "y": "2"}, {"p": 2, "q": [1, 2]})
