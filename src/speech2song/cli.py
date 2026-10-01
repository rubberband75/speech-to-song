"""speech2song command-line interface.

Commands map to pipeline steps; a step runs one or more cached stages (see pipeline.py).
Every command prints what it read, what it wrote, and what it spent.
"""

import functools
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.markup import escape
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from speech2song import __version__
from speech2song.audio.io import probe
from speech2song.config import AppConfig, list_presets, load_config, load_preset
from speech2song.costs import CostLog, render_estimates
from speech2song.errors import S2SError
from speech2song.manifest import Run
from speech2song.models import ClipSet, RunOptions
from speech2song.pipeline import Context, Stage, check, execute
from speech2song.review import run_review
from speech2song.selection import clip_targets, estimate_input_tokens_from_duration
from speech2song.stages.align import AlignStage
from speech2song.stages.ingest import IngestStage, IsolateStage
from speech2song.stages.select_clips import (
    CLIPS,
    ClipsStage,
    SelectStage,
    select_estimates,
    write_preview,
    write_review,
)
from speech2song.stages.transcribe import AsrStage

# --- Steps -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Step:
    name: str
    milestone: str
    stages: Callable[[], list[Stage]] | None  # None: not implemented yet


STEPS: list[Step] = [
    Step("ingest", "M1", lambda: [IngestStage(), IsolateStage()]),
    Step("transcribe", "M1", lambda: [AsrStage(), AlignStage()]),
    Step("select", "M2", lambda: [SelectStage(), ClipsStage()]),
    Step("melody", "M3", None),
    Step("arrange", "M4", None),
    Step("generate", "M4", None),
    Step("mix", "M4", None),
]
STEP_BY_NAME = {step.name: step for step in STEPS}


class StepName(StrEnum):
    ingest = "ingest"
    transcribe = "transcribe"
    select = "select"
    melody = "melody"
    arrange = "arrange"
    generate = "generate"
    mix = "mix"


class MusicBackend(StrEnum):
    stub = "stub"
    elevenlabs = "elevenlabs"


# --- App plumbing ----------------------------------------------------------------------

app = typer.Typer(
    help="Turn a spoken-word recording into a finished electronic song.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)
presets_app = typer.Typer(help="Style presets.", no_args_is_help=True)
app.add_typer(presets_app, name="presets")


@dataclass
class Env:
    config: AppConfig
    console: Console
    verbose: bool


def _print_error(exc: Exception) -> None:
    console = Console(stderr=True, highlight=False, soft_wrap=True)
    console.print(Text.assemble(("Error: ", "bold red"), str(exc)))


def cli_errors[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """Report S2SError as a one-line message with exit code 1 instead of a traceback."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return func(*args, **kwargs)
        except S2SError as exc:
            _print_error(exc)
            raise typer.Exit(1) from None

    return wrapper


def _setup_logging(verbose: bool) -> None:
    logger = logging.getLogger("speech2song")
    logger.setLevel(logging.DEBUG)
    for handler in list(logger.handlers):
        if isinstance(handler, RichHandler):
            logger.removeHandler(handler)
    console_handler = RichHandler(console=Console(stderr=True), show_path=False)
    console_handler.setLevel(logging.DEBUG if verbose else logging.WARNING)
    logger.addHandler(console_handler)


def _version(value: bool) -> None:
    if value:
        typer.echo(f"speech2song {__version__}")
        raise typer.Exit()


@app.callback()
def _global(
    ctx: typer.Context,
    config: Annotated[
        Path | None, typer.Option(help="config.yaml to use (default: ./config.yaml if present).")
    ] = None,
    runs_dir: Annotated[Path | None, typer.Option(help="Override runs_dir from config.")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False,
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Show version.")
    ] = False,
) -> None:
    _setup_logging(verbose)
    try:
        cfg = load_config(config)
    except S2SError as exc:
        _print_error(exc)
        raise typer.Exit(1) from None
    if runs_dir is not None:
        cfg = cfg.model_copy(update={"runs_dir": runs_dir})
    width = None if sys.stdout.isatty() else 120  # wider tables in logs and pipes
    ctx.obj = Env(cfg, Console(highlight=False, soft_wrap=True, width=width), verbose)


def _env(ctx: typer.Context) -> Env:
    env = ctx.find_root().obj
    assert isinstance(env, Env)
    return env


RunRef = Annotated[
    str | None,
    typer.Option("--run", help="Run ID, unique ID prefix, path, or 'latest' (default)."),
]
Force = Annotated[bool, typer.Option("--force", help="Re-run stages even if cached.")]


def _open_run(env: Env, ref: str | None) -> Run:
    run = Run.open(env.config.runs_dir, ref)
    shown = "latest run" if ref in (None, "latest") else "run"
    env.console.print(f"Using {shown} [bold]{escape(run.id)}[/] ({escape(str(run.root))})")
    return run


def _create_run(
    env: Env, input_path: Path, transcript: Path | None, preset: str | None, options: RunOptions
) -> Run:
    if not input_path.is_file():
        raise S2SError(f"Input file not found: {input_path}")
    if transcript is not None and not transcript.is_file():
        raise S2SError(f"Transcript file not found: {transcript}")
    preset_name = preset or env.config.default_preset
    load_preset(preset_name, env.config.presets_dir)  # fail early on a bad preset
    run = Run.create(
        env.config.runs_dir,
        input_path,
        preset=preset_name,
        transcript=transcript,
        options=options,
        settings=env.config.snapshot(),
    )
    env.console.print(f"Created run [bold]{escape(run.id)}[/] ({escape(str(run.root))})")
    return run


def _update_options(run: Run, **changes: Any) -> None:
    """Apply CLI flags to the run's sticky options (None means 'keep')."""
    updates = {key: value for key, value in changes.items() if value is not None}
    if updates:
        run.manifest.options = run.manifest.options.model_copy(update=updates)


def _set_preset(env: Env, run: Run, preset: str | None) -> None:
    if preset is not None:
        load_preset(preset, env.config.presets_dir)
        run.manifest.preset = preset


def _set_transcript(run: Run, transcript: Path | None) -> None:
    if transcript is not None:
        if not transcript.is_file():
            raise S2SError(f"Transcript file not found: {transcript}")
        run.manifest.transcript = run.file_ref(transcript)


def _context(env: Env, run: Run, *, force: bool = False, dry_run: bool = False, yes: bool = False):
    return Context(run, env.config, env.console, force=force, dry_run=dry_run, yes=yes)


Hook = Callable[[Context], None]


def _run_steps(ctx: Context, steps: list[Step], hooks: dict[str, Hook] | None = None) -> bool:
    """Execute steps in order, calling a step's hook after it. Returns False if it
    stopped at an unimplemented step."""
    for step in steps:
        if step.stages is None:
            ctx.say(f"[yellow]Stopping before `{step.name}`: planned for {step.milestone}.[/]")
            return False
        for stage in step.stages():
            execute(stage, ctx)
        if hooks and step.name in hooks:
            hooks[step.name](ctx)
    _dry_run_total(ctx)
    return True


def _dry_run_total(ctx: Context) -> None:
    if not ctx.dry_run:
        return
    if not ctx.estimates:
        ctx.console.print("Dry run: no paid calls would be made.")
        return
    known = sum(e.usd for e in ctx.estimates if e.usd is not None)
    unknown = sum(1 for e in ctx.estimates if e.usd is None)
    suffix = f" plus {unknown} unpriced item(s)" if unknown else ""
    ctx.console.print(
        f"Dry run total, worst case (every retry happens): [bold]${known:.4f}[/]{suffix}. "
        "Nothing was executed."
    )


def _ask_review() -> str:
    try:
        return Prompt.ask("review")
    except EOFError:
        return "quit"


def _review_hook(env: Env, run: Run) -> Hook:
    """After `select`: let the user drop, reorder and audition clips, then re-cut."""

    def hook(ctx: Context) -> None:
        if ctx.dry_run:
            return
        clip_set = ClipSet.model_validate_json(run.path(CLIPS).read_text(encoding="utf-8"))
        state = run_review(
            env.console,
            clip_set,
            ask=_ask_review,
            preview=lambda clip_id: write_preview(run, clip_set, clip_id),
        )
        if state is None:
            env.console.print("Review closed without saving.")
            return
        path = write_review(ctx, state.order, state.dropped)
        env.console.print(f"Saved {escape(path.name)}; applying it.")
        execute(ClipsStage(), ctx)

    return hook


def _spend_line(env: Env, run: Run, before: float) -> None:
    spent = CostLog(run.costs_path).totals()["total"] - before
    env.console.print(f"Spend this command: ${spent:.4f} (run total in costs.json)")


def _run_step_command(
    env: Env,
    run: Run,
    step_name: str,
    *,
    force: bool = False,
    dry_run: bool = False,
    yes: bool = False,
    hooks: dict[str, Hook] | None = None,
) -> None:
    step = STEP_BY_NAME[step_name]
    if step.stages is None:
        env.console.print(f"`{step.name}` is planned for {step.milestone}; not implemented yet.")
        raise typer.Exit(2)
    run.manifest.settings = env.config.snapshot()
    run.save()
    before = CostLog(run.costs_path).totals()["total"]
    with run.logging_to_file(logging.DEBUG if env.verbose else logging.INFO):
        _run_steps(_context(env, run, force=force, dry_run=dry_run, yes=yes), [step], hooks)
    _spend_line(env, run, before)


# --- Pipeline commands -----------------------------------------------------------------


@app.command("run")
@cli_errors
def run_cmd(
    ctx: typer.Context,
    input_path: Annotated[
        Path | None,
        typer.Argument(metavar="INPUT", help="Audio/video file for a new run (omit with --run)."),
    ] = None,
    transcript: Annotated[Path | None, typer.Option(help="Official transcript (.txt).")] = None,
    preset: Annotated[str | None, typer.Option(help="Preset name (default from config).")] = None,
    isolate_voice: Annotated[
        bool | None, typer.Option("--isolate-voice/--no-isolate-voice", help="Use demucs.")
    ] = None,
    clips: Annotated[int | None, typer.Option(min=1, help="Number of clips to select.")] = None,
    music_backend: Annotated[MusicBackend | None, typer.Option(help="Music backend.")] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show what would run and cost; do nothing.")
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Don't ask before paid calls.")] = False,
    stop_after: Annotated[StepName | None, typer.Option(help="Last step to run.")] = None,
    interactive_review: Annotated[
        bool, typer.Option("--interactive-review", help="Review clips after selection.")
    ] = False,
    force: Force = False,
    run_ref: RunRef = None,
) -> None:
    """Run the whole pipeline (or resume a run with --run)."""
    env = _env(ctx)
    if input_path is not None and run_ref is not None:
        raise S2SError("Pass INPUT to start a new run, or --run to resume one, not both.")
    last = STEPS.index(STEP_BY_NAME[stop_after.value]) if stop_after else len(STEPS) - 1
    steps = STEPS[: last + 1]
    options = {
        "isolate_voice": isolate_voice,
        "clips": clips,
        "music_backend": music_backend.value if music_backend else None,
    }
    if input_path is None:
        run = _open_run(env, run_ref)
        _update_options(run, **options)
        _set_preset(env, run, preset)
        _set_transcript(run, transcript)
    elif dry_run:
        _dry_run_new_input(env, input_path, preset, clips, steps)
        return
    else:
        initial = RunOptions(**{k: v for k, v in options.items() if v is not None})
        run = _create_run(env, input_path, transcript, preset, initial)
    run.manifest.settings = env.config.snapshot()
    run.save()
    before = CostLog(run.costs_path).totals()["total"]
    with run.logging_to_file(logging.DEBUG if env.verbose else logging.INFO):
        context = _context(env, run, force=force, dry_run=dry_run, yes=yes)
        if dry_run:
            env.console.print("Dry run: nothing will be executed.")
        hooks = {"select": _review_hook(env, run)} if interactive_review else None
        _run_steps(context, steps, hooks)
    _spend_line(env, run, before)


def _dry_run_new_input(
    env: Env, input_path: Path, preset: str | None, clips: int | None, steps: list[Step]
) -> None:
    """Estimate a new run's spend from the input's duration (ffprobe only, no stages)."""
    if not input_path.is_file():
        raise S2SError(f"Input file not found: {input_path}")
    names = ", ".join(step.name for step in steps)
    env.console.print(f"Dry run: would create a new run for {escape(str(input_path))}")
    env.console.print(f"and run: {names}.")
    if not any(step.name == "select" for step in steps):
        env.console.print("No paid calls before the steps you asked for. Nothing was executed.")
        return
    preset_obj = load_preset(preset or env.config.default_preset, env.config.presets_dir)
    targets = clip_targets(preset_obj, RunOptions(clips=clips))
    try:
        seconds = probe(input_path).duration_s or 900.0
    except S2SError as exc:
        env.console.print(f"[yellow]Could not read the input ({escape(str(exc))}); "
                          "assuming a 15-minute talk.[/]")  # fmt: skip
        seconds = 900.0
    estimates = select_estimates(
        env.config,
        env.config.claude_model,
        estimate_input_tokens_from_duration(seconds),
        targets.count,
        f"estimated from {seconds / 60:.0f} min of audio",
    )
    render_estimates(env.console, estimates)
    total = sum(e.usd or 0.0 for e in estimates)
    env.console.print(
        f"Dry run total, worst case (every retry happens): [bold]${total:.4f}[/]. "
        "Music generation is not implemented yet (M4/M5). Nothing was executed."
    )


@app.command()
@cli_errors
def ingest(
    ctx: typer.Context,
    input_path: Annotated[
        Path | None,
        typer.Argument(metavar="INPUT", help="Audio/video file for a new run (omit with --run)."),
    ] = None,
    transcript: Annotated[Path | None, typer.Option(help="Official transcript (.txt).")] = None,
    preset: Annotated[str | None, typer.Option(help="Preset name (default from config).")] = None,
    isolate_voice: Annotated[
        bool | None, typer.Option("--isolate-voice/--no-isolate-voice", help="Use demucs.")
    ] = None,
    force: Force = False,
    run_ref: RunRef = None,
) -> None:
    """Extract 00_source.wav (and the optional voice-isolated 01_clean.wav)."""
    env = _env(ctx)
    if input_path is not None and run_ref is not None:
        raise S2SError("Pass INPUT to start a new run, or --run to re-ingest one, not both.")
    if input_path is None:
        run = _open_run(env, run_ref)
        _update_options(run, isolate_voice=isolate_voice)
        _set_preset(env, run, preset)
        _set_transcript(run, transcript)
    else:
        options = RunOptions(isolate_voice=bool(isolate_voice))
        run = _create_run(env, input_path, transcript, preset, options)
    _run_step_command(env, run, "ingest", force=force)


@app.command()
@cli_errors
def transcribe(
    ctx: typer.Context,
    transcript: Annotated[
        Path | None, typer.Option(help="Official transcript (.txt) to align to.")
    ] = None,
    no_transcript: Annotated[
        bool, typer.Option("--no-transcript", help="Stop using the official transcript.")
    ] = False,
    whisper_model: Annotated[str | None, typer.Option(help="Whisper model for this run.")] = None,
    language: Annotated[str | None, typer.Option(help="Language code (default: detect).")] = None,
    force: Force = False,
    run_ref: RunRef = None,
) -> None:
    """Transcribe with local Whisper and align to the official transcript if given."""
    env = _env(ctx)
    if transcript is not None and no_transcript:
        raise S2SError("Use either --transcript or --no-transcript.")
    run = _open_run(env, run_ref)
    _set_transcript(run, transcript)
    if no_transcript:
        run.manifest.transcript = None
    _update_options(run, whisper_model=whisper_model, language=language)
    _run_step_command(env, run, "transcribe", force=force)


@app.command()
@cli_errors
def select(
    ctx: typer.Context,
    clips: Annotated[int | None, typer.Option(min=1, help="Number of clips to select.")] = None,
    model: Annotated[str | None, typer.Option(help="Claude model for this run.")] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show the estimated cost; call nothing.")
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Don't ask before paid calls.")] = False,
    interactive_review: Annotated[
        bool, typer.Option("--interactive-review", help="Drop, reorder or audition clips.")
    ] = False,
    force: Force = False,
    run_ref: RunRef = None,
) -> None:
    """Ask Claude for the best lines, then cut them as sample-exact clips."""
    env = _env(ctx)
    run = _open_run(env, run_ref)
    _update_options(run, clips=clips, claude_model=model)
    hooks = {"select": _review_hook(env, run)} if interactive_review else None
    _run_step_command(env, run, "select", force=force, dry_run=dry_run, yes=yes, hooks=hooks)


def _stub_command(name: str) -> None:
    step = STEP_BY_NAME[name]

    @cli_errors
    def command(ctx: typer.Context, run_ref: RunRef = None) -> None:
        env = _env(ctx)
        _run_step_command(env, _open_run(env, run_ref), name)

    command.__doc__ = f"Pipeline step `{name}` (planned for {step.milestone})."
    app.command(name)(command)


for _name in ("melody", "arrange", "generate", "mix"):
    _stub_command(_name)


@app.command()
@cli_errors
def regenerate(
    ctx: typer.Context,
    section: Annotated[str, typer.Option(help="Arrangement section ID, e.g. s5.")],
    note: Annotated[str | None, typer.Option(help="Guidance for the new take.")] = None,
    run_ref: RunRef = None,
) -> None:
    """Regenerate one music section via inpainting (planned for M5)."""
    _env(ctx).console.print("`regenerate` is planned for M5; not implemented yet.")
    raise typer.Exit(2)


# --- Inspection commands ---------------------------------------------------------------


@app.command()
@cli_errors
def costs(ctx: typer.Context, run_ref: RunRef = None) -> None:
    """Show the paid calls logged for a run."""
    env = _env(ctx)
    run = _open_run(env, run_ref)
    log_ = CostLog(run.costs_path)
    entries = log_.read()
    if not entries:
        env.console.print("No paid calls recorded for this run. Spend: $0.0000")
        return
    table = Table(title=f"Paid calls · {run.id}")
    for column in ("Time", "Stage", "Service", "Model", "Units", "USD"):
        table.add_column(column)
    for entry in entries:
        units = ", ".join(f"{k}={v:,.0f}" for k, v in entry.units.items())
        usd = f"${entry.usd:.4f}" + (" (est.)" if entry.estimated else "")
        table.add_row(
            f"{entry.ts:%Y-%m-%d %H:%M}", entry.stage, entry.service, entry.model, units, usd
        )
    env.console.print(table)
    totals = log_.totals()
    parts = [f"{service} ${usd:.4f}" for service, usd in totals.items() if service != "total"]
    env.console.print(f"Total: [bold]${totals['total']:.4f}[/] ({', '.join(parts)})")


@app.command()
@cli_errors
def status(ctx: typer.Context, run_ref: RunRef = None) -> None:
    """Show each stage's cache state for a run."""
    env = _env(ctx)
    run = _open_run(env, run_ref)
    manifest = run.manifest
    env.console.print(f"Input: {escape(manifest.input.path)}")
    transcript = manifest.transcript.path if manifest.transcript else "none"
    env.console.print(f"Transcript: {escape(transcript)} · preset: {manifest.preset}")
    table = Table()
    for column in ("Step", "Stage", "State", "Finished", "Took"):
        table.add_column(column)
    context = _context(env, run)
    for step in STEPS:
        if step.stages is None:
            table.add_row(step.name, "-", f"planned for {step.milestone}", "", "")
            continue
        for stage in step.stages():
            state = check(stage, context).state
            record = manifest.stages.get(stage.name)
            finished = took = ""
            if record is not None and record.finished_at is not None:
                finished = f"{record.finished_at:%Y-%m-%d %H:%M}"
            if record is not None and record.elapsed_s is not None:
                took = f"{record.elapsed_s:.1f} s"
            table.add_row(step.name, stage.name, state.value, finished, took)
    run.save()  # keeps freshly computed hashes in the memo
    env.console.print(table)
    env.console.print(f"Spend so far: ${CostLog(run.costs_path).totals()['total']:.4f}")


@presets_app.command("list")
@cli_errors
def presets_list(ctx: typer.Context) -> None:
    """List presets and whether they validate."""
    env = _env(ctx)
    listings = list_presets(env.config.presets_dir)
    if not listings:
        env.console.print(f"No presets found in {escape(str(env.config.presets_dir))}")
        return
    table = Table()
    for column in ("Name", "Description", "Status"):
        table.add_column(column)
    for item in listings:
        if item.preset is not None:
            description = " ".join(item.preset.description.split())
            table.add_row(item.name, description, "[green]ok[/]")
        else:
            first_line = (item.error or "").splitlines()[0]
            table.add_row(item.name, "", f"[red]invalid[/]: {escape(first_line)}")
    env.console.print(table)


def main() -> None:
    app()
