"""Stage 5: the arrangement. Optionally Claude refines the arc (paid), then the sections
are laid out on the bar grid (free).

`arc` runs only with `--refine-arc` and writes Claude's raw answers to 05_arc.json. Its
cache key is the request it would send, so it repeats the call only when the prompt
changes. `arrange` writes 05_arrangement.json; it checks Claude's plan again and uses the
preset's arc whenever the plan can't be used.
"""

from pathlib import Path
from typing import ClassVar

from rich.markup import escape
from rich.table import Table

from speech2song.arrangement import (
    ARC_OUTPUT_TOKENS,
    ClipInfo,
    arc_feedback_prompt,
    arc_schema,
    arrangement_warnings,
    build_arc_prompt,
    build_arrangement,
    check_parts,
    clip_info,
    default_parts,
    estimate_arc_input_tokens,
    timeline_rows,
    timeline_strip,
    validate_arrangement,
)
from speech2song.config import AppConfig, load_preset, load_secrets
from speech2song.costs import CostLog, SpendEstimate, claude_usd
from speech2song.errors import S2SError, StageError
from speech2song.llm.claude import ClaudeJson, request_digest
from speech2song.manifest import write_json
from speech2song.models import ArcAttempt, ArcPlan, ArcResult, Arrangement, ClipSet, Melody
from speech2song.pipeline import Context, Stage, StagePlan, StageResult
from speech2song.stages.melody import MELODY, kept_clips
from speech2song.stages.select_clips import CLIPS, claude_model

ARC = "05_arc.json"
ARRANGEMENT = "05_arrangement.json"


def _clips(ctx: Context) -> tuple[ClipSet, list[ClipInfo]]:
    clip_set = ClipSet.model_validate_json(ctx.run.path(CLIPS).read_text(encoding="utf-8"))
    return clip_set, [clip_info(clip) for clip in kept_clips(clip_set)]


def usable_arrangement(ctx: Context) -> Arrangement:
    """05_arrangement.json, checked for hand-edit mistakes before music is generated,
    chosen or mixed for it."""
    arrangement = Arrangement.model_validate_json(
        ctx.run.path(ARRANGEMENT).read_text(encoding="utf-8")
    )
    clip_set = ClipSet.model_validate_json(ctx.run.path(CLIPS).read_text(encoding="utf-8"))
    problems = validate_arrangement(arrangement, {c.id: clip_info(c) for c in clip_set.clips})
    if problems:
        raise StageError(f"{ARRANGEMENT} can't be used: " + "; ".join(problems))
    return arrangement


def _melody(ctx: Context) -> Melody:
    return Melody.model_validate_json(ctx.run.path(MELODY).read_text(encoding="utf-8"))


ARC_GUESS_INPUT_TOKENS = 3000  # before the clips exist (the prompt is about 2,000)


def arc_estimates(
    config: AppConfig,
    model: str,
    input_tokens: int = ARC_GUESS_INPUT_TOKENS,
    basis: str = "rough guess before the clips exist",
) -> list[SpendEstimate]:
    """Spend for the arc call plus the retry that happens only if the answer fails."""
    units = {"input_tokens": input_tokens, "output_tokens": ARC_OUTPUT_TOKENS}
    usd = claude_usd(config.pricing, model, input_tokens, ARC_OUTPUT_TOKENS)
    return [
        SpendEstimate(service="anthropic", model=model, units=units, usd=usd,
                      description=f"refine the arc ({basis})"),
        SpendEstimate(service="anthropic", model=model, units=units, usd=usd,
                      description="retry, only if the answer fails the checks"),
    ]  # fmt: skip


class ArcStage(Stage):
    """Claude proposes the song's arc around the clips (opt-in: `--refine-arc`)."""

    name: ClassVar[str] = "arc"
    version: ClassVar[int] = 2  # 2 (M7): treatments, styles per part, the ending
    paid: ClassVar[bool] = True

    def _prompt(self, ctx: Context) -> tuple[str, str, dict[str, object]] | None:
        if not (ctx.run.path(CLIPS).exists() and ctx.run.path(MELODY).exists()):
            return None
        preset = load_preset(ctx.run.manifest.preset, ctx.config.presets_dir)
        clip_set, clips = _clips(ctx)
        system, user = build_arc_prompt(preset, clips, _melody(ctx).bpm, clip_set.notes)
        return system, user, arc_schema(list(preset.section_roles), [c.id for c in clips])

    def plan(self, ctx: Context) -> StagePlan:
        prompt = self._prompt(ctx)
        model = claude_model(ctx)
        request = None
        if prompt is not None:
            claude = ClaudeJson(ctx.config, CostLog(ctx.run.costs_path), ctx.run.id, model)
            request = request_digest(claude.request(*prompt))
        waits = {"clips": ctx.run.path(CLIPS), "melody": ctx.run.path(MELODY)}
        params = {"model": model, "request": request}
        return StagePlan({}, params, [ARC], waits_for=waits)

    def estimate(self, ctx: Context, plan: StagePlan) -> list[SpendEstimate]:
        prompt = self._prompt(ctx)
        if prompt is None:
            return arc_estimates(ctx.config, claude_model(ctx))
        tokens = estimate_arc_input_tokens(prompt[0], prompt[1])
        return arc_estimates(ctx.config, claude_model(ctx), tokens, "from the prompt")

    def run(self, ctx: Context, plan: StagePlan) -> StageResult:
        prompt = self._prompt(ctx)
        if prompt is None:
            raise StageError("arc: the clips and melody must exist first")
        system, user, schema = prompt
        preset = load_preset(ctx.run.manifest.preset, ctx.config.presets_dir)
        _, clips = _clips(ctx)
        load_secrets()
        claude = ClaudeJson(ctx.config, CostLog(ctx.run.costs_path), ctx.run.id, claude_model(ctx))
        attempts: list[ArcAttempt] = []

        def ask(text: str) -> tuple[ArcPlan, list[str]]:
            ctx.say(f"  asking {escape(claude.model)} to refine the arc "
                    f"(attempt {len(attempts) + 1})")  # fmt: skip
            reply = claude.call(system=system, user=text, schema=schema, stage=self.name,
                                operation="messages.create")  # fmt: skip
            try:
                answer = ArcPlan.model_validate(reply.data)
            except ValueError as exc:
                raise S2SError(f"Claude's answer did not match the arc schema: {exc}") from exc
            problems = check_parts(answer.parts, clips, preset)
            attempts.append(
                ArcAttempt(attempt=len(attempts) + 1, model=reply.model,
                           request_id=reply.request_id, input_tokens=reply.input_tokens,
                           output_tokens=reply.output_tokens, usd=reply.usd, answer=answer,
                           problems=problems)
            )  # fmt: skip
            if reply.fallback_from:
                ctx.say(f"[yellow]  {reply.fallback_from} declined; {reply.model} answered[/]")
            return answer, problems

        answer, problems = ask(user)
        chosen = 0
        if problems:
            ctx.say(f"[yellow]  the arc had {len(problems)} problem(s); asking once more[/]")
            for problem in problems:
                ctx.say(f"    - {escape(problem)}")
            _, second = ask(arc_feedback_prompt(user, answer.parts, problems))
            if len(second) <= len(problems):
                chosen = 1
        write_json(ctx.run.path(ARC), ArcResult(model=claude.model, attempts=attempts,
                                                chosen=chosen))  # fmt: skip
        final = attempts[chosen]
        if final.problems:
            ctx.say("[yellow]  Claude's arc still has problems; `arrange` will use the "
                    "preset's arc[/]")  # fmt: skip
        elif final.answer and final.answer.notes:
            ctx.say(f"  notes: {escape(final.answer.notes)}")
        spent = sum(a.usd or 0.0 for a in attempts)
        return StageResult(summary={"attempts": len(attempts), "usable": not final.problems,
                                    "usd": spent})  # fmt: skip


class ArrangeStage(Stage):
    name: ClassVar[str] = "arrange"
    # 2: sections carry `silent`; 3 (M7): lead-ins, treatments, progressions, development,
    # melody references, the ending (plan version 2)
    version: ClassVar[int] = 3

    def plan(self, ctx: Context) -> StagePlan:
        preset = load_preset(ctx.run.manifest.preset, ctx.config.presets_dir)
        inputs: dict[str, Path] = {"clips": ctx.run.path(CLIPS), "melody": ctx.run.path(MELODY)}
        if ctx.run.manifest.options.refine_arc:
            inputs["arc"] = ctx.run.path(ARC)
        params = {
            "arc": preset.arc,
            "section_roles": {k: v.model_dump() for k, v in preset.section_roles.items()},
            "speech_interaction": preset.speech_interaction.model_dump(),
            "ending": preset.ending,
            "layer_roles": preset.melody.layer_roles,
        }
        return StagePlan(inputs, params, [ARRANGEMENT])

    def run(self, ctx: Context, plan: StagePlan) -> StageResult:
        preset = load_preset(ctx.run.manifest.preset, ctx.config.presets_dir)
        clip_set, clips = _clips(ctx)
        if not clips:
            raise StageError("No clips to arrange (all were dropped).")
        melody = _melody(ctx)
        missing = sorted({c.id for c in clips} - {m.clip_id for m in melody.clips})
        if missing:
            raise StageError(f"04_melody.json has no melody for {', '.join(missing)}; "
                             "re-run `melody`")  # fmt: skip

        warnings: list[str] = []
        parts, source, notes = default_parts(preset, clips), "preset", clip_set.notes
        ending = None
        if "arc" in plan.inputs:
            result = ArcResult.model_validate_json(plan.inputs["arc"].read_text())
            proposal = result.answer()
            problems = check_parts(proposal.parts, clips, preset) if proposal else ["no answer"]
            if problems:
                warnings.append("Claude's arc was not usable, so the preset's arc is used: "
                                + "; ".join(problems))  # fmt: skip
            elif proposal is not None:
                parts, source, notes = proposal.parts, "claude", proposal.notes or notes
                ending = proposal.ending
        arrangement = build_arrangement(parts, clips, melody, preset, arc_source=source,
                                        notes=notes, warnings=warnings,
                                        ending=ending)  # fmt: skip
        problems = validate_arrangement(arrangement, {c.id: c for c in clips})
        if problems:  # a bug, not a user error: the builder made an invalid arrangement
            raise StageError("invalid arrangement: " + "; ".join(problems))
        for warning in warnings:
            ctx.say(f"[yellow]  {escape(warning)}[/]")
        write_json(ctx.run.path(ARRANGEMENT), arrangement)
        return StageResult(
            summary={
                "sections": len(arrangement.sections),
                "bars": arrangement.total_bars,
                "seconds": arrangement.total_seconds,
                "arc": source,
                "ending": arrangement.ending,
                "alone": [x.clip_id for x in arrangement.sections if x.rest],
            }
        )


def show_arrangement(ctx: Context) -> None:
    """Print the arrangement's timeline (whether it was just built or cached)."""
    path = ctx.run.path(ARRANGEMENT)
    if not path.exists():
        return
    arrangement = Arrangement.model_validate_json(path.read_text(encoding="utf-8"))
    texts: dict[str, str] = {}
    if ctx.run.path(CLIPS).exists():
        texts = {c.id: c.text for c in _clips(ctx)[0].clips}
    minutes, seconds = divmod(round(arrangement.total_seconds), 60)
    table = Table(title=f"Arrangement · {arrangement.key}, {arrangement.bpm:g} BPM, "
                        f"{arrangement.total_bars} bars, {minutes}:{seconds:02d} "
                        f"(arc from the {arrangement.arc_source})")  # fmt: skip
    for column in ("", "Bar", "Bars", "Time", "Role", "Energy", "Clip / melody layer"):
        table.add_column(column)
    preset = load_preset(ctx.run.manifest.preset, ctx.config.presets_dir)
    layer = (ctx.run.manifest.options.melody_layer or preset.melody.layer) != "off"
    for row in timeline_rows(arrangement, texts, melody_layer=layer):
        table.add_row(*(escape(cell) for cell in row))
    ctx.console.print(table)
    for line in timeline_strip(arrangement):
        ctx.console.print(escape(line))
    if arrangement.ending:
        ctx.console.print(f"Ending: {arrangement.ending.replace('_', ' ')}")
    if arrangement.notes:
        ctx.console.print(f"Notes: {escape(arrangement.notes)}")
    kept = [s.clip_id for s in arrangement.sections if s.clip_id]
    if ctx.run.path(CLIPS).exists():
        kept = [c.id for c in _clips(ctx)[1]]
    for warning in arrangement_warnings(arrangement, kept):
        ctx.console.print(f"[yellow]{escape(warning)}[/]")
    ctx.console.print(f"Edit {escape(str(path))} by hand to change it; later steps pick up "
                      "the edit.")  # fmt: skip
