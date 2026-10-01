"""Stage 2a: speech recognition (02_asr.json)."""

from typing import ClassVar

from rich.progress import BarColumn, Progress, TimeElapsedColumn, TimeRemainingColumn

from speech2song.backends import transcribe_base
from speech2song.manifest import write_json
from speech2song.pipeline import Context, Stage, StagePlan, StageResult
from speech2song.stages.ingest import CLEAN

ASR = "02_asr.json"


class AsrStage(Stage):
    name: ClassVar[str] = "asr"
    version: ClassVar[int] = 2

    def plan(self, ctx: Context) -> StagePlan:
        transcriber = transcribe_base.make_transcriber(ctx.config, ctx.run.manifest.options)
        return StagePlan({"audio": ctx.run.path(CLEAN)}, transcriber.params(), [ASR])

    def run(self, ctx: Context, plan: StagePlan) -> StageResult:
        transcriber = transcribe_base.make_transcriber(ctx.config, ctx.run.manifest.options)
        ctx.say(f"  transcribing {CLEAN} with {plan.params.get('model', transcriber.name)}")
        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=ctx.console,
            transient=True,
        ) as bar:
            task = bar.add_task("transcribing", total=1.0)
            result = transcriber.transcribe(
                plan.inputs["audio"], progress=lambda done: bar.update(task, completed=done)
            )
        write_json(ctx.run.path(ASR), result)
        words = len(result.words())
        ctx.say(
            f"  language {result.language} (p={result.language_probability or 0:.2f}), "
            f"{len(result.segments)} segments, {words} words"
        )
        for repair in result.repairs:
            ctx.say(
                f"[yellow]  repaired a gap at {repair.gap_start:.1f}-{repair.gap_end:.1f} s "
                f"({repair.speech_s:.1f} s of speech had no words): re-transcribed "
                f"{repair.start:.1f}-{repair.end:.1f} s, {repair.words_before} -> "
                f"{repair.words_after} words[/]"
            )
        return StageResult(
            summary={
                "language": result.language,
                "words": words,
                "model": result.model,
                "repairs": len(result.repairs),
            }
        )
