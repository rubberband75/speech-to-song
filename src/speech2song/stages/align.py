"""Stage 2b: build 02_transcript.json from 02_asr.json, aligned to the official transcript."""

from pathlib import Path
from typing import ClassVar

from rich.markup import escape

from speech2song.errors import S2SError
from speech2song.manifest import write_json
from speech2song.models import AsrResult, Transcript
from speech2song.pipeline import Context, Stage, StagePlan, StageResult
from speech2song.stages.transcribe import ASR
from speech2song.text.align import build_transcript

TRANSCRIPT = "02_transcript.json"
_SHOW_SPANS = 15


def read_official(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise S2SError(f"The transcript must be UTF-8 plain text: {path}") from exc


def _quote(text: str, limit: int = 70) -> str:
    return escape(text if len(text) <= limit else text[: limit - 1] + "…")


class AlignStage(Stage):
    name: ClassVar[str] = "align"
    version: ClassVar[int] = 1

    def plan(self, ctx: Context) -> StagePlan:
        inputs = {"asr": ctx.run.path(ASR)}
        if ctx.run.manifest.transcript is not None:
            inputs["official"] = Path(ctx.run.manifest.transcript.path)
        return StagePlan(inputs, ctx.config.align.model_dump(), [TRANSCRIPT])

    def run(self, ctx: Context, plan: StagePlan) -> StageResult:
        asr = AsrResult.model_validate_json(plan.inputs["asr"].read_text(encoding="utf-8"))
        official = read_official(plan.inputs["official"]) if "official" in plan.inputs else None
        transcript = build_transcript(
            asr, official, ctx.config.align, source=ctx.run.manifest.input.path
        )
        write_json(ctx.run.path(TRANSCRIPT), transcript)
        self._report(ctx, transcript)
        summary: dict[str, object] = {
            "official_transcript_used": transcript.official_transcript_used,
            "words": len(transcript.words),
            "sentences": len(transcript.sentences),
        }
        if transcript.alignment is not None:
            keys = {"quality", "quality_spoken", "coverage", "unspoken", "asr_only", "fallback"}
            summary |= transcript.alignment.model_dump(include=keys)
        return StageResult(summary=summary)

    @staticmethod
    def _report(ctx: Context, transcript: Transcript) -> None:
        report = transcript.alignment
        by_source = {"official": 0, "asr": 0}
        for sentence in transcript.sentences:
            by_source[sentence.source] += 1
        if report is None:
            ctx.say(
                f"  no official transcript: {len(transcript.words)} ASR words, "
                f"{len(transcript.sentences)} sentences"
            )
            return
        ctx.say(
            f"  official transcript: {report.official_words} words · matched {report.matched}"
            f" · fuzzy {report.fuzzy} · interpolated {report.interpolated}"
            f" · unspoken {report.unspoken} · ASR-only {report.asr_only}"
        )
        ctx.say(
            f"  quality {report.quality:.1%} (spoken text {report.quality_spoken:.1%})"
            f" · ASR words tied to the official text {report.coverage:.1%}"
        )
        if report.fallback:
            ctx.say(
                "[yellow]  The official transcript barely matches the audio "
                f"({report.anchor_coverage:.0%} solid matches), so it was ignored. "
                "Is it the right file?[/]"
            )
        for span in report.unspoken_spans[:_SHOW_SPANS]:
            ctx.say(f'  unspoken (line {span.line}): "{_quote(span.text)}"')
        for span in report.asr_only_spans[:_SHOW_SPANS]:
            ctx.say(f'  ASR-only {span.start:.1f}-{span.end:.1f} s: "{_quote(span.text)}"')
        hidden = max(0, len(report.unspoken_spans) - _SHOW_SPANS) + max(
            0, len(report.asr_only_spans) - _SHOW_SPANS
        )
        if hidden:
            ctx.say(f"  … {hidden} more spans in {TRANSCRIPT} (alignment)")
        ctx.say(
            f"  {len(transcript.sentences)} sentences "
            f"({by_source['official']} official, {by_source['asr']} ASR-only)"
        )
