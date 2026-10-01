"""Stage 3: Claude selects clips (paid), then the clips are cut sample-exactly (free).

`select` writes 03_selection.json; `clips` writes 03_clips.json and clips/clip_NNN.wav.
They are separate cached stages so that changing fades or boundary settings, or editing
the order in review, never repeats the Claude call.
"""

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
import soundfile as sf
from rich.markup import escape
from rich.table import Table

from speech2song.audio.dsp import fade_edges, quietest_point
from speech2song.config import AppConfig, load_preset, load_secrets
from speech2song.costs import CostLog, SpendEstimate, claude_usd
from speech2song.errors import S2SError, StageError
from speech2song.llm.claude import ClaudeJson
from speech2song.manifest import Run, write_json
from speech2song.models import (
    Clip,
    ClipReview,
    ClipSelection,
    ClipSet,
    SelectedClip,
    SelectionAttempt,
    SelectionResult,
    Transcript,
    Word,
)
from speech2song.pipeline import Context, Stage, StagePlan, StageResult
from speech2song.selection import (
    SELECTION_SCHEMA,
    build_prompt,
    clip_targets,
    estimate_input_tokens,
    estimate_input_tokens_from_duration,
    estimate_output_tokens,
    feedback_prompt,
    finalize,
    needs_retry,
    prompt_digest,
    validate,
)
from speech2song.stages.align import TRANSCRIPT
from speech2song.stages.ingest import CLEAN

SELECTION = "03_selection.json"
CLIPS = "03_clips.json"
REVIEW = "03_review.json"
CLIPS_DIR = "clips"
FRAME_MS = 10.0  # analysis window for finding quiet cut points
FADE_MARGIN_MS = 20.0  # aim cuts this far beyond the fade, so fades fall in silence
ASSUMED_TALK_S = 900.0  # used by dry runs when nothing about the source is known yet


def claude_model(ctx: Context) -> str:
    return ctx.run.manifest.options.claude_model or ctx.config.claude_model


def select_estimates(
    config: AppConfig, model: str, input_tokens: int, count: int, basis: str
) -> list[SpendEstimate]:
    """Spend for one selection call plus the retry that happens only if validation fails."""
    output_tokens = estimate_output_tokens(count)
    usd = claude_usd(config.pricing, model, input_tokens, output_tokens)
    units = {"input_tokens": input_tokens, "output_tokens": output_tokens}
    return [
        SpendEstimate(service="anthropic", model=model, description=f"select {count} clips "
                      f"({basis})", units=units, usd=usd),
        SpendEstimate(service="anthropic", model=model,
                      description="retry, only if the answer fails validation", units=units,
                      usd=usd),
    ]  # fmt: skip


class SelectStage(Stage):
    name: ClassVar[str] = "select"
    version: ClassVar[int] = 1
    paid: ClassVar[bool] = True

    def plan(self, ctx: Context) -> StagePlan:
        preset = load_preset(ctx.run.manifest.preset, ctx.config.presets_dir)
        targets = clip_targets(preset, ctx.run.manifest.options)
        params = {
            "model": claude_model(ctx),
            "effort": ctx.config.claude_effort,
            "fallbacks": ctx.config.claude_fallbacks,
            "targets": targets.model_dump(),
            "preset": {"description": preset.description, "arc": preset.arc},
            "prompt": prompt_digest(),
        }
        return StagePlan({"transcript": ctx.run.path(TRANSCRIPT)}, params, [SELECTION])

    def estimate(self, ctx: Context, plan: StagePlan) -> list[SpendEstimate]:
        preset = load_preset(ctx.run.manifest.preset, ctx.config.presets_dir)
        targets = clip_targets(preset, ctx.run.manifest.options)
        path = plan.inputs["transcript"]
        if path.exists():
            transcript = Transcript.model_validate_json(path.read_text(encoding="utf-8"))
            tokens = estimate_input_tokens(*build_prompt(transcript, preset, targets))
            basis = f"{len(transcript.sentences)} sentences"
        else:
            source = ctx.run.manifest.source or ctx.run.manifest.source_probe
            seconds = getattr(source, "duration_s", None) or ASSUMED_TALK_S
            tokens = estimate_input_tokens_from_duration(seconds)
            basis = f"estimated from {seconds / 60:.0f} min of audio"
        return select_estimates(ctx.config, claude_model(ctx), tokens, targets.count, basis)

    def run(self, ctx: Context, plan: StagePlan) -> StageResult:
        transcript = Transcript.model_validate_json(
            plan.inputs["transcript"].read_text(encoding="utf-8")
        )
        if not transcript.sentences:
            raise StageError("The transcript has no sentences to choose from.")
        preset = load_preset(ctx.run.manifest.preset, ctx.config.presets_dir)
        targets = clip_targets(preset, ctx.run.manifest.options)
        load_secrets()  # puts the key from .env into the environment for the SDK
        claude = ClaudeJson(ctx.config, CostLog(ctx.run.costs_path), ctx.run.id, claude_model(ctx))
        system, user = build_prompt(transcript, preset, targets)

        attempts: list[SelectionAttempt] = []

        def ask(prompt: str) -> ClipSelection:
            ctx.say(f"  asking {escape(claude.model)} for {targets.count} clips "
                    f"(attempt {len(attempts) + 1})")  # fmt: skip
            reply = claude.call(
                system=system, user=prompt, schema=SELECTION_SCHEMA, stage=self.name,
                operation="messages.create",
            )  # fmt: skip
            try:
                answer = ClipSelection.model_validate(reply.data)
            except ValueError as exc:
                raise S2SError(f"Claude's answer did not match the clip schema: {exc}") from exc
            problems = validate(answer, transcript, targets)
            attempts.append(
                SelectionAttempt(
                    attempt=len(attempts) + 1,
                    model=reply.model,
                    request_id=reply.request_id,
                    input_tokens=reply.input_tokens,
                    output_tokens=reply.output_tokens,
                    usd=reply.usd,
                    answer=answer,
                    problems=[p.message for p in problems],
                )
            )
            if reply.fallback_from:
                ctx.say(f"[yellow]  {reply.fallback_from} declined; {reply.model} answered[/]")
            return answer

        answer = ask(user)
        problems = validate(answer, transcript, targets)
        chosen = finalize(answer, transcript, targets)
        if needs_retry(problems):
            ctx.say(f"[yellow]  the answer had {len(problems)} problem(s); asking once more "
                    "with feedback[/]")  # fmt: skip
            for problem in problems:
                ctx.say(f"    - {escape(problem.message)}")
            second = finalize(ask(feedback_prompt(user, answer, problems)), transcript, targets)
            if len(second[0].clips) >= len(chosen[0].clips):
                chosen = second
        selection, warnings = chosen
        for warning in warnings:
            ctx.say(f"[yellow]  {escape(warning)}[/]")

        result = SelectionResult(
            model=claude.model, targets=targets, attempts=attempts, selection=selection,
            warnings=warnings,
        )  # fmt: skip
        write_json(ctx.run.path(SELECTION), result)
        ctx.console.print(_selection_table(selection, transcript))
        if selection.notes:
            ctx.say(f"  notes: {escape(selection.notes)}")
        spent = sum(a.usd or 0.0 for a in attempts)
        return StageResult(
            summary={"clips": len(selection.clips), "attempts": len(attempts), "usd": spent}
        )


def _selection_table(selection: ClipSelection, transcript: Transcript) -> Table:
    by_id = {s.id: s for s in transcript.sentences}
    table = Table(title="Selected clips")
    for column in ("#", "Clip", "Sentences", "Time", "Role", "Score", "Text"):
        table.add_column(column)
    position = {clip_id: n for n, clip_id in enumerate(selection.suggested_order, start=1)}
    for clip in selection.clips:
        start, end = by_id[clip.start_sentence].start, by_id[clip.end_sentence].end
        text = clip.text if len(clip.text) <= 70 else clip.text[:69] + "..."
        table.add_row(
            str(position.get(clip.id, "")), clip.id,
            f"{clip.start_sentence}-{clip.end_sentence}", f"{start:.1f}-{end:.1f} s",
            clip.role, f"{clip.score:.2f}", escape(text),
        )  # fmt: skip
    return table


class ClipsStage(Stage):
    name: ClassVar[str] = "clips"
    version: ClassVar[int] = 1

    def plan(self, ctx: Context) -> StagePlan:
        preset = load_preset(ctx.run.manifest.preset, ctx.config.presets_dir)
        inputs = {
            "selection": ctx.run.path(SELECTION),
            "transcript": ctx.run.path(TRANSCRIPT),
            "audio": ctx.run.path(CLEAN),
        }
        if ctx.run.path(REVIEW).exists():
            inputs["review"] = ctx.run.path(REVIEW)
        params = {
            "fade_ms": preset.clips.fade_ms,
            "lead_search_ms": preset.clips.lead_search_ms,
            "tail_search_ms": preset.clips.tail_search_ms,
            "inner_search_ms": preset.clips.inner_search_ms,
            "frame_ms": FRAME_MS,
        }
        return StagePlan(inputs, params, [CLIPS])

    def run(self, ctx: Context, plan: StagePlan) -> StageResult:
        result = SelectionResult.model_validate_json(plan.inputs["selection"].read_text())
        transcript = Transcript.model_validate_json(plan.inputs["transcript"].read_text())
        fade_ms = plan.params["fade_ms"]
        search = Search(
            lead_s=plan.params["lead_search_ms"] / 1000,
            tail_s=plan.params["tail_search_ms"] / 1000,
            inner_s=plan.params["inner_search_ms"] / 1000,
        )
        clips_dir = ctx.run.path(CLIPS_DIR)
        clips_dir.mkdir(exist_ok=True)
        for stale in clips_dir.glob("clip_*.wav"):
            stale.unlink()

        clips: list[Clip] = []
        with sf.SoundFile(str(plan.inputs["audio"])) as audio:
            rate, channels = audio.samplerate, audio.channels
            for number, selected in enumerate(result.selection.clips, start=1):
                clip = _cut(audio, selected, transcript, number, search, fade_ms)
                block = _read(audio, clip.start_sample, clip.end_sample)
                fade = round(fade_ms / 1000 * rate)
                sf.write(str(ctx.run.path(clip.file)), fade_edges(block, fade), rate,
                         subtype="FLOAT")  # fmt: skip
                clips.append(clip)

        order, dropped = list(result.selection.suggested_order), []
        review = self._review(ctx, plan, {c.id for c in clips})
        if review is not None:
            order, dropped = review.order, review.dropped
        clip_set = ClipSet(
            source=CLEAN,
            isolated=ctx.run.manifest.options.isolate_voice,
            sample_rate=rate,
            channels=channels,
            fade_ms=fade_ms,
            clips=clips,
            order=order,
            dropped=dropped,
            notes=result.selection.notes,
        )
        write_json(ctx.run.path(CLIPS), clip_set)
        for clip in clips:
            state = "dropped" if clip.id in dropped else f"#{order.index(clip.id) + 1}"
            ctx.say(
                f"  {clip.id} {state:>7}  {clip.start_s:7.2f}-{clip.end_s:7.2f} s "
                f"({clip.duration_s:4.1f} s)  cut levels {clip.cut_level_db[0]:.0f}/"
                f"{clip.cut_level_db[1]:.0f} dBFS  -> {clip.file}"
            )
        return StageResult(
            outputs=[CLIPS, *(c.file for c in clips)],
            summary={"clips": len(clips), "kept": len(order), "dropped": len(dropped)},
        )

    @staticmethod
    def _review(ctx: Context, plan: StagePlan, clip_ids: set[str]) -> ClipReview | None:
        path = plan.inputs.get("review")
        if path is None:
            return None
        review = ClipReview.model_validate_json(path.read_text())
        current, _ = ctx.run.hash_file(plan.inputs["selection"])
        if review.selection_sha256 != current:
            ctx.say("[yellow]  03_review.json was made for an earlier selection; ignoring it[/]")
            return None
        if not set(review.order) | set(review.dropped) <= clip_ids:
            ctx.say("[yellow]  03_review.json names unknown clips; ignoring it[/]")
            return None
        return review


def _read(audio: sf.SoundFile, start: int, end: int) -> np.ndarray:
    audio.seek(start)
    return audio.read(end - start, dtype="float32", always_2d=True)


def _mid(word: Word) -> float:
    return (word.start + word.end) / 2


def _find_cut(
    audio: sf.SoundFile,
    boundary_s: float,
    lo_s: float,
    hi_s: float,
    side: Literal["start", "end"],
    pad_s: float,
) -> tuple[int, float]:
    """Sample index of the cut between lo_s and hi_s, and the level there."""
    rate = audio.samplerate
    frame = round(FRAME_MS / 1000 * rate)
    lo = max(0, round(lo_s * rate))
    hi = min(audio.frames, max(lo, round(hi_s * rate)))
    start = max(0, lo - frame)
    block = _read(audio, start, min(audio.frames, hi + frame)).mean(axis=1)
    index, level = quietest_point(
        block, lo - start, hi - start, round(boundary_s * rate) - start, frame=frame,
        side=side, pad=round(pad_s * rate),
    )  # fmt: skip
    return start + index, level


@dataclass(frozen=True)
class Search:
    lead_s: float  # before the first word
    tail_s: float  # after the last word
    inner_s: float  # into the clip


def _cut(
    audio: sf.SoundFile,
    selected: SelectedClip,
    transcript: Transcript,
    number: int,
    search: Search,
    fade_ms: float,
) -> Clip:
    """Cut points: within the search limits of the sentence edges, never past the middle of a
    neighbouring word, in the quiet stretch nearest the transcript's boundary, a fade
    length (plus a margin) away from the speech so the fade falls in the pause."""
    by_id = {s.id: s for s in transcript.sentences}
    first, last = by_id[selected.start_sentence], by_id[selected.end_sentence]
    words = transcript.words
    ws, we = first.word_start, last.word_end
    duration = audio.frames / audio.samplerate
    nominal_start, nominal_end = words[ws].start, max(w.end for w in words[ws:we])

    pad = (fade_ms + FADE_MARGIN_MS) / 1000
    lo = max(nominal_start - search.lead_s, _mid(words[ws - 1]) if ws > 0 else 0.0)
    hi = min(nominal_start + search.inner_s, _mid(words[ws]))
    start_sample, start_db = _find_cut(audio, nominal_start, min(lo, hi), hi, "start", pad)
    lo = max(nominal_end - search.inner_s, _mid(words[we - 1]))
    hi = min(nominal_end + search.tail_s, _mid(words[we]) if we < len(words) else duration)
    end_sample, end_db = _find_cut(audio, nominal_end, lo, max(lo, hi), "end", pad)
    if end_sample <= start_sample:
        raise StageError(f"{selected.id}: could not place cut points")

    rate = audio.samplerate
    return Clip(
        id=selected.id,
        file=f"{CLIPS_DIR}/clip_{number:03d}.wav",
        start_sentence=selected.start_sentence,
        end_sentence=selected.end_sentence,
        text=selected.text,
        score=selected.score,
        role=selected.role,
        reason=selected.reason,
        nominal_start_s=round(nominal_start, 3),
        nominal_end_s=round(nominal_end, 3),
        start_s=round(start_sample / rate, 4),
        end_s=round(end_sample / rate, 4),
        start_sample=start_sample,
        end_sample=end_sample,
        duration_s=round((end_sample - start_sample) / rate, 4),
        cut_level_db=(round(start_db, 1), round(end_db, 1)),
    )


def write_review(ctx: Context, order: list[str], dropped: list[str]) -> Path:
    sha, _ = ctx.run.hash_file(ctx.run.path(SELECTION))
    path = ctx.run.path(REVIEW)
    write_json(path, ClipReview(selection_sha256=sha, order=order, dropped=dropped))
    return path


PREVIEW_DIR = "03_preview"
PREVIEW_CONTEXT_S = 1.0


def write_preview(run: Run, clip_set: ClipSet, clip_id: str) -> str:
    """Write the clip with a second of surrounding audio, and play it when possible."""
    clip = next(c for c in clip_set.clips if c.id == clip_id)
    rate = clip_set.sample_rate
    pad = round(PREVIEW_CONTEXT_S * rate)
    with sf.SoundFile(str(run.path(clip_set.source))) as audio:
        start, end = max(0, clip.start_sample - pad), min(audio.frames, clip.end_sample + pad)
        block = _read(audio, start, end)
    path = run.path(f"{PREVIEW_DIR}/{clip_id}.wav")
    path.parent.mkdir(exist_ok=True)
    sf.write(str(path), fade_edges(block, round(0.01 * rate)), rate, subtype="FLOAT")
    player = shutil.which("ffplay")
    if player and sys.stdin.isatty():
        subprocess.run(
            [player, "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)], check=False
        )
        return f"played {clip_id} (with context): {path}"
    return f"preview of {clip_id} with context: {path}"
