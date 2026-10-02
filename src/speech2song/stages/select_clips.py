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
from speech2song.audio.synth import write_wav
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
    ClipTargets,
    RequiredQuote,
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
from speech2song.text.parts import plan_parts
from speech2song.text.quotes import match_quotes, parse_quotes

SELECTION = "03_selection.json"
CLIPS = "03_clips.json"
REVIEW = "03_review.json"
CLIPS_DIR = "clips"
FRAME_MS = 10.0  # analysis window for finding quiet cut points
FADE_MARGIN_MS = 20.0  # aim cuts this far beyond the fade, so fades fall in silence
ASSUMED_TALK_S = 900.0  # used by dry runs when nothing about the source is known yet


def claude_model(ctx: Context) -> str:
    return ctx.run.manifest.options.claude_model or ctx.config.claude_model


def required_quotes(ctx: Context) -> list[RequiredQuote]:
    """The run's --quotes, matched to the transcript (empty without a file or before the
    transcript exists). Raises when a quote can't be found."""
    ref = ctx.run.manifest.quotes
    path = ctx.run.path(TRANSCRIPT)
    if ref is None or not path.exists():
        return []
    transcript = Transcript.model_validate_json(path.read_text(encoding="utf-8"))
    return match_quotes(transcript, parse_quotes(Path(ref.path)))


def _targets(ctx: Context) -> ClipTargets:
    preset = load_preset(ctx.run.manifest.preset, ctx.config.presets_dir)
    return clip_targets(preset, ctx.run.manifest.options, required_quotes(ctx))


def select_estimates(
    config: AppConfig, model: str, input_tokens: int, count: int, basis: str
) -> list[SpendEstimate]:
    """Spend for one selection call plus the retry that happens only if validation fails."""
    output_tokens = estimate_output_tokens(count)
    usd = claude_usd(config.pricing, model, input_tokens, output_tokens)
    units = {"input_tokens": input_tokens, "output_tokens": output_tokens}
    return [
        SpendEstimate(service="anthropic", model=model, description=f"select up to {count} clips "
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
        targets = _targets(ctx)
        params = {
            "model": claude_model(ctx),
            "effort": ctx.config.claude_effort,
            "fallbacks": ctx.config.claude_fallbacks,
            "targets": targets.model_dump(),
            # The prompt also shows the arc, but only for orientation: the arrangement fits
            # any clips to it, so arc edits don't invalidate a paid selection.
            "preset": {"description": preset.description},
            "prompt": prompt_digest(),
            "part_seconds": preset.clips.part_seconds,  # the prompt mentions it
        }
        inputs = {"transcript": ctx.run.path(TRANSCRIPT)}
        if ctx.run.manifest.quotes is not None:
            inputs["quotes"] = Path(ctx.run.manifest.quotes.path)
        return StagePlan(inputs, params, [SELECTION])

    def estimate(self, ctx: Context, plan: StagePlan) -> list[SpendEstimate]:
        preset = load_preset(ctx.run.manifest.preset, ctx.config.presets_dir)
        targets = _targets(ctx)
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
        return select_estimates(ctx.config, claude_model(ctx), tokens, targets.count_range[1],
                                basis)  # fmt: skip

    def run(self, ctx: Context, plan: StagePlan) -> StageResult:
        transcript = Transcript.model_validate_json(
            plan.inputs["transcript"].read_text(encoding="utf-8")
        )
        if not transcript.sentences:
            raise StageError("The transcript has no sentences to choose from.")
        preset = load_preset(ctx.run.manifest.preset, ctx.config.presets_dir)
        targets = _targets(ctx)
        for quote in targets.required:
            ctx.say(f"  required {quote.id}: sentences {quote.start_sentence}-"
                    f"{quote.end_sentence} ({quote.similarity:.0%} match)")  # fmt: skip
            if quote.occurrences > 1:
                ctx.say(f"[yellow]    it appears {quote.occurrences} times in the talk; using "
                        "the first. Quote a few more words to choose another.[/]")  # fmt: skip
        load_secrets()  # puts the key from .env into the environment for the SDK
        claude = ClaudeJson(ctx.config, CostLog(ctx.run.costs_path), ctx.run.id, claude_model(ctx))
        system, user = build_prompt(transcript, preset, targets)

        attempts: list[SelectionAttempt] = []

        def ask(prompt: str) -> ClipSelection:
            low, high = targets.count_range
            ctx.say(f"  asking {escape(claude.model)} for {low if low == high else f'{low}-{high}'}"
                    f" clips (attempt {len(attempts) + 1})")  # fmt: skip
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
        chosen, usable = 0, _usable(answer, transcript, targets)
        if needs_retry(problems):
            ctx.say(f"[yellow]  the answer had {len(problems)} problem(s); asking once more "
                    "with feedback[/]")  # fmt: skip
            for problem in problems:
                ctx.say(f"    - {escape(problem.message)}")
            second = ask(feedback_prompt(user, answer, problems))
            if _usable(second, transcript, targets) >= usable:
                chosen = 1
        if not _usable(attempts[chosen].answer, transcript, targets):
            raise StageError("None of Claude's clips could be used; see 03_selection.json.")

        result = SelectionResult(model=claude.model, targets=targets, attempts=attempts,
                                 chosen=chosen)  # fmt: skip
        write_json(ctx.run.path(SELECTION), result)
        selection, _ = finalize(result.answer(), transcript, targets)
        ctx.console.print(_selection_table(selection, transcript))
        if selection.notes:
            ctx.say(f"  notes: {escape(selection.notes)}")
        spent = sum(a.usd or 0.0 for a in attempts)
        return StageResult(
            summary={"clips": len(selection.clips), "attempts": len(attempts), "usd": spent}
        )


def _usable(answer: ClipSelection | None, transcript: Transcript, targets: ClipTargets) -> int:
    """How many clips survive the clean-up (0 if none)."""
    if answer is None:
        return 0
    try:
        return len(finalize(answer, transcript, targets)[0].clips)
    except StageError:
        return 0


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
    """Cuts every selected quote sample-exactly. A quote with more than
    `clips.part_seconds` of speech is cut into parts at its natural pauses; each part is
    its own clip (`c3-1`, `c3-2`, ...) and the arrangement plays them with a little music
    in between."""

    name: ClassVar[str] = "clips"
    version: ClassVar[int] = 2  # 2: long quotes in parts

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
            "part_seconds": preset.clips.part_seconds,
        }
        return StagePlan(inputs, params, [CLIPS])

    def run(self, ctx: Context, plan: StagePlan) -> StageResult:
        result = SelectionResult.model_validate_json(plan.inputs["selection"].read_text())
        transcript = Transcript.model_validate_json(plan.inputs["transcript"].read_text())
        selection, warnings = finalize(result.answer(), transcript, result.targets)
        for warning in warnings:
            ctx.say(f"[yellow]  {escape(warning)}[/]")
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
        parts_of: dict[str, list[str]] = {}
        with sf.SoundFile(str(plan.inputs["audio"])) as audio:
            rate, channels = audio.samplerate, audio.channels
            for selected in selection.clips:
                pieces = _cut_quote(audio, selected, transcript, len(clips) + 1, search,
                                    fade_ms, plan.params["part_seconds"])  # fmt: skip
                for clip in pieces:
                    block = _read(audio, clip.start_sample, clip.end_sample)
                    fade = round(fade_ms / 1000 * rate)
                    write_wav(ctx.run.path(clip.file), fade_edges(block, fade), rate)
                clips += pieces
                parts_of[selected.id] = [clip.id for clip in pieces]

        order = [part for quote in selection.suggested_order for part in parts_of[quote]]
        dropped: list[str] = []
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
            notes=selection.notes,
            warnings=warnings,
        )
        write_json(ctx.run.path(CLIPS), clip_set)
        for clip in clips:
            state = "dropped" if clip.id in dropped else f"#{order.index(clip.id) + 1}"
            part = f" (part {clip.part} of {clip.parts})" if clip.parts > 1 else ""
            ctx.say(
                f"  {clip.id:6} {state:>7}  {clip.start_s:7.2f}-{clip.end_s:7.2f} s "
                f"({clip.duration_s:4.1f} s)  cut levels {clip.cut_level_db[0]:.0f}/"
                f"{clip.cut_level_db[1]:.0f} dBFS  -> {clip.file}{part}"
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


def _cut_words(
    audio: sf.SoundFile,
    words: list[Word],
    ws: int,
    we: int,
    search: Search,
    fade_ms: float,
    floor_sample: int = 0,
) -> tuple[int, float, int, float]:
    """Cut points around words[ws:we]: within the search limits of the word edges, never
    past the middle of a neighbouring word (nor before `floor_sample`), in the quiet
    stretch nearest the word boundary, a fade length (plus a margin) away from the speech
    so the fade falls in the pause. Returns (start, start dB, end, end dB)."""
    duration = audio.frames / audio.samplerate
    nominal_start, nominal_end = words[ws].start, max(w.end for w in words[ws:we])
    pad = (fade_ms + FADE_MARGIN_MS) / 1000
    floor = floor_sample / audio.samplerate
    lo = max(nominal_start - search.lead_s, _mid(words[ws - 1]) if ws > 0 else 0.0, floor)
    hi = min(nominal_start + search.inner_s, _mid(words[ws]))
    start_sample, start_db = _find_cut(audio, nominal_start, min(lo, hi), hi, "start", pad)
    lo = max(nominal_end - search.inner_s, _mid(words[we - 1]))
    hi = min(nominal_end + search.tail_s, _mid(words[we]) if we < len(words) else duration)
    end_sample, end_db = _find_cut(audio, nominal_end, lo, max(lo, hi), "end", pad)
    return start_sample, start_db, end_sample, end_db


def _sentence_of(transcript: Transcript, word: int) -> int:
    return next(s.id for s in transcript.sentences if s.word_start <= word < s.word_end)


def _cut_quote(
    audio: sf.SoundFile,
    selected: SelectedClip,
    transcript: Transcript,
    number: int,
    search: Search,
    fade_ms: float,
    part_seconds: float,
) -> list[Clip]:
    """The clip for a selected quote, or its parts when it has more speech than
    `part_seconds`. Parts split at the quote's best pauses; each part's start cut stays
    after the previous part's end cut. Files are numbered from `number`."""
    by_id = {s.id: s for s in transcript.sentences}
    first, last = by_id[selected.start_sentence], by_id[selected.end_sentence]
    words = transcript.words
    ws, we = first.word_start, last.word_end
    ends = {s.word_end - 1 - ws for s in transcript.sentences
            if ws < s.word_end <= we}  # fmt: skip
    pieces = plan_parts(words[ws:we], ends, part_seconds)
    rate = audio.samplerate
    clips: list[Clip] = []
    floor = 0
    for index, (a, b) in enumerate(pieces, start=1):
        start, start_db, end, end_db = _cut_words(audio, words, ws + a, ws + b, search,
                                                  fade_ms, floor)  # fmt: skip
        if end <= start:
            raise StageError(f"{selected.id}: could not place cut points")
        floor = end
        whole = len(pieces) == 1
        clips.append(Clip(
            id=selected.id if whole else f"{selected.id}-{index}",
            file=f"{CLIPS_DIR}/clip_{number + index - 1:03d}.wav",
            start_sentence=_sentence_of(transcript, ws + a),
            end_sentence=_sentence_of(transcript, ws + b - 1),
            text=selected.text if whole else " ".join(w.w for w in words[ws + a : ws + b]),
            score=selected.score, role=selected.role, reason=selected.reason,
            nominal_start_s=round(words[ws + a].start, 3),
            nominal_end_s=round(max(w.end for w in words[ws + a : ws + b]), 3),
            start_s=round(start / rate, 4), end_s=round(end / rate, 4),
            start_sample=start, end_sample=end, duration_s=round((end - start) / rate, 4),
            cut_level_db=(round(start_db, 1), round(end_db, 1)),
            quote=selected.id, part=index, parts=len(pieces),
        ))  # fmt: skip
    return clips


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
    write_wav(path, fade_edges(block, round(0.01 * rate)), rate)
    player = shutil.which("ffplay")
    if player and sys.stdin.isatty():
        subprocess.run(
            [player, "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)], check=False
        )
        return f"played {clip_id} (with context): {path}"
    return f"preview of {clip_id} with context: {path}"
