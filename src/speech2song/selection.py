"""Clip selection logic, free of I/O: prompt, JSON schema, and checks on Claude's answer.

Claude picks sentence ranges. Before anything is cut, its answer is checked against the
transcript: the sentences must exist, the quoted text must match them, each clip must
fit the duration bounds (a long quote is allowed up to `long_max_seconds`), clips must
not overlap, the total must fit the speech budget, and every quote the user required
must be there. One retry with feedback is allowed; after that, unusable clips are
dropped and missing required quotes are added.
"""

import hashlib
import re
from dataclasses import dataclass
from string import Template
from typing import Literal

from rapidfuzz.distance import Indel

from speech2song.config import Preset
from speech2song.errors import StageError
from speech2song.llm.claude import load_prompt
from speech2song.models import (
    ClipSelection,
    ClipTargets,
    RequiredQuote,
    RunOptions,
    SelectedClip,
    Transcript,
)
from speech2song.text.normalize import normalize_word

CLIP_ROLES = ("hook", "build", "payoff", "breakdown", "outro")
TEXT_MATCH_MIN = 0.8  # normalized similarity between a clip's quote and its sentences
CLIP_ID = re.compile(r"[A-Za-z0-9_-]{1,16}")

# Rough, offline token estimates (dry runs make no API calls). Calibrated on the sample
# talk: timestamps and numbers tokenize densely (about 2.2 characters per token).
CHARS_PER_TOKEN = 2.2
SCHEMA_OVERHEAD_TOKENS = 600
OUTPUT_TOKENS_BASE = 3000  # adaptive thinking plus JSON scaffolding
OUTPUT_TOKENS_PER_CLIP = 250
WORDS_PER_SECOND = 2.6  # for estimating a transcript that does not exist yet
TOKENS_PER_WORD = 1.6  # includes the per-sentence ID and timing columns

_CLIP_PROPERTIES = {
    "id": {"type": "string"},
    "start_sentence": {"type": "integer"},
    "end_sentence": {"type": "integer"},
    "text": {"type": "string"},
    "score": {"type": "number"},
    "role": {"type": "string", "enum": list(CLIP_ROLES)},
    "reason": {"type": "string"},
}
SELECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "clips": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": _CLIP_PROPERTIES,
                "required": list(_CLIP_PROPERTIES),
                "additionalProperties": False,
            },
        },
        "suggested_order": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "string"},
    },
    "required": ["clips", "suggested_order", "notes"],
    "additionalProperties": False,
}


def clip_targets(
    preset: Preset, options: RunOptions, required: list[RequiredQuote] | None = None
) -> ClipTargets:
    """The preset's targets. `--clips N` asks for exactly N clips; otherwise Claude may
    choose `count_tolerance` fewer or more, so the set can cover the whole talk."""
    spec = preset.clips
    count = options.clips or spec.count
    spread = 0 if options.clips else spec.count_tolerance
    return ClipTargets(
        count=count,
        min_count=max(1, count - spread, len(required or [])),
        max_count=max(count + spread, len(required or [])),
        min_seconds=spec.min_seconds,
        max_seconds=spec.max_seconds,
        long_max_seconds=spec.long_max_seconds,
        total_speech_seconds=spec.total_speech_seconds,
        required=list(required or []),
    )


def _template() -> tuple[str, str]:
    return load_prompt("select_clips")


def prompt_digest() -> str:
    """Hash of the prompt template: editing it invalidates cached selections."""
    system, user = _template()
    return hashlib.sha256(f"{system}\n{user}".encode()).hexdigest()[:16]


def render_sentences(transcript: Transcript) -> str:
    return "\n".join(
        f"{s.id} | {s.start:.1f}-{s.end:.1f} | {s.end - s.start:.1f}s | {s.text}"
        for s in transcript.sentences
    )


def render_required(targets: ClipTargets) -> str:
    if not targets.required:
        return ""
    lines = [
        "The producer requires these passages. Include each one as a clip with exactly the "
        "sentence range given (it counts toward the totals and may be longer than the usual "
        "limits); choose the other clips around them:",
    ]
    for quote in targets.required:
        role = f", role {quote.role}" if quote.role else ""
        lines.append(f"- sentences {quote.start_sentence}-{quote.end_sentence}{role}: "
                     f'"{quote.text}"')  # fmt: skip
    return "\n".join(lines) + "\n\n"


def build_prompt(transcript: Transcript, preset: Preset, targets: ClipTargets) -> tuple[str, str]:
    system, user = _template()
    low, high = targets.count_range
    values = {
        "min_seconds": f"{targets.min_seconds:g}",
        "max_seconds": f"{targets.max_seconds:g}",
        "long_max_seconds": f"{targets.longest:g}",
        "part_seconds": f"{preset.clips.part_seconds:g}",
    }
    rendered = Template(user).substitute(
        preset_description=" ".join(preset.description.split()),
        arc=", ".join(preset.arc),
        count_rule=f"exactly {low}" if low == high else f"{low} to {high} (about {targets.count})",
        total_speech_seconds=f"{targets.total_speech_seconds:g}",
        required=render_required(targets),
        sentences=render_sentences(transcript),
        **values,
    )
    return Template(system).substitute(**values), rendered


def feedback_prompt(user: str, answer: ClipSelection, problems: list["Problem"]) -> str:
    listed = "\n".join(f"- {p.message}" for p in problems)
    return (
        f"{user}\n\nYour previous answer had these problems:\n{listed}\n\n"
        f"Previous answer:\n{answer.model_dump_json()}\n\n"
        "Return a corrected, complete answer that fixes every problem."
    )


ProblemKind = Literal["invalid", "overlap", "budget", "count", "order", "score", "required"]
RETRY_KINDS = {"invalid", "overlap", "budget", "required"}


@dataclass(frozen=True)
class Problem:
    clip: str | None
    kind: ProblemKind
    message: str


def _norm(text: str) -> str:
    return " ".join(t for word in text.split() for t in normalize_word(word))


def text_matches(quote: str, actual: str) -> bool:
    return Indel.normalized_similarity(_norm(quote), _norm(actual)) >= TEXT_MATCH_MIN


def _span(transcript: Transcript, clip: SelectedClip) -> tuple[str, float] | None:
    """(text, duration) of a clip's sentence range, or None if the range is invalid."""
    by_id = {s.id: s for s in transcript.sentences}
    if not (clip.start_sentence in by_id and clip.end_sentence in by_id):
        return None
    if clip.end_sentence < clip.start_sentence:
        return None
    sentences = [by_id[i] for i in range(clip.start_sentence, clip.end_sentence + 1)]
    text = " ".join(s.text for s in sentences)
    return text, max(s.end for s in sentences) - sentences[0].start


def validate(
    selection: ClipSelection, transcript: Transcript, targets: ClipTargets
) -> list[Problem]:
    problems: list[Problem] = []
    last_id = transcript.sentences[-1].id if transcript.sentences else 0
    seen: set[str] = set()
    usable: list[tuple[SelectedClip, float]] = []
    for clip in selection.clips:
        name = clip.id
        if name in seen:
            problems.append(Problem(name, "invalid", f"{name}: the clip id is used twice"))
            continue
        seen.add(name)
        span = _span(transcript, clip)
        rng = f"sentences {clip.start_sentence}-{clip.end_sentence}"
        if span is None:
            problems.append(
                Problem(
                    name, "invalid", f"{name}: {rng} is not a valid range (IDs run 1-{last_id})"
                )
            )
            continue
        text, duration = span
        required = _required_for(clip, targets)
        if not text_matches(clip.text, text):
            problems.append(
                Problem(
                    name, "invalid", f'{name}: the text does not match {rng}, which read: "{text}"'
                )
            )
            continue
        if required is None and not targets.min_seconds <= duration <= targets.longest:
            problems.append(
                Problem(
                    name,
                    "invalid",
                    f"{name}: {rng} last {duration:.1f} s; clips must be "
                    f"{targets.min_seconds:g}-{targets.longest:g} s",
                )
            )
            continue
        if not 0 <= clip.score <= 1:
            problems.append(Problem(name, "score", f"{name}: score must be between 0 and 1"))
        usable.append((clip, duration))

    usable.sort(key=lambda item: item[0].start_sentence)
    reach: SelectedClip | None = None  # the clip reaching furthest so far (catches nesting)
    for clip, _ in usable:
        if reach is not None and clip.start_sentence <= reach.end_sentence:
            problems.append(
                Problem(clip.id, "overlap", f"{reach.id} and {clip.id} share sentences")
            )
        if reach is None or clip.end_sentence > reach.end_sentence:
            reach = clip
    total = sum(duration for _, duration in usable)
    if total > targets.total_speech_seconds:
        problems.append(
            Problem(
                None,
                "budget",
                f"the clips total {total:.1f} s; the budget is {targets.total_speech_seconds:g} s",
            )
        )
    low, high = targets.count_range
    if not low <= len(selection.clips) <= high:
        wanted = f"{low}" if low == high else f"{low}-{high}"
        problems.append(
            Problem(
                None,
                "count",
                f"{len(selection.clips)} clips were returned; {wanted} were asked for",
            )
        )
    chosen = {(c.start_sentence, c.end_sentence) for c, _ in usable}
    for quote in targets.required:
        if (quote.start_sentence, quote.end_sentence) not in chosen:
            problems.append(
                Problem(
                    None,
                    "required",
                    f"the required passage {quote.id} (sentences {quote.start_sentence}-"
                    f"{quote.end_sentence}) must be one of the clips, with exactly that range",
                )
            )
    order = selection.suggested_order
    if sorted(order) != sorted(seen) or len(set(order)) != len(order):
        problems.append(
            Problem(None, "order", "suggested_order must list every clip id exactly once")
        )
    return problems


def _required_for(clip: SelectedClip, targets: ClipTargets) -> RequiredQuote | None:
    return next((q for q in targets.required if (q.start_sentence, q.end_sentence)
                 == (clip.start_sentence, clip.end_sentence)), None)  # fmt: skip


def _required_clip(quote: RequiredQuote, transcript: Transcript, taken: set[str]) -> SelectedClip:
    span = _span(transcript, SelectedClip(id=quote.id, start_sentence=quote.start_sentence,
                                          end_sentence=quote.end_sentence, text="", score=1.0,
                                          role="build", reason=""))  # fmt: skip
    assert span is not None
    name = quote.id if quote.id not in taken else f"{quote.id}r"
    return SelectedClip(id=name, start_sentence=quote.start_sentence,
                        end_sentence=quote.end_sentence, text=span[0], score=1.0,
                        role=quote.role or "build", reason="required by the producer")  # fmt: skip


def _overlaps(a: SelectedClip, b: SelectedClip) -> bool:
    return a.start_sentence <= b.end_sentence and b.start_sentence <= a.end_sentence


def needs_retry(problems: list[Problem]) -> bool:
    return any(p.kind in RETRY_KINDS for p in problems)


def finalize(
    selection: ClipSelection, transcript: Transcript, targets: ClipTargets
) -> tuple[ClipSelection, list[str]]:
    """Keep the usable clips, resolve overlaps and limits by score, sort them by time, and
    use the transcript's own text for each clip.

    Claude's clip IDs are kept, because its notes and reasons refer to them; they are
    replaced by c1..cN (in time order) only if any ID is malformed.
    """
    problems = validate(selection, transcript, targets)
    warnings = [p.message for p in problems if p.kind in ("invalid", "score")]
    invalid = {p.clip for p in problems if p.kind == "invalid"}
    clips: dict[str, tuple[SelectedClip, str, float]] = {}
    for clip in selection.clips:
        if clip.id in clips or clip.id in invalid:
            continue
        span = _span(transcript, clip)
        assert span is not None
        clamped = clip.model_copy(update={"score": min(1.0, max(0.0, clip.score))})
        clips[clip.id] = (clamped, *span)
    must = {(q.start_sentence, q.end_sentence) for q in targets.required}
    present = {(c.start_sentence, c.end_sentence) for c, _, _ in clips.values()}
    added = []
    for quote in targets.required:  # the producer's quotes are never left out
        if (quote.start_sentence, quote.end_sentence) not in present:
            clip = _required_clip(quote, transcript, set(clips))
            clips[clip.id] = (clip, *_span(transcript, clip))  # type: ignore[misc]
            added.append(clip.id)
            warnings.append(f"added the required passage {quote.id} as {clip.id} (sentences "
                            f"{quote.start_sentence}-{quote.end_sentence})")  # fmt: skip

    def priority(item: tuple[SelectedClip, str, float]) -> tuple[bool, float]:
        clip = item[0]
        return ((clip.start_sentence, clip.end_sentence) not in must, -clip.score)

    kept: list[tuple[SelectedClip, str, float]] = []
    for item in sorted(clips.values(), key=priority):
        clip = item[0]
        clash = next((k[0].id for k in kept if _overlaps(clip, k[0])), None)
        if clash:
            warnings.append(f"dropped {clip.id}: it overlaps {clash}, which scored higher or "
                            "is required")  # fmt: skip
            continue
        kept.append(item)
    high = targets.count_range[1]
    while len(kept) > high and (kept[-1][0].start_sentence, kept[-1][0].end_sentence) not in must:
        dropped = kept.pop()  # lowest score (kept is in priority order)
        warnings.append(f"dropped {dropped[0].id}: more clips than the {high} asked for")
    while (sum(item[2] for item in kept) > targets.total_speech_seconds
           and (kept[-1][0].start_sentence, kept[-1][0].end_sentence) not in must):  # fmt: skip
        dropped = kept.pop()
        warnings.append(f"dropped {dropped[0].id}: the clips exceeded the speech budget")
    if not kept:
        raise StageError("None of Claude's clips could be used; see 03_selection.json attempts.")

    kept.sort(key=lambda item: item[0].start_sentence)
    if all(CLIP_ID.fullmatch(item[0].id) for item in kept):
        rename = {item[0].id: item[0].id for item in kept}
    else:
        rename = {item[0].id: f"c{i}" for i, item in enumerate(kept, start=1)}
        warnings.append("renamed malformed clip IDs to c1..cN; the notes may use the old IDs")
    order = [rename[i] for i in dict.fromkeys(selection.suggested_order) if i in rename]
    for item in kept:  # added required passages go before the first clip later in the talk
        name = rename[item[0].id]
        if item[0].id in added and name not in order:
            start = item[0].start_sentence
            later = next((n for n, i in enumerate(order) if _start(kept, rename, i) > start),
                         len(order))  # fmt: skip
            order.insert(later, name)
    by_score = sorted(kept, key=lambda item: -item[0].score)
    order += [rename[item[0].id] for item in by_score if rename[item[0].id] not in order]
    final = [
        item[0].model_copy(update={"id": rename[item[0].id], "text": item[1]}) for item in kept
    ]
    return ClipSelection(clips=final, suggested_order=order, notes=selection.notes), warnings


def _start(kept: list[tuple[SelectedClip, str, float]], rename: dict[str, str], name: str) -> int:
    return next(item[0].start_sentence for item in kept if rename[item[0].id] == name)


def estimate_input_tokens(system: str, user: str) -> int:
    return round((len(system) + len(user)) / CHARS_PER_TOKEN) + SCHEMA_OVERHEAD_TOKENS


def estimate_input_tokens_from_duration(duration_s: float) -> int:
    system, user = _template()
    words = duration_s * WORDS_PER_SECOND
    return round(words * TOKENS_PER_WORD) + estimate_input_tokens(system, user)


def estimate_output_tokens(count: int) -> int:
    return OUTPUT_TOKENS_BASE + OUTPUT_TOKENS_PER_CLIP * count
