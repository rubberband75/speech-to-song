"""Clip selection logic, free of I/O: prompt, JSON schema, and checks on Claude's answer.

Claude picks sentence ranges. Before anything is cut, its answer is checked against the
transcript: the sentences must exist, the quoted text must match them, each clip must
fit the duration bounds, clips must not overlap, and the total must fit the speech
budget. One retry with feedback is allowed; after that, unusable clips are dropped.
"""

import hashlib
from dataclasses import dataclass
from importlib import resources
from string import Template
from typing import Literal

from rapidfuzz.distance import Indel

from speech2song.config import Preset
from speech2song.errors import StageError
from speech2song.models import ClipSelection, ClipTargets, RunOptions, SelectedClip, Transcript
from speech2song.text.normalize import normalize_word

CLIP_ROLES = ("hook", "build", "payoff", "breakdown", "outro")
TEXT_MATCH_MIN = 0.8  # normalized similarity between a clip's quote and its sentences

# Rough, offline token estimates (dry runs make no API calls). Deliberately generous.
CHARS_PER_TOKEN = 3.0
SCHEMA_OVERHEAD_TOKENS = 400
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


def clip_targets(preset: Preset, options: RunOptions) -> ClipTargets:
    spec = preset.clips
    return ClipTargets(
        count=options.clips or spec.count,
        min_seconds=spec.min_seconds,
        max_seconds=spec.max_seconds,
        total_speech_seconds=spec.total_speech_seconds,
    )


def _template() -> tuple[str, str]:
    text = resources.files("speech2song.llm").joinpath("prompts/select_clips.md").read_text()
    system = text.split("## System", 1)[1].split("## User", 1)[0].strip()
    user = text.split("## User", 1)[1].strip()
    return system, user


def prompt_digest() -> str:
    """Hash of the prompt template: editing it invalidates cached selections."""
    system, user = _template()
    return hashlib.sha256(f"{system}\n{user}".encode()).hexdigest()[:16]


def render_sentences(transcript: Transcript) -> str:
    return "\n".join(
        f"{s.id} | {s.start:.1f}-{s.end:.1f} | {s.end - s.start:.1f}s | {s.text}"
        for s in transcript.sentences
    )


def build_prompt(transcript: Transcript, preset: Preset, targets: ClipTargets) -> tuple[str, str]:
    system, user = _template()
    rendered = Template(user).substitute(
        preset_description=" ".join(preset.description.split()),
        arc=", ".join(preset.arc),
        count=targets.count,
        min_seconds=f"{targets.min_seconds:g}",
        max_seconds=f"{targets.max_seconds:g}",
        total_speech_seconds=f"{targets.total_speech_seconds:g}",
        sentences=render_sentences(transcript),
    )
    return system, rendered


def feedback_prompt(user: str, answer: ClipSelection, problems: list["Problem"]) -> str:
    listed = "\n".join(f"- {p.message}" for p in problems)
    return (
        f"{user}\n\nYour previous answer had these problems:\n{listed}\n\n"
        f"Previous answer:\n{answer.model_dump_json()}\n\n"
        "Return a corrected, complete answer that fixes every problem."
    )


ProblemKind = Literal["invalid", "overlap", "budget", "count", "order", "score"]
RETRY_KINDS = {"invalid", "overlap", "budget"}


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
        if not text_matches(clip.text, text):
            problems.append(
                Problem(
                    name, "invalid", f'{name}: the text does not match {rng}, which read: "{text}"'
                )
            )
            continue
        if not targets.min_seconds <= duration <= targets.max_seconds:
            problems.append(
                Problem(
                    name,
                    "invalid",
                    f"{name}: {rng} last {duration:.1f} s; clips must be "
                    f"{targets.min_seconds:g}-{targets.max_seconds:g} s",
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
    if len(selection.clips) != targets.count:
        problems.append(
            Problem(
                None,
                "count",
                f"{len(selection.clips)} clips were returned; {targets.count} were asked for",
            )
        )
    order = selection.suggested_order
    if sorted(order) != sorted(seen) or len(set(order)) != len(order):
        problems.append(
            Problem(None, "order", "suggested_order must list every clip id exactly once")
        )
    return problems


def _overlaps(a: SelectedClip, b: SelectedClip) -> bool:
    return a.start_sentence <= b.end_sentence and b.start_sentence <= a.end_sentence


def needs_retry(problems: list[Problem]) -> bool:
    return any(p.kind in RETRY_KINDS for p in problems)


def finalize(
    selection: ClipSelection, transcript: Transcript, targets: ClipTargets
) -> tuple[ClipSelection, list[str]]:
    """Keep the usable clips, resolve overlaps and limits by score, renumber c1..cN in
    time order, and use the transcript's own text for each clip."""
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

    kept: list[tuple[SelectedClip, str, float]] = []
    for item in sorted(clips.values(), key=lambda it: -it[0].score):
        clip = item[0]
        clash = next((k[0].id for k in kept if _overlaps(clip, k[0])), None)
        if clash:
            warnings.append(f"dropped {clip.id}: it overlaps {clash}, which scored higher")
            continue
        kept.append(item)
    while len(kept) > targets.count:
        dropped = kept.pop()  # lowest score (kept is in descending score order)
        warnings.append(f"dropped {dropped[0].id}: more clips than the {targets.count} asked for")
    while kept and sum(item[2] for item in kept) > targets.total_speech_seconds:
        dropped = kept.pop()
        warnings.append(f"dropped {dropped[0].id}: the clips exceeded the speech budget")
    if not kept:
        raise StageError("None of Claude's clips could be used; see 03_selection.json attempts.")

    kept.sort(key=lambda item: item[0].start_sentence)
    rename = {item[0].id: f"c{i}" for i, item in enumerate(kept, start=1)}
    order = [rename[i] for i in dict.fromkeys(selection.suggested_order) if i in rename]
    by_score = sorted(kept, key=lambda item: -item[0].score)
    order += [rename[item[0].id] for item in by_score if rename[item[0].id] not in order]
    final = [
        item[0].model_copy(update={"id": rename[item[0].id], "text": item[1]}) for item in kept
    ]
    return ClipSelection(clips=final, suggested_order=order, notes=selection.notes), warnings


def estimate_input_tokens(system: str, user: str) -> int:
    return round((len(system) + len(user)) / CHARS_PER_TOKEN) + SCHEMA_OVERHEAD_TOKENS


def estimate_input_tokens_from_duration(duration_s: float) -> int:
    system, user = _template()
    words = duration_s * WORDS_PER_SECOND
    return round(words * TOKENS_PER_WORD) + estimate_input_tokens(system, user)


def estimate_output_tokens(count: int) -> int:
    return OUTPUT_TOKENS_BASE + OUTPUT_TOKENS_PER_CLIP * count
