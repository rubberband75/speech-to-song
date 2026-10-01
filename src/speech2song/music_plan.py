"""The Eleven Music composition plan for an arrangement (music_v2/v2.5 chunks). Pure.

Docs read 2026-10-01: a plan is up to 30 chunks; a generation chunk has `text` (a
section name in brackets, inline directions in braces), `duration_ms` (3,000-120,000,
always enforced on v2), `positive_styles`/`negative_styles` (up to 50 each, English),
`context_adherence`, and optionally `conditioning_ref` + `condition_strength`. The first
chunk's styles set the tone of the whole song; the model holds a stated BPM and key.

Sections become chunks: neighbours with the same role merge (two speech beds in a row
are one quiet stretch of music), anything shorter than 3 s (a one-bar gap) folds into
the chunk before it with a timed direction, and anything over 2 minutes is split.
Chunk boundaries sit exactly on the arrangement's bar times, rounded to milliseconds.
"""

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from speech2song.arrangement import bar_seconds
from speech2song.config import SPEECH_ROLE, Preset
from speech2song.models import Arrangement, Section

MIN_CHUNK_MS = 3_000
MAX_CHUNK_MS = 120_000
MAX_CHUNKS = 30
MAX_STYLES = 50
MAX_REFERENCE_MS = 30_000
MAX_SONG_MS = 600_000  # composition plans: 3 s to 10 minutes
INSTRUMENTAL = "instrumental only"
NO_VOICES = ["vocals", "singing", "lyrics", "voiceover", "spoken word", "rap"]
SPEECH_LABEL = "Ambient Bed"  # a "speech" label could invite a generated voice


@dataclass
class ChunkLayout:
    """Which arrangement sections a chunk covers, and where (song milliseconds)."""

    sections: list[str]
    roles: list[str]
    start_ms: int
    end_ms: int
    folded: list[tuple[str, int]] = field(default_factory=list)  # (section id, ms) folded in

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


def section_bounds_ms(arrangement: Arrangement) -> list[tuple[Section, int, int]]:
    """Each section's start and end in whole milliseconds; the ends tile the song."""
    bar = bar_seconds(arrangement.bpm)
    bounds = []
    for section in arrangement.sections:
        start = round(section.start_bar * bar * 1000)
        end = round((section.start_bar + section.bars) * bar * 1000)
        bounds.append((section, start, end))
    return bounds


def _label(role: str) -> str:
    return SPEECH_LABEL if role == SPEECH_ROLE else role.replace("_", " ").title()


def layout_chunks(arrangement: Arrangement) -> list[ChunkLayout]:
    """Group sections into chunks within the API's duration limits."""
    chunks: list[ChunkLayout] = []
    for section, start, end in section_bounds_ms(arrangement):
        last = chunks[-1] if chunks else None
        if (last is not None and last.roles[-1] == section.role
                and end - last.start_ms <= MAX_CHUNK_MS):  # fmt: skip
            last.sections.append(section.id)
            last.end_ms = end
            continue
        chunks.append(ChunkLayout([section.id], [section.role], start, end))
    merged: list[ChunkLayout] = []
    for chunk in chunks:  # fold chunks under 3 s into the one before (or after, if first)
        if chunk.duration_ms < MIN_CHUNK_MS and merged:
            host = merged[-1]
            host.sections += chunk.sections
            host.folded.append((chunk.sections[0], chunk.duration_ms))
            host.end_ms = chunk.end_ms
        else:
            merged.append(chunk)
    if len(merged) > 1 and merged[0].duration_ms < MIN_CHUNK_MS:
        first = merged.pop(0)
        merged[0].sections[:0] = first.sections
        merged[0].start_ms = first.start_ms
    result: list[ChunkLayout] = []
    for chunk in merged:  # split chunks over 2 minutes into equal parts
        parts = max(1, math.ceil(chunk.duration_ms / MAX_CHUNK_MS))
        edges = [chunk.start_ms + round(i * chunk.duration_ms / parts) for i in range(parts + 1)]
        for a, b in itertools.pairwise(edges):
            result.append(ChunkLayout(list(chunk.sections), list(chunk.roles), a, b,
                                      list(chunk.folded) if b == chunk.end_ms else []))  # fmt: skip
    return result


def _dedupe(styles: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for style in styles:
        key = style.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(style.strip())
    return out[:MAX_STYLES]


def tempo_and_key(arrangement: Arrangement, preset: Preset) -> list[str]:
    styles = [f"{arrangement.bpm:g} BPM", arrangement.key]
    if preset.tempo.feel != "normal":
        styles.append(f"{preset.tempo.feel} feel")
    return styles


def build_plan(
    arrangement: Arrangement,
    preset: Preset,
    *,
    context_adherence: str = "high",
    reference_song_id: str | None = None,
    reference_ms: int = MAX_REFERENCE_MS,
    condition_strength: str = "low",
) -> tuple[dict, list[ChunkLayout]]:
    """({"chunks": [...]}, the chunk layout). Styles and shapes come from the arrangement's
    sections (hand edits count); the first chunk adds the preset's global styles. Every
    chunk states tempo, key and "instrumental only" and bans voices."""
    layout = layout_chunks(arrangement)
    by_id = {section.id: section for section in arrangement.sections}
    common = tempo_and_key(arrangement, preset)
    negative = _dedupe([*preset.negative_styles, *NO_VOICES])
    chunks = []
    for index, chunk in enumerate(layout):
        main = by_id[chunk.sections[0]]
        lines = [f"[{_label(main.role)}]"]
        timed = []
        for section_id, ms in chunk.folded:
            folded = by_id[section_id]
            cue = folded.styles[0] if folded.styles else folded.role.replace("_", " ")
            lines.append(f"{{{cue}}}")
            timed.append(f"ends with {ms / 1000:.1f} seconds of {', '.join(folded.styles) or cue}")
        positive = [*common, INSTRUMENTAL, *main.styles, *timed]
        if index == 0:
            positive = [*common, INSTRUMENTAL, *preset.positive_styles, *main.styles, *timed]
        if main.shape == "rise":
            positive.append("steadily building")
        elif main.shape == "fall":
            positive.append("fading out to silence at the end")
        item: dict = {
            "text": "\n".join(lines),
            "duration_ms": chunk.duration_ms,
            "positive_styles": _dedupe(positive),
            "negative_styles": negative,
            "context_adherence": context_adherence,
        }
        if index == 0 and reference_song_id:
            item["conditioning_ref"] = {
                "song_id": reference_song_id,
                "range": {"start_ms": 0, "end_ms": min(reference_ms, MAX_REFERENCE_MS)},
            }
            item["condition_strength"] = condition_strength
        chunks.append(item)
    return {"chunks": chunks}, layout


def check_plan(plan: dict) -> list[str]:
    """The API's documented limits, checked before any money is spent."""
    chunks = plan.get("chunks", [])
    problems = []
    if not 1 <= len(chunks) <= MAX_CHUNKS:
        problems.append(f"{len(chunks)} chunks; the API allows 1 to {MAX_CHUNKS}")
    total = 0
    for n, chunk in enumerate(chunks, start=1):
        ms = chunk.get("duration_ms", 0)
        total += ms
        if not MIN_CHUNK_MS <= ms <= MAX_CHUNK_MS:
            problems.append(f"chunk {n} is {ms} ms; chunks must be 3,000-120,000 ms")
        for key in ("positive_styles", "negative_styles"):
            if len(chunk.get(key, [])) > MAX_STYLES:
                problems.append(f"chunk {n} has more than {MAX_STYLES} {key}")
    if not MIN_CHUNK_MS <= total <= MAX_SONG_MS:
        problems.append(f"the song is {total} ms; plans must be 3 s to 10 minutes")
    return problems


def plan_minutes(plan: dict) -> float:
    """Minutes of music a plan produces (kept slices count: billing for them is unknown)."""
    return total_ms(plan) / 60_000


# --- Inpainting -------------------------------------------------------------------------------

MIN_RANGE_MS = 50  # shortest audio-reference slice the API accepts


def regeneration_range(layout: Sequence[ChunkLayout], section_id: str,
                       bounds: dict[str, tuple[int, int]]) -> tuple[int, int]:  # fmt: skip
    """The song range to regenerate for a section: the section itself when it is long
    enough for a generation chunk, else the whole chunk that holds it."""
    start, end = bounds[section_id]
    if end - start >= MIN_CHUNK_MS:
        return start, end
    chunk = next(c for c in layout if section_id in c.sections)
    return chunk.start_ms, chunk.end_ms


def _kept(song_id: str, start: int, end: int) -> list[dict]:
    """Audio-reference chunks that keep [start, end) of the stored song, in pieces no
    longer than a chunk may be."""
    if end - start < MIN_RANGE_MS:
        return []
    parts = max(1, math.ceil((end - start) / MAX_CHUNK_MS))
    edges = [start + round(i * (end - start) / parts) for i in range(parts + 1)]
    return [{"song_id": song_id, "range": {"start_ms": a, "end_ms": b}}
            for a, b in itertools.pairwise(edges)]  # fmt: skip


def inpaint_plan(
    arrangement: Arrangement,
    preset: Preset,
    song_id: str,
    section_id: str,
    note: str | None,
    *,
    context_adherence: str = "high",
) -> tuple[dict, tuple[int, int]]:
    """A plan that keeps the stored song around one section and regenerates that section
    with its styles (plus `note`). Returns the plan and the regenerated range (ms)."""
    by_id = {s.id: s for s in arrangement.sections}
    if section_id not in by_id:
        raise ValueError(f"no section {section_id!r} (sections: {', '.join(by_id)})")
    bounds = {s.id: (a, b) for s, a, b in section_bounds_ms(arrangement)}
    total = section_bounds_ms(arrangement)[-1][2]
    start, end = regeneration_range(layout_chunks(arrangement), section_id, bounds)
    covered = [s for s in arrangement.sections if bounds[s.id][0] < end and bounds[s.id][1] > start]
    main = by_id[section_id] if bounds[section_id][1] - bounds[section_id][0] >= MIN_CHUNK_MS \
        else max(covered, key=lambda s: bounds[s.id][1] - bounds[s.id][0])  # fmt: skip
    positive = [*tempo_and_key(arrangement, preset), INSTRUMENTAL, *main.styles]
    lines = [f"[{_label(main.role)}]"]
    for section in covered:
        if section is not main and section.styles:
            lines.append(f"{{{section.styles[0]}}}")
    if note:
        positive.insert(0, note)
    generated = {
        "text": "\n".join(lines),
        "duration_ms": end - start,
        "positive_styles": _dedupe(positive),
        "negative_styles": _dedupe([*preset.negative_styles, *NO_VOICES]),
        "context_adherence": context_adherence,
    }
    chunks = [*_kept(song_id, 0, start), generated, *_kept(song_id, end, total)]
    return {"chunks": chunks}, (start, end)


def check_inpaint_plan(plan: dict) -> list[str]:
    """Limits for a plan mixing kept audio slices and one generated chunk."""
    chunks = plan.get("chunks", [])
    problems = []
    if not 1 <= len(chunks) <= MAX_CHUNKS:
        problems.append(f"{len(chunks)} chunks; the API allows 1 to {MAX_CHUNKS}")
    for n, chunk in enumerate(chunks, start=1):
        if "song_id" in chunk:
            span = chunk["range"]["end_ms"] - chunk["range"]["start_ms"]
            if not MIN_RANGE_MS <= span <= MAX_CHUNK_MS:
                problems.append(f"chunk {n} keeps {span} ms; slices must be 50-120,000 ms")
        elif not MIN_CHUNK_MS <= chunk.get("duration_ms", 0) <= MAX_CHUNK_MS:
            problems.append(f"chunk {n} generates {chunk.get('duration_ms')} ms; "
                            "chunks must be 3,000-120,000 ms")  # fmt: skip
    return problems


def total_ms(plan: dict) -> int:
    return sum(
        c["range"]["end_ms"] - c["range"]["start_ms"] if "song_id" in c else c["duration_ms"]
        for c in plan.get("chunks", [])
    )
