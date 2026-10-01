"""The Eleven Music composition plan for an arrangement (music_v2/v2.5 chunks). Pure.

Docs read 2026-10-01: a plan is up to 30 chunks; a generation chunk has `text` (a
section name in brackets, inline directions in braces), `duration_ms` (3,000-120,000,
always enforced on v2), `positive_styles`/`negative_styles` (up to 50 each, English),
`context_adherence`, and optionally `conditioning_ref` + `condition_strength`. The first
chunk's styles set the tone of the whole song; the model holds a stated BPM and key.

Sections become chunks: neighbours with the same role merge (two speech beds in a row
are one quiet stretch of music), silent sections (gaps) and anything shorter than 3 s
fold into the chunk before them, and anything over 2 minutes is split. Chunk boundaries
sit exactly on the arrangement's bar times, rounded to milliseconds.

Folded sections get no direction of their own. Inline directions apply to a whole
chunk, so a "{near silence}" cue for a one-bar gap turned a live take's 16-bar build
nearly silent; gaps are muted in the mix instead, where the timing is exact.
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
    leading: list[tuple[Section, int, int]] = []  # silent sections before any music
    for section, start, end in section_bounds_ms(arrangement):
        last = chunks[-1] if chunks else None
        if section.silent and last is None:
            leading.append((section, start, end))
            continue
        if section.silent and last is not None:
            last.sections.append(section.id)
            last.folded.append((section.id, end - start))
            last.end_ms = end
            continue
        if (last is not None and last.roles[-1] == section.role
                and end - last.start_ms <= MAX_CHUNK_MS):  # fmt: skip
            last.sections.append(section.id)
            last.end_ms = end
            continue
        chunks.append(ChunkLayout([section.id], [section.role], start, end))
        if leading:
            chunks[0].sections[:0] = [s.id for s, _, _ in leading]
            chunks[0].folded[:0] = [(s.id, b - a) for s, a, b in leading]
            chunks[0].start_ms = leading[0][1]
            leading = []
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


def grid_ms(arrangement: Arrangement) -> list[int]:
    """Chunk lengths: the timing a take is generated with (see TakeMeta.grid_ms)."""
    return [chunk.duration_ms for chunk in layout_chunks(arrangement)]


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


def shape_styles(section: Section) -> list[str]:
    if section.shape == "rise":  # "steadily building" alone gave a near-silent inpainted build
        return ["steadily building", "rising energy throughout, from moderate to intense"]
    if section.shape == "fall":
        return ["fading out to silence at the end"]
    return []


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
        main = next(by_id[s] for s in chunk.sections if not by_id[s].silent)
        globals_ = preset.positive_styles if index == 0 else []
        positive = [*common, INSTRUMENTAL, *globals_, *main.styles, *shape_styles(main)]
        item: dict = {
            "text": f"[{_label(main.role)}]",
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


def expand_sections(arrangement: Arrangement, specs: Sequence[str]) -> list[str]:
    """Section IDs from `sN` and `sA-sB` specs, in song order. They must be neighbours."""
    order = [s.id for s in arrangement.sections]
    chosen: set[str] = set()
    for spec in specs:
        ends = [part.strip() for part in spec.split("-")]
        for end in ends:
            if end not in order:
                raise ValueError(f"no section {end!r} (sections: {', '.join(order)})")
        first, last = sorted(order.index(end) for end in (ends[0], ends[-1]))
        chosen.update(order[first : last + 1])
    indices = sorted(order.index(section_id) for section_id in chosen)
    if not indices:
        raise ValueError("no section given")
    if indices[-1] - indices[0] + 1 != len(indices):
        raise ValueError("sections to regenerate together must be next to each other "
                         "(e.g. s3-s5)")  # fmt: skip
    return [order[i] for i in indices]


def regeneration_pieces(
    arrangement: Arrangement, section_ids: Sequence[str]
) -> list[tuple[int, int, Section]]:
    """The song range to regenerate for neighbouring sections, as generation chunks
    (start ms, end ms, the section whose styles it gets). Pieces follow the plan's chunk
    boundaries; a piece under 3 s joins its neighbour, and a range under 3 s grows to
    the whole chunks it touches."""
    bounds = {s.id: (a, b) for s, a, b in section_bounds_ms(arrangement)}
    layout = layout_chunks(arrangement)
    start, end = bounds[section_ids[0]][0], bounds[section_ids[-1]][1]
    if end - start < MIN_CHUNK_MS:
        touched = [c for c in layout if c.start_ms < end and c.end_ms > start]
        start, end = touched[0].start_ms, touched[-1].end_ms
    edges: list[list[int]] = []
    for chunk in layout:
        a, b = max(chunk.start_ms, start), min(chunk.end_ms, end)
        if b > a:
            edges.append([a, b])
    merged: list[list[int]] = []
    for piece in edges:
        if merged and (piece[1] - piece[0] < MIN_CHUNK_MS
                       or merged[-1][1] - merged[-1][0] < MIN_CHUNK_MS):  # fmt: skip
            merged[-1][1] = piece[1]
        else:
            merged.append(piece)
    pieces = []
    for a, b in merged:
        inside = [s for s in arrangement.sections
                  if bounds[s.id][0] < b and bounds[s.id][1] > a and not s.silent]  # fmt: skip
        main = max(inside, key=lambda s: min(b, bounds[s.id][1]) - max(a, bounds[s.id][0]))
        pieces.append((a, b, main))
    return pieces


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
    section_ids: Sequence[str],
    note: str | None,
    *,
    context_adherence: str = "high",
) -> tuple[dict, tuple[int, int]]:
    """A plan that keeps the stored song around neighbouring sections and regenerates
    them with their styles (plus `note`). Returns the plan and the regenerated range."""
    by_id = {s.id: s for s in arrangement.sections}
    for section_id in section_ids:
        if section_id not in by_id:
            raise ValueError(f"no section {section_id!r} (sections: {', '.join(by_id)})")
    if all(by_id[section_id].silent for section_id in section_ids):
        raise ValueError(f"{', '.join(section_ids)} is silent in the mix; regenerate the "
                         "sections around it instead")  # fmt: skip
    total = section_bounds_ms(arrangement)[-1][2]
    pieces = regeneration_pieces(arrangement, section_ids)
    common = [*tempo_and_key(arrangement, preset), INSTRUMENTAL]
    negative = _dedupe([*preset.negative_styles, *NO_VOICES])
    generated = []
    for a, b, main in pieces:
        positive = [*([note] if note else []), *common, *main.styles, *shape_styles(main)]
        generated.append({
            "text": f"[{_label(main.role)}]",
            "duration_ms": b - a,
            "positive_styles": _dedupe(positive),
            "negative_styles": negative,
            "context_adherence": context_adherence,
        })  # fmt: skip
    start, end = pieces[0][0], pieces[-1][1]
    chunks = [*_kept(song_id, 0, start), *generated, *_kept(song_id, end, total)]
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
