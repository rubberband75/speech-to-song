"""Arrangement logic, free of I/O: the song's sections on a bar grid.

Each clip plays in its own speech_bed section, starting on the section's first bar. The
preset's arc is a template whose speech_bed entries are slots: the clips, in play order,
are shared out over the slots as consecutive groups that balance speech time (a `hook`
leans to the first slot, an `outro` clip to the last), and a group's clips play back to
back. With fewer clips than slots, the first slots and the last one are kept. Other roles
take the preset's bar counts.

Claude can optionally propose the arc instead, as a list of parts (a music section with
its bars, or a speech passage with its clips). Its plan is checked here, and the preset's
arc is used whenever it can't be.
"""

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from string import Template
from typing import Literal

from speech2song.config import SPEECH_ROLE, Preset
from speech2song.llm.claude import load_prompt
from speech2song.models import ArcPart, Arrangement, Clip, Melody, Section

BEATS_PER_BAR = 4
ROLE_HINT_COST = 0.5  # a hook outside the first slot, or an outro clip outside the last
MAX_PART_BARS = 32  # longest music section Claude may propose
MAX_PARTS = 40
MAX_MUSIC_BARS = 160  # music (non-speech) bars in a proposed arc, in total
SONG_TAIL_S = 2.0  # the mix runs this long past the last bar, so tails ring out
RISE_STEP = 0.25  # a rising section with nothing louder ahead climbs this much
ENERGY_BLOCKS = " ▁▂▃▄▅▆▇█"


@dataclass(frozen=True)
class ClipInfo:
    id: str
    role: str  # Claude's role for the clip: hook, build, payoff, breakdown, outro
    text: str
    duration_s: float  # the cut file
    spoken_s: float  # from the start of the cut file to the end of the last word


def clip_info(clip: Clip) -> ClipInfo:
    spoken = max(0.0, min(clip.duration_s, clip.nominal_end_s - clip.start_s))
    return ClipInfo(clip.id, clip.role, clip.text, clip.duration_s, spoken)


def beat_seconds(bpm: float) -> float:
    return 60.0 / bpm


def bar_seconds(bpm: float) -> float:
    return BEATS_PER_BAR * 60.0 / bpm


def bed_bars(clip: ClipInfo, bpm: float, tail_beats: float) -> int:
    """Whole bars that hold the clip file and `tail_beats` after its last word."""
    beat = beat_seconds(bpm)
    beats = max(clip.duration_s / beat, clip.spoken_s / beat + tail_beats)
    return max(1, math.ceil(beats / BEATS_PER_BAR - 1e-9))


# --- The preset's arc ----------------------------------------------------------------------


def _group_cost(groups: Sequence[Sequence[ClipInfo]]) -> float:
    """Imbalance of speech time across groups, plus role-hint penalties."""
    totals = [sum(c.spoken_s for c in group) for group in groups]
    mean = sum(totals) / len(totals)
    balance = sum((t - mean) ** 2 for t in totals) / max(mean, 1e-9) ** 2
    hints = 0.0
    last = len(groups) - 1
    for index, group in enumerate(groups):
        for clip in group:
            if clip.role == "hook" and index != 0:
                hints += ROLE_HINT_COST
            if clip.role == "outro" and index != last:
                hints += ROLE_HINT_COST
    return balance + hints


def group_clips(clips: Sequence[ClipInfo], slots: int) -> list[list[ClipInfo]]:
    """Split clips (in play order) into consecutive groups, one per slot, choosing the
    split with the lowest `_group_cost`. With no more clips than slots, each clip is its
    own group."""
    if slots < 1:
        raise ValueError("the arc has no speech_bed slot")
    if len(clips) <= slots:
        return [[clip] for clip in clips]
    best: list[list[ClipInfo]] = []
    best_cost = math.inf
    for cuts in itertools.combinations(range(1, len(clips)), slots - 1):
        bounds = (0, *cuts, len(clips))
        groups = [list(clips[a:b]) for a, b in itertools.pairwise(bounds)]
        cost = _group_cost(groups)
        if cost < best_cost - 1e-12:
            best, best_cost = groups, cost
    return best


def used_slots(groups: int, slots: int) -> list[int]:
    """Which slots hold the groups: all of them, or the first ones and the last."""
    if groups >= slots:
        return list(range(slots))
    if groups == 1:
        return [0]
    return [*range(groups - 1), slots - 1]


def default_parts(preset: Preset, clips: Sequence[ClipInfo]) -> list[ArcPart]:
    """The preset's arc with the clips shared out over its speech_bed slots."""
    slot_positions = [i for i, role in enumerate(preset.arc) if role == SPEECH_ROLE]
    groups = group_clips(clips, len(slot_positions))
    chosen = used_slots(len(groups), len(slot_positions))
    group_at = {slot_positions[s]: group for s, group in zip(chosen, groups, strict=True)}
    parts = []
    for position, role in enumerate(preset.arc):
        if role != SPEECH_ROLE:
            parts.append(ArcPart(role=role, bars=preset.role_bars(role)))
        elif position in group_at:
            parts.append(ArcPart(role=role, clips=[c.id for c in group_at[position]]))
    return parts


def check_parts(parts: Sequence[ArcPart], clips: Sequence[ClipInfo], preset: Preset) -> list[str]:
    """Problems with a proposed arc (empty if it can be used)."""
    problems = []
    if not parts:
        return ["the arc has no parts"]
    if len(parts) > MAX_PARTS:
        problems.append(f"the arc has {len(parts)} parts; at most {MAX_PARTS} are allowed")
    seen: list[str] = []
    music_bars = 0
    for number, part in enumerate(parts, start=1):
        if part.role not in preset.section_roles:
            roles = ", ".join(preset.section_roles)
            problems.append(f"part {number}: unknown role {part.role!r} (use one of {roles})")
            continue
        if part.role == SPEECH_ROLE:
            if not part.clips:
                problems.append(f"part {number}: a speech_bed part needs at least one clip")
            seen += part.clips
            continue
        if part.clips:
            problems.append(f"part {number}: only speech_bed parts carry clips ({part.role} "
                            f"lists {', '.join(part.clips)})")  # fmt: skip
        if not 1 <= part.bars <= MAX_PART_BARS:
            problems.append(f"part {number}: {part.role} has {part.bars} bars; use 1 to "
                            f"{MAX_PART_BARS}")  # fmt: skip
        music_bars += part.bars
    order = [c.id for c in clips]
    if seen != order:
        problems.append(f"every clip must appear exactly once, in this order: {', '.join(order)} "
                        f"(the arc has: {', '.join(seen) or 'none'})")  # fmt: skip
    if music_bars > MAX_MUSIC_BARS:
        problems.append(f"the music sections total {music_bars} bars; keep them under "
                        f"{MAX_MUSIC_BARS}")  # fmt: skip
    return problems


# --- Sections --------------------------------------------------------------------------------


def _phrase_chords(melody: Melody, clip_id: str | None) -> list[str]:
    phrases = {phrase.clip_id: phrase for phrase in melody.clips}
    phrase = phrases.get(clip_id or "") or phrases.get(melody.main_clip)
    if phrase is None or not phrase.chords:
        return [melody.key.split()[0] + ("m" if melody.mode == "minor" else "")]
    return [chord.name for chord in phrase.chords]


def _cycle(chords: list[str], bars: int) -> list[str]:
    return [chords[i % len(chords)] for i in range(bars)]


def build_arrangement(
    parts: Sequence[ArcPart],
    clips: Sequence[ClipInfo],
    melody: Melody,
    preset: Preset,
    *,
    arc_source: Literal["preset", "claude"] = "preset",
    notes: str | None = None,
    warnings: Sequence[str] = (),
) -> Arrangement:
    """Sections on the bar grid. Speech beds take their clip's chords from the melody;
    other sections loop the main phrase's chords. Sections in `melody.layer_roles` note
    the clip heard last, whose melody the layer replays there."""
    bpm = melody.bpm
    tail = preset.speech_interaction.tail_beats
    by_id = {clip.id: clip for clip in clips}
    main_chords = _phrase_chords(melody, None)
    sections: list[Section] = []
    bar = 0
    last_heard: str | None = None

    def add(role: str, bars: int, **fields: object) -> None:
        nonlocal bar
        spec = preset.section_roles[role]
        sections.append(
            Section(
                id=f"s{len(sections) + 1}",
                role=role,
                start_bar=bar,
                bars=bars,
                energy=spec.energy,
                shape=spec.shape,
                styles=list(spec.styles),
                start_s=round(bar * bar_seconds(bpm), 4),
                seconds=round(bars * bar_seconds(bpm), 4),
                **fields,  # type: ignore[arg-type]
            )
        )
        bar += bars

    for part in parts:
        if part.role == SPEECH_ROLE:
            for clip_id in part.clips:
                bars = bed_bars(by_id[clip_id], bpm, tail)
                add(part.role, bars, chords=_cycle(_phrase_chords(melody, clip_id), bars),
                    clip_id=clip_id)  # fmt: skip
                last_heard = clip_id
        else:
            replay = last_heard if part.role in preset.melody.layer_roles else None
            add(part.role, part.bars, chords=_cycle(main_chords, part.bars), melody_phrase=replay)
    return Arrangement(
        bpm=bpm,
        key=melody.key,
        arc_source=arc_source,
        sections=sections,
        total_bars=bar,
        total_seconds=round(bar * bar_seconds(bpm), 4),
        notes=notes,
        warnings=list(warnings),
    )


def clip_start_s(section: Section, bpm: float) -> float:
    return (section.start_bar * BEATS_PER_BAR + section.clip_offset_beats) * beat_seconds(bpm)


def validate_arrangement(arrangement: Arrangement, clips: dict[str, ClipInfo]) -> list[str]:
    """Structural problems (the arrangement can't be used) in a possibly hand-edited file:
    sections must be contiguous from bar 0, clips must exist, start inside the song, and
    not overlap each other."""
    problems = []
    if not arrangement.sections:
        return ["the arrangement has no sections"]
    if arrangement.bpm <= 0:
        problems.append("bpm must be positive")
        return problems
    bar = 0
    placed: list[tuple[float, float, str]] = []
    for section in arrangement.sections:
        if section.start_bar != bar:
            problems.append(f"{section.id} starts at bar {section.start_bar}; the previous "
                            f"section ends at bar {bar}")  # fmt: skip
        if section.bars < 1:
            problems.append(f"{section.id} must be at least 1 bar long")
        bar = section.start_bar + max(section.bars, 1)
        if section.clip_id is None:
            continue
        clip = clips.get(section.clip_id)
        if clip is None:
            problems.append(f"{section.id} names unknown clip {section.clip_id!r}")
            continue
        start = clip_start_s(section, arrangement.bpm)
        if start < 0:
            problems.append(f"{section.id}: {clip.id} would start before the song does")
        placed.append((start, start + clip.duration_s, f"{section.id} ({clip.id})"))
    if arrangement.total_bars != bar:
        problems.append(f"total_bars is {arrangement.total_bars}; the sections add up to {bar}")
    placed.sort()
    for (_, end, name), (start, _, other) in itertools.pairwise(placed):
        if start < end - 1e-9:
            problems.append(f"{name} and {other} overlap")
    song_end = bar * bar_seconds(arrangement.bpm) + SONG_TAIL_S
    for _, end, name in placed:
        if end > song_end + 1e-9:
            problems.append(f"{name} runs past the end of the song")
    return problems


def arrangement_warnings(arrangement: Arrangement, kept: Sequence[str]) -> list[str]:
    """Non-fatal oddities: kept clips that the arrangement doesn't play."""
    used = {s.clip_id for s in arrangement.sections if s.clip_id}
    missing = [clip_id for clip_id in kept if clip_id not in used]
    return [f"the arrangement does not play {', '.join(missing)}"] if missing else []


# --- Energy ---------------------------------------------------------------------------------


def energy_bounds(sections: Sequence[Section]) -> list[tuple[float, float]]:
    """(start, end) energy of each section. `rise` climbs to the loudest of the next two
    sections (a build rises through a gap into the drop), `fall` fades to 0."""
    bounds = []
    for index, section in enumerate(sections):
        start = section.energy
        if section.shape == "rise":
            ahead = [s.energy for s in sections[index + 1 : index + 3]]
            target = max(ahead, default=0.0)
            end = target if target > start else min(1.0, start + RISE_STEP)
        elif section.shape == "fall":
            end = 0.0
        else:
            end = start
        bounds.append((start, end))
    return bounds


def bar_energies(arrangement: Arrangement) -> list[float]:
    """Mean energy of every bar."""
    energies = []
    for section, (a, b) in zip(arrangement.sections, energy_bounds(arrangement.sections),
                               strict=True):  # fmt: skip
        for i in range(section.bars):
            energies.append(a + (b - a) * (i + 0.5) / section.bars)
    return energies


# --- Timeline ---------------------------------------------------------------------------------


def _clock(seconds: float) -> str:
    minutes, rest = divmod(round(seconds), 60)
    return f"{minutes}:{rest:02d}"


def timeline_strip(arrangement: Arrangement, width: int = 110) -> list[str]:
    """Two text rows, one column per bar (or per few bars, to fit `width`): energy as
    block characters, and where speech plays."""
    per_column = max(1, math.ceil(arrangement.total_bars / width))
    energies = bar_energies(arrangement)
    speech = [False] * arrangement.total_bars
    for section in arrangement.sections:
        if section.clip_id:
            for bar in range(section.start_bar, section.start_bar + section.bars):
                speech[bar] = True
    energy_row, speech_row = [], []
    for start in range(0, arrangement.total_bars, per_column):
        chunk = energies[start : start + per_column]
        level = sum(chunk) / len(chunk)
        energy_row.append(ENERGY_BLOCKS[round(level * (len(ENERGY_BLOCKS) - 1))])
        speech_row.append("━" if any(speech[start : start + per_column]) else " ")
    scale = "1 bar" if per_column == 1 else f"{per_column} bars"
    return [
        f"energy │{''.join(energy_row)}│",
        f"speech │{''.join(speech_row)}│  (one column = {scale})",
    ]


def timeline_rows(arrangement: Arrangement, clip_texts: dict[str, str]) -> list[list[str]]:
    """Table rows: section, bars, time, role, energy, and the clip or replayed phrase."""
    rows = []
    bpm = arrangement.bpm
    for section in arrangement.sections:
        start = section.start_bar * bar_seconds(bpm)
        end = (section.start_bar + section.bars) * bar_seconds(bpm)
        if section.clip_id:
            text = clip_texts.get(section.clip_id, "")
            text = text if len(text) <= 48 else text[:47] + "…"
            what = f"{section.clip_id}  {text}"
        elif section.melody_phrase:
            what = f"(melody of {section.melody_phrase})"
        else:
            what = ""
        energy = f"{section.energy:.2f}" + {"flat": "", "rise": " ↗", "fall": " ↘"}[section.shape]
        rows.append([
            section.id, f"{section.start_bar}", f"{section.bars}",
            f"{_clock(start)}-{_clock(end)}", section.role, energy, what,
        ])  # fmt: skip
    return rows


# --- Claude's arc refinement (optional, paid) -----------------------------------------------

ARC_OUTPUT_TOKENS = 3000  # adaptive thinking plus a short JSON answer
CHARS_PER_TOKEN = 2.2  # as calibrated for clip selection (numbers tokenize densely)
SCHEMA_OVERHEAD_TOKENS = 600


def arc_schema(roles: Sequence[str], clip_ids: Sequence[str]) -> dict[str, object]:
    """JSON schema for Claude's answer. Bar limits are checked in code (structured
    outputs don't support numeric bounds)."""
    part = {
        "type": "object",
        "properties": {
            "role": {"type": "string", "enum": list(roles)},
            "bars": {"type": "integer"},
            "clips": {"type": "array", "items": {"type": "string", "enum": list(clip_ids)}},
        },
        "required": ["role", "bars", "clips"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"parts": {"type": "array", "items": part}, "notes": {"type": "string"}},
        "required": ["parts", "notes"],
        "additionalProperties": False,
    }


def render_parts(parts: Sequence[ArcPart]) -> str:
    lines = []
    for part in parts:
        if part.role == SPEECH_ROLE:
            lines.append(f"- speech_bed: {', '.join(part.clips)}")
        else:
            lines.append(f"- {part.role}: {part.bars} bars")
    return "\n".join(lines)


def build_arc_prompt(
    preset: Preset, clips: Sequence[ClipInfo], bpm: float, notes: str | None
) -> tuple[str, str]:
    system, user = load_prompt("arrange")
    tail = preset.speech_interaction.tail_beats
    roles = "\n".join(
        f"- {name}: energy {spec.energy:g}, "
        + ("fitted to its clips" if name == SPEECH_ROLE else f"{preset.role_bars(name)} bars")
        + (f"; {', '.join(spec.styles)}" if spec.styles else "")
        for name, spec in preset.section_roles.items()
    )
    clip_lines = "\n".join(
        f'- {c.id} ({c.role}, {c.spoken_s:.1f} s, {bed_bars(c, bpm, tail)} bars): "{c.text}"'
        for c in clips
    )
    rendered = Template(user).substitute(
        preset_description=" ".join(preset.description.split()),
        bpm=f"{bpm:g}",
        bar_seconds=f"{bar_seconds(bpm):.2f}",
        roles=roles,
        arc=", ".join(preset.arc),
        clips=clip_lines,
        notes=notes or "none",
        default_parts=render_parts(default_parts(preset, clips)),
    )
    return system, rendered


def arc_feedback_prompt(user: str, parts: Sequence[ArcPart], problems: Sequence[str]) -> str:
    listed = "\n".join(f"- {p}" for p in problems)
    return (
        f"{user}\n\nYour previous answer had these problems:\n{listed}\n\n"
        f"Previous parts:\n{render_parts(parts)}\n\n"
        "Return a corrected, complete answer that fixes every problem."
    )


def estimate_arc_input_tokens(system: str, user: str) -> int:
    return round((len(system) + len(user)) / CHARS_PER_TOKEN) + SCHEMA_OVERHEAD_TOKENS
