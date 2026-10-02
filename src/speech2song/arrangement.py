"""Arrangement logic, free of I/O: the song's sections on a bar grid.

Each clip plays in its own speech_bed section, starting on the section's first bar. The
preset's arc is a template whose speech_bed entries are slots: the clips, in play order,
are shared out over the slots as consecutive groups that balance speech time (a `hook`
leans to the first slot, an `outro` clip to the last), and a group's clips play back to
back. The parts of a long quote always share a slot, with `part_pause_beats` of music
after each part but the last. With fewer quotes than slots, the first slots and the last
one are kept. Other roles take the preset's bar counts.

Claude can optionally propose the arc instead, as a list of parts (a music section with
its bars, or a speech passage with its clips, how the music meets it and a few style
words). Its plan is checked here, and the preset's arc is used whenever it can't be.

M7 song form: a passage's music starts `lead_in_beats` before its first quote, so the
music has settled when the voice comes in; a passage played "alone" has no music at all
(the music stops on its bar line and returns on the next downbeat). Music sections take
a progression that fits the nearby speech melody, and a role's first and last
occurrence get styles of their own. The ending is realized by the plan and the mix
(the music model fades out at the end of any generation, whatever it is asked): a song
ending on a held chord or a stop comes home to the tonic in its last bar.
"""

import itertools
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from string import Template
from typing import Literal

import numpy as np

from speech2song.audio.theory import Key, section_progression
from speech2song.config import SPEECH_ROLE, Preset
from speech2song.llm.claude import load_prompt
from speech2song.models import ArcPart, Arrangement, Clip, Ending, Melody, Section

BEATS_PER_BAR = 4
ROLE_HINT_COST = 0.5  # a hook outside the first slot, or an outro clip outside the last
MAX_PART_BARS = 32  # longest music section Claude may propose
MAX_PARTS = 40
MAX_MUSIC_BARS = 160  # music (non-speech) bars in a proposed arc, in total
MAX_ALONE = 2  # passages played without music, per song
MAX_PART_STYLES = 4
MAX_STYLE_CHARS = 80
# Style words that could bring a generated voice into the music.
VOICE_WORDS = re.compile(r"\b(vocals?|vocalists?|voices?|sing|singing|singers?|sung|lyrics?|"
                         r"choirs?|choral|chants?|chanting|spoken|speech|narrat\w*|raps?|"
                         r"rapping|humming|whisper\w*)\b", re.IGNORECASE)  # fmt: skip
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
    quote: str = ""  # the selected quote this clip is part of ("" for the clip itself)
    last_part: bool = True

    @property
    def quote_id(self) -> str:
        return self.quote or self.id


def clip_info(clip: Clip) -> ClipInfo:
    spoken = max(0.0, min(clip.duration_s, clip.nominal_end_s - clip.start_s))
    return ClipInfo(clip.id, clip.role, clip.text, clip.duration_s, spoken,
                    clip.quote or clip.id, clip.part == clip.parts)  # fmt: skip


def beat_seconds(bpm: float) -> float:
    return 60.0 / bpm


def bar_seconds(bpm: float) -> float:
    return BEATS_PER_BAR * 60.0 / bpm


def bed_bars(
    clip: ClipInfo, bpm: float, tail_beats: float, part_pause_beats: float = 0.0,
    lead_in_beats: float = 0.0,
) -> int:  # fmt: skip
    """Whole bars that hold `lead_in_beats` of music, then the clip file and `tail_beats`
    after its last word (or the longer `part_pause_beats`, when more of its quote
    follows)."""
    beat = beat_seconds(bpm)
    tail = tail_beats if clip.last_part else max(tail_beats, part_pause_beats)
    beats = lead_in_beats + max(clip.duration_s / beat, clip.spoken_s / beat + tail)
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


def quote_runs(clips: Sequence[ClipInfo]) -> list[list[ClipInfo]]:
    """Clips in play order, with neighbouring parts of the same quote kept together."""
    runs: list[list[ClipInfo]] = []
    for clip in clips:
        if runs and runs[-1][-1].quote_id == clip.quote_id:
            runs[-1].append(clip)
        else:
            runs.append([clip])
    return runs


def group_clips(clips: Sequence[ClipInfo], slots: int) -> list[list[ClipInfo]]:
    """Split clips (in play order) into consecutive groups, one per slot, choosing the
    split with the lowest `_group_cost`. A quote's parts are never split up. With no more
    quotes than slots, each quote is its own group."""
    if slots < 1:
        raise ValueError("the arc has no speech_bed slot")
    runs = quote_runs(clips)
    if len(runs) <= slots:
        return [list(run) for run in runs]
    best: list[list[ClipInfo]] = []
    best_cost = math.inf
    for cuts in itertools.combinations(range(1, len(runs)), slots - 1):
        bounds = (0, *cuts, len(runs))
        groups = [[clip for run in runs[a:b] for clip in run]
                  for a, b in itertools.pairwise(bounds)]  # fmt: skip
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
    quote_of = {c.id: c.quote_id for c in clips}
    passages: dict[str, set[int]] = {}
    for number, part in enumerate(parts):
        for clip_id in part.clips:
            passages.setdefault(quote_of.get(clip_id, clip_id), set()).add(number)
    for quote, where in passages.items():
        if len(where) > 1:
            problems.append(f"the parts of quote {quote} must play in the same speech_bed part")
    alone = [n for n, part in enumerate(parts, start=1)
             if part.role == SPEECH_ROLE and part.treatment == "alone"]  # fmt: skip
    for number in alone:
        quotes = {quote_of.get(c, c) for c in parts[number - 1].clips}
        if len(quotes) > 1:
            problems.append(f"part {number}: a passage played alone holds one quote (it has "
                            f"{', '.join(sorted(quotes))})")  # fmt: skip
    if len(alone) > MAX_ALONE:
        problems.append(f"{len(alone)} passages are played alone; use at most {MAX_ALONE}, "
                        "for the lines that matter most")  # fmt: skip
    for number, part in enumerate(parts, start=1):
        problems += [f"part {number}: {p}" for p in style_problems(part.styles)]
    if music_bars > MAX_MUSIC_BARS:
        problems.append(f"the music sections total {music_bars} bars; keep them under "
                        f"{MAX_MUSIC_BARS}")  # fmt: skip
    return problems


def style_problems(styles: Sequence[str]) -> list[str]:
    """Problems with a part's extra style words (empty if they can go to the model)."""
    problems = []
    if len(styles) > MAX_PART_STYLES:
        problems.append(f"{len(styles)} styles; give at most {MAX_PART_STYLES}")
    for style in styles:
        if len(style) > MAX_STYLE_CHARS:
            problems.append(f"the style {style[:40]!r}... is longer than {MAX_STYLE_CHARS} "
                            "characters")  # fmt: skip
        if (match := VOICE_WORDS.search(style)) is not None:
            problems.append(f"the style {style!r} mentions {match.group(0)!r}; the music must "
                            "stay instrumental (the speaker is the only voice)")  # fmt: skip
    return problems


# --- Sections --------------------------------------------------------------------------------


def _phrase_chords(melody: Melody, clip_id: str | None) -> list[str]:
    phrases = {phrase.clip_id: phrase for phrase in melody.clips}
    phrase = phrases.get(clip_id or "") or phrases.get(melody.main_clip)
    if phrase is None or not phrase.chords:
        return [_tonic_chord(melody)]
    return [chord.name for chord in phrase.chords]


def _tonic_chord(melody: Melody) -> str:
    return melody.key.split()[0] + ("m" if melody.mode == "minor" else "")


def _cycle(chords: list[str], bars: int) -> list[str]:
    return [chords[i % len(chords)] for i in range(bars)]


def _pitch_weights(melody: Melody, clip_id: str | None) -> np.ndarray:
    """How long each pitch class sounds in a clip's melody (else the main phrase's)."""
    phrases = {phrase.clip_id: phrase for phrase in melody.clips}
    phrase = phrases.get(clip_id or "") or phrases.get(melody.main_clip)
    weights = np.zeros(12)
    for note in phrase.notes if phrase is not None else []:
        weights[note.midi % 12] += note.beats
    return weights


def _bed_chords(melody: Melody, clip_id: str, bars: int, offset_beats: float) -> list[str]:
    """The clip's own chords, starting where the clip does (a lead-in holds its first)."""
    own = _phrase_chords(melody, clip_id)
    lead = min(bars, int(offset_beats // BEATS_PER_BAR))
    return [own[0]] * lead + _cycle(own, bars - lead)


def _has_notes(melody: Melody, clip_id: str | None) -> bool:
    return any(p.clip_id == clip_id and p.notes for p in melody.clips)


def build_arrangement(
    parts: Sequence[ArcPart],
    clips: Sequence[ClipInfo],
    melody: Melody,
    preset: Preset,
    *,
    arc_source: Literal["preset", "claude"] = "preset",
    notes: str | None = None,
    warnings: Sequence[str] = (),
    ending: Ending | None = None,
) -> Arrangement:
    """Sections on the bar grid (an M7 arrangement, plan version 2).

    A passage's first clip starts `lead_in_beats` into its bed (`alone_lead_in_beats`
    when it plays alone). Beds take their clip's chords; music sections take a
    progression for their role that fits the melody heard last, different from the
    role's previous section. A role's first and last occurrence add their styles, and a
    part's own styles (Claude's) come last. The song's first music section and the music
    right after each passage are conditioned on a quote's melody (`melody_ref`). With
    `ending` (else the preset's) "fade", the last music section falls; otherwise it
    holds steady and lands on the tonic in its last bar, where the mix rings it out.
    Sections in `melody.layer_roles` note the clip heard last, for the melody layer."""
    bpm = melody.bpm
    speech = preset.speech_interaction
    tail, pause = speech.tail_beats, speech.part_pause_beats
    ending = ending or preset.ending
    key = Key(melody.tonic, melody.mode)
    by_id = {clip.id: clip for clip in clips}
    main_chords = _phrase_chords(melody, None)
    music_roles = [p.role for p in parts if p.role != SPEECH_ROLE]
    seen: dict[str, int] = {}
    last_progression: dict[str, tuple[int, ...]] = {}
    sections: list[Section] = []
    bar = 0
    last_heard: str | None = None
    answer: str | None = melody.main_clip  # the quote the next music section answers

    def add(role: str, bars: int, **fields: object) -> None:
        nonlocal bar
        spec = preset.section_roles[role]
        values = {"energy": spec.energy, "shape": spec.shape, "styles": list(spec.styles),
                  "silent": spec.silent, **fields}  # fmt: skip
        sections.append(
            Section(
                id=f"s{len(sections) + 1}",
                role=role,
                start_bar=bar,
                bars=bars,
                start_s=round(bar * bar_seconds(bpm), 4),
                seconds=round(bars * bar_seconds(bpm), 4),
                **values,  # type: ignore[arg-type]
            )
        )
        bar += bars

    def chords_for(role: str, bars: int) -> list[str]:
        found = section_progression(role, key, bars, _pitch_weights(melody, last_heard),
                                    last_progression.get(role))  # fmt: skip
        if found is None:
            return _cycle(main_chords, bars)
        last_progression[role] = found[0]
        return found[1]

    for part in parts:
        if part.role == SPEECH_ROLE:
            alone = part.treatment == "alone"
            lead = speech.alone_lead_in_beats if alone else speech.lead_in_beats
            for n, clip_id in enumerate(part.clips):
                offset = lead if n == 0 else 0.0
                bars = bed_bars(by_id[clip_id], bpm, tail, pause, offset)
                styles = [*preset.section_roles[SPEECH_ROLE].styles, *part.styles]
                add(
                    part.role,
                    bars,
                    chords=_bed_chords(melody, clip_id, bars, offset),
                    clip_id=clip_id,
                    clip_offset_beats=offset,
                    treatment=part.treatment,
                    styles=[] if alone else styles,
                    **({"energy": 0.0} if alone else {}),
                )
                last_heard = clip_id
            answer = part.clips[-1] if part.clips else answer
            continue
        spec = preset.section_roles[part.role]
        seen[part.role] = seen.get(part.role, 0) + 1
        occurrence = []
        if music_roles.count(part.role) > 1:
            if seen[part.role] == 1:
                occurrence = spec.first_styles
            elif seen[part.role] == music_roles.count(part.role):
                occurrence = spec.last_styles
        replay = last_heard if part.role in preset.melody.layer_roles else None
        reference = None
        if not spec.silent:  # only the music right after a passage answers it
            reference = answer if answer is not None and _has_notes(melody, answer) else None
            answer = None
        add(part.role, part.bars, chords=chords_for(part.role, part.bars), melody_phrase=replay,
            styles=[*spec.styles, *occurrence, *part.styles], melody_ref=reference)  # fmt: skip
    music = [i for i, s in enumerate(sections) if not s.rest and not s.silent]
    if music:  # how the last music ends
        last = sections[music[-1]]
        if ending == "fade":  # it fades away by itself
            update: dict[str, object] = {"shape": "fall"}
        else:  # it comes home and holds (or stops) there: the mix rings that chord out
            update = {"shape": "flat" if last.shape == "fall" else last.shape,
                      "chords": [*last.chords[:-1], _tonic_chord(melody)]}  # fmt: skip
        sections[music[-1]] = last.model_copy(update=update)
    return Arrangement(
        bpm=bpm,
        key=melody.key,
        arc_source=arc_source,
        sections=sections,
        total_bars=bar,
        total_seconds=round(bar * bar_seconds(bpm), 4),
        notes=notes,
        warnings=list(warnings),
        plan_version=2,
        ending=ending,
    )


def music_bars(arrangement: Arrangement) -> list[int]:
    """Where each section starts in the generated music, in bars. A passage played alone
    takes no music time: the take is spliced open there, so later sections start that
    much earlier in the music than in the song."""
    starts, removed = [], 0
    for section in arrangement.sections:
        starts.append(section.start_bar - removed)
        if section.rest:
            removed += section.bars
    return starts


def music_total_bars(arrangement: Arrangement) -> int:
    return arrangement.total_bars - sum(s.bars for s in arrangement.sections if s.rest)


def passages(arrangement: Arrangement) -> list[list[Section]]:
    """Runs of neighbouring speech beds that share a treatment: the stretches of song
    the music treats as one (one quiet bed, or one silence)."""
    runs: list[list[Section]] = []
    for section in arrangement.sections:
        if section.clip_id is None:
            continue
        previous = runs[-1][-1] if runs else None
        if (previous is not None and previous.start_bar + previous.bars == section.start_bar
                and previous.treatment == section.treatment):  # fmt: skip
            runs[-1].append(section)
        else:
            runs.append([section])
    return runs


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


def timeline_rows(
    arrangement: Arrangement, clip_texts: dict[str, str], melody_layer: bool = True
) -> list[list[str]]:
    """Table rows: section, bars, time, role, energy, and the clip or replayed phrase."""
    rows = []
    bpm = arrangement.bpm
    for section in arrangement.sections:
        start = section.start_bar * bar_seconds(bpm)
        end = (section.start_bar + section.bars) * bar_seconds(bpm)
        if section.clip_id:
            text = clip_texts.get(section.clip_id, "")
            text = text if len(text) <= 48 else text[:47] + "…"
            what = f"{section.clip_id}  {text}" + ("  (alone)" if section.rest else "")
        elif section.silent:
            what = "(lift into the next section)"
        elif section.melody_phrase and melody_layer:
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
            "treatment": {"type": "string", "enum": ["under", "alone"]},
            "styles": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["role", "bars", "clips", "treatment", "styles"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "parts": {"type": "array", "items": part},
            "ending": {"type": "string", "enum": ["held_chord", "fade", "stop"]},
            "notes": {"type": "string"},
        },
        "required": ["parts", "ending", "notes"],
        "additionalProperties": False,
    }


def render_parts(parts: Sequence[ArcPart]) -> str:
    lines = []
    for part in parts:
        styles = f" [{'; '.join(part.styles)}]" if part.styles else ""
        if part.role == SPEECH_ROLE:
            alone = " (alone)" if part.treatment == "alone" else ""
            lines.append(f"- speech_bed{alone}: {', '.join(part.clips)}{styles}")
        else:
            lines.append(f"- {part.role}: {part.bars} bars{styles}")
    return "\n".join(lines)


def build_arc_prompt(
    preset: Preset, clips: Sequence[ClipInfo], bpm: float, notes: str | None
) -> tuple[str, str]:
    system, user = load_prompt("arrange")
    speech = preset.speech_interaction
    tail, pause = speech.tail_beats, speech.part_pause_beats
    roles = "\n".join(
        f"- {name}: energy {spec.energy:g}, "
        + ("fitted to its clips" if name == SPEECH_ROLE else f"{preset.role_bars(name)} bars")
        + (f"; {', '.join(spec.styles)}" if spec.styles else "")
        for name, spec in preset.section_roles.items()
    )

    def part_of(c: ClipInfo) -> str:
        siblings = [x.id for x in clips if x.quote_id == c.quote_id]
        return f", part {siblings.index(c.id) + 1} of {len(siblings)} of quote {c.quote_id}" \
            if len(siblings) > 1 else ""  # fmt: skip

    clip_lines = "\n".join(
        f"- {c.id} ({c.role}, {c.spoken_s:.1f} s, {bed_bars(c, bpm, tail, pause)} bars"
        f'{part_of(c)}): "{c.text}"'
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
        lead_in=f"{speech.lead_in_beats:g}",
        max_alone=MAX_ALONE,
        max_styles=MAX_PART_STYLES,
        ending=preset.ending,
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
