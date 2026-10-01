"""Mixing logic on arrays: where clips go, the exact speech stem, the processed speech
bus, the music's shaping, and the melody layer's notes. No file I/O.

The speech stem is the clips copied verbatim onto silence (the exactness invariant).
Everything else (gain, high-pass, compression, reverb and delay sends) happens on the
speech bus, which only the mix hears. Each clip's sends are rendered on their own and
faded out before the next clip starts, so tails never run into the next line.

The music is shaped toward the arrangement: sections whose level strays far from their
energy are pulled back toward it, and silent sections (gaps) are cut, leaving only a
reverb tail of the music before them.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from speech2song.arrangement import BEATS_PER_BAR, bar_seconds, beat_seconds, clip_start_s
from speech2song.audio.dsp import db_to_gain
from speech2song.models import Arrangement, ClipMelody, Melody, PlacedClip, Section

TAIL_MAX_S = 6.0  # longest reverb/delay tail after a clip
TAIL_FADE_S = 0.3  # tails fade out over this long before they are cut
COMP_RATIO = 2.5
COMP_ABOVE_LU = 4.0  # compressor threshold, relative to the speech loudness
COMP_ATTACK_MS = 10.0
COMP_RELEASE_MS = 120.0
REVERB_ROOM = 0.75
DELAY_BEATS = 0.75  # dotted eighth
DELAY_FEEDBACK = 0.35


def stereo(audio: np.ndarray) -> np.ndarray:
    audio = audio[:, None] if audio.ndim == 1 else audio
    return audio if audio.shape[1] == 2 else np.repeat(audio[:, :1], 2, axis=1)


def place_clips(
    arrangement: Arrangement, files: dict[str, str], lengths: dict[str, int], sr: int
) -> list[PlacedClip]:
    """Sample positions of every clip the arrangement plays, in time order."""
    placed = []
    for section in arrangement.sections:
        if section.clip_id is None:
            continue
        start = round(clip_start_s(section, arrangement.bpm) * sr)
        placed.append(
            PlacedClip(
                clip_id=section.clip_id,
                section_id=section.id,
                file=files[section.clip_id],
                start_sample=start,
                end_sample=start + lengths[section.clip_id],
                start_s=round(start / sr, 6),
            )
        )
    return sorted(placed, key=lambda p: p.start_sample)


def speech_stem(
    placed: Sequence[PlacedClip], audio: dict[str, np.ndarray], frames: int, channels: int
) -> np.ndarray:
    """The clips copied sample for sample onto silence (float32, the clips' channels)."""
    out = np.zeros((frames, channels), dtype=np.float32)
    for p in placed:
        clip = audio[p.clip_id]
        if p.end_sample > frames:
            raise ValueError(f"{p.clip_id} runs past the end of the mix")
        out[p.start_sample : p.end_sample] = clip
    return out


def highpass(audio: np.ndarray, sr: int, hz: float) -> np.ndarray:
    if hz <= 0:
        return audio
    from scipy.signal import butter, sosfilt

    sos = butter(2, hz, btype="highpass", fs=sr, output="sos")
    return sosfilt(sos, audio, axis=0).astype(np.float32)


def compress(audio: np.ndarray, sr: int, threshold_db: float) -> np.ndarray:
    import pedalboard

    compressor = pedalboard.Compressor(
        threshold_db=threshold_db,
        ratio=COMP_RATIO,
        attack_ms=COMP_ATTACK_MS,
        release_ms=COMP_RELEASE_MS,
    )
    return compressor(audio.T.astype(np.float32), sr).T


def tail_limits(placed: Sequence[PlacedClip], frames: int, sr: int) -> list[int]:
    """Per clip: how many samples its tail may ring after it ends (until the next clip
    starts, or the end of the mix, at most TAIL_MAX_S)."""
    limits = []
    for i, p in enumerate(placed):
        next_start = placed[i + 1].start_sample if i + 1 < len(placed) else frames
        limits.append(max(0, min(round(TAIL_MAX_S * sr), next_start - p.end_sample)))
    return limits


def sends(
    clip: np.ndarray, tail: int, sr: int, *, reverb: float, delay: float, delay_s: float
) -> np.ndarray:
    """Reverb and delay returns (stereo) for one clip, `tail` samples longer than it,
    faded out at the end so nothing is cut abruptly."""
    import pedalboard

    padded = np.concatenate([stereo(clip), np.zeros((tail, 2), dtype=np.float32)])
    wet = np.zeros_like(padded)
    block = padded.T.astype(np.float32)
    if reverb > 0:
        room = pedalboard.Reverb(room_size=REVERB_ROOM, damping=0.5, wet_level=1.0,
                                 dry_level=0.0, width=1.0)  # fmt: skip
        wet += reverb * room(block, sr).T
    if delay > 0:
        echo = pedalboard.Delay(delay_seconds=delay_s, feedback=DELAY_FEEDBACK, mix=1.0)
        wet += delay * echo(block, sr).T
    fade = min(len(wet), round(TAIL_FADE_S * sr), tail) if tail else 0
    if fade:
        wet[len(wet) - fade :] *= np.linspace(1, 0, fade, dtype=np.float32)[:, None]
    elif len(wet):
        wet[-1] = 0.0
    return wet


@dataclass(frozen=True)
class SpeechBus:
    dry: np.ndarray  # processed speech (stereo)
    wet: np.ndarray  # reverb and delay returns (stereo)


def speech_bus(
    placed: Sequence[PlacedClip],
    audio: dict[str, np.ndarray],
    frames: int,
    sr: int,
    *,
    gain_db: float,
    highpass_hz: float,
    threshold_db: float,
    reverb_send: float,
    delay_send: float,
    delay_s: float,
) -> SpeechBus:
    """Gain, high-pass and gentle compression per clip, then the sends."""
    dry = np.zeros((frames, 2), dtype=np.float32)
    wet = np.zeros((frames, 2), dtype=np.float32)
    gain = np.float32(10 ** (gain_db / 20))
    processed = {}
    for p in placed:
        if p.clip_id not in processed:
            clip = highpass(stereo(audio[p.clip_id]) * gain, sr, highpass_hz)
            processed[p.clip_id] = compress(clip, sr, threshold_db)
        dry[p.start_sample : p.end_sample] += processed[p.clip_id]
    for p, tail in zip(placed, tail_limits(placed, frames, sr), strict=True):
        returns = sends(processed[p.clip_id], tail, sr, reverb=reverb_send, delay=delay_send,
                        delay_s=delay_s)  # fmt: skip
        wet[p.start_sample : p.start_sample + len(returns)] += returns
    return SpeechBus(dry, wet)


def delay_seconds(bpm: float) -> float:
    return DELAY_BEATS * beat_seconds(bpm)


# --- Music shaping ----------------------------------------------------------------------------

ENERGY_RAMP_S = 0.5  # gain changes finish where the next section starts
GAP_CUT_S = 0.02  # the fade into a silent section
GAP_RETURN_S = 0.005  # the fade back in, ending where the next section starts
GAP_SOURCE_S = 1.0  # how much music before a gap feeds its reverb tail
GAP_ROOM = 0.9


LATE_ENTRY_DB = 15.0  # a section after a gap that opens this far below its median bar...
ENTRY_DB = 6.0  # ...comes in at its first bar within this of the median
SPILL_DB = 6.0  # the next section's first bars this close to the late section's level...
SPILL_STEP_DB = 3.0  # ...and this much louder than the rest of it carry the late section
SPLICE_S = 0.03  # equal-power crossfade where moved music meets the original


@dataclass(frozen=True)
class EntryFix:
    """Music after a gap that came in `bars` bars late, and what the mix does about it:
    "shift" takes the section (and its spill into the next one) from `bars` bars later;
    "fill" replaces the quiet opening bars with the bars after them."""

    section_id: str
    bars: int
    mode: Literal["shift", "fill"]
    start: int  # samples: the moved stretch, taken from `offset` samples later
    end: int
    offset: int


def bar_levels_db(music: np.ndarray, start: int, bars: int, bar: float) -> list[float]:
    levels = []
    for i in range(bars):
        block = music[start + round(i * bar) : start + round((i + 1) * bar)]
        rms = float(np.sqrt(np.mean(np.square(block)))) if len(block) else 0.0
        levels.append(20 * math.log10(rms) if rms > 0 else -120.0)
    return levels


def late_entries(
    music: np.ndarray, arrangement: Arrangement, sr: int, max_bars: int
) -> list[EntryFix]:
    """Sections right after a silent one that open near-silent and reach their full
    level within `max_bars` bars: generated drops often open with a silent bar and a
    riser, which after the mix's own gap leaves seconds of near-silence. Soft (not
    near-silent) or longer openings are left alone as deliberate."""
    bar = bar_seconds(arrangement.bpm) * sr
    spans = section_spans(arrangement, sr)
    fixes = []
    for i, (section, start, end) in enumerate(spans):
        if i == 0 or not spans[i - 1][0].silent or section.silent or max_bars <= 0:
            continue
        levels = bar_levels_db(music, start, section.bars, bar)
        reference = float(np.median(levels))
        if not levels or levels[0] >= reference - LATE_ENTRY_DB:
            continue
        late = next(n for n, level in enumerate(levels) if level >= reference - ENTRY_DB)
        if late > max_bars or 2 * late > section.bars:
            continue
        offset = round(late * bar)
        following = spans[i + 1] if i + 1 < len(spans) else None
        spill = False
        if following is not None and following[0].bars >= 2 * late:
            after = bar_levels_db(music, end, following[0].bars, bar)
            rest = float(np.median(after[late:]))
            spill = min(after[:late]) >= max(reference - SPILL_DB, rest + SPILL_STEP_DB)
        stop = end + offset if spill else start + offset
        if stop + offset + round(SPLICE_S * sr) > len(music):
            continue
        fixes.append(EntryFix(section.id, late, "shift" if spill else "fill", start, stop, offset))
    return fixes


def apply_entry_fix(music: np.ndarray, fix: EntryFix, sr: int) -> np.ndarray:
    """The music between `fix.start` and `fix.end`, taken `fix.offset` samples later,
    crossfaded back into the original at the end."""
    out = music.copy()
    out[fix.start : fix.end] = music[fix.start + fix.offset : fix.end + fix.offset]
    fade = round(SPLICE_S * sr)
    t = np.linspace(0, np.pi / 2, fade, dtype=np.float32)[:, None]
    moved = music[fix.end + fix.offset : fix.end + fix.offset + fade]
    out[fix.end : fix.end + fade] = moved * np.cos(t) + music[fix.end : fix.end + fade] * np.sin(t)
    return out


def section_spans(arrangement: Arrangement, sr: int) -> list[tuple[Section, int, int]]:
    """Each section's first and end sample."""
    bar = bar_seconds(arrangement.bpm)
    return [(s, round(s.start_bar * bar * sr), round((s.start_bar + s.bars) * bar * sr))
            for s in arrangement.sections]  # fmt: skip


def energy_gains(
    levels: Sequence[float | None],
    energies: Sequence[float],
    *,
    range_db: float,
    tolerance_db: float,
    max_db: float,
) -> tuple[list[float], list[float | None]]:
    """(gain dB, target level) per section. The targets lie on a line `range_db` steeper
    from energy 0 to 1, through the median section. Only the part of a deviation beyond
    `tolerance_db` is corrected, by at most `max_db`. Sections without a level (silent,
    or too short to measure) get no target and no gain."""
    offsets = [lv - range_db * e for lv, e in zip(levels, energies, strict=True) if lv is not None]
    if len(offsets) < 2 or max_db <= 0:
        return [0.0] * len(levels), [None] * len(levels)
    anchor = float(np.median(offsets))
    gains: list[float] = []
    targets: list[float | None] = []
    for level, energy in zip(levels, energies, strict=True):
        if level is None:
            gains.append(0.0)
            targets.append(None)
            continue
        target = anchor + range_db * energy
        excess = abs(target - level) - tolerance_db
        gains.append(math.copysign(min(excess, max_db), target - level) if excess > 0 else 0.0)
        targets.append(target)
    return gains, targets


def gain_curve(
    spans: Sequence[tuple[int, int]], gains_db: Sequence[float], frames: int, sr: int
) -> np.ndarray:
    """Per-sample gain: each section's gain, moving to the next one's over the last
    ENERGY_RAMP_S of the section (raised cosine), so downbeats get their own gain. The
    song's tail keeps the last section's gain."""
    curve = np.ones(frames, dtype=np.float32)
    gains = [db_to_gain(g) for g in gains_db]
    for (start, end), gain in zip(spans, gains, strict=True):
        curve[start:end] = gain
    if spans:
        curve[spans[-1][1] :] = gains[-1]
    ramp = round(ENERGY_RAMP_S * sr)
    for i in range(1, len(spans)):
        before, after = gains[i - 1], gains[i]
        end = min(spans[i][0], frames)
        start = max(spans[i - 1][0], end - ramp)
        if before == after or end <= start:
            continue
        shape = 0.5 - 0.5 * np.cos(np.pi * np.arange(end - start) / (end - start))
        curve[start:end] = before + (after - before) * shape
    return curve


def gap_mask(gaps: Sequence[tuple[int, int]], frames: int, sr: int) -> np.ndarray:
    """1, except 0 inside each gap: a GAP_CUT_S fade at its start, and a GAP_RETURN_S
    fade back that ends where the next section begins."""
    mask = np.ones(frames, dtype=np.float32)
    for start, end in gaps:
        end = min(end, frames)
        if end <= start:
            continue
        mask[start:end] = 0.0
        cut = min(round(GAP_CUT_S * sr), end - start)
        mask[start : start + cut] = np.linspace(1, 0, cut, endpoint=False)
        back = min(round(GAP_RETURN_S * sr), end - start - cut)
        if back > 0:
            mask[end - back : end] = np.linspace(0, 1, back, endpoint=False)
    return mask


def gap_tail(music: np.ndarray, start: int, length: int, sr: int, level: float) -> np.ndarray:
    """What a reverb would ring on with after the music stops at `start`: `length`
    samples (stereo), fading in over the cut and out to silence by the end."""
    out = np.zeros((length, 2), dtype=np.float32)
    source = min(start, round(GAP_SOURCE_S * sr))
    if level <= 0 or length <= 0 or source <= 0:
        return out
    import pedalboard

    feed = stereo(music[start - source : start]).astype(np.float32)
    lead = min(source, round(0.05 * sr))
    feed[:lead] *= np.linspace(0, 1, lead, dtype=np.float32)[:, None]
    padded = np.concatenate([feed, np.zeros((length, 2), dtype=np.float32)])
    room = pedalboard.Reverb(room_size=GAP_ROOM, damping=0.4, wet_level=1.0, dry_level=0.0,
                             width=1.0)  # fmt: skip
    wet = room(padded.T, sr).T[source:]
    envelope = np.square(np.linspace(1, 0, length, dtype=np.float32))
    cut = min(length, round(GAP_CUT_S * sr))
    envelope[:cut] *= np.linspace(0, 1, cut, dtype=np.float32)
    return (wet * envelope[:, None] * np.float32(level)).astype(np.float32)


# --- Melody layer ------------------------------------------------------------------------------


@dataclass(frozen=True)
class NoteEvent:
    start_s: float
    end_s: float
    midi: int
    velocity: int  # 0-100, from the melody


def _phrase_notes(phrase: ClipMelody, start_beat: float, stop_beat: float, bpm: float,
                  loop: bool) -> list[NoteEvent]:  # fmt: skip
    """The phrase's notes from `start_beat`, looped (or once) and cut at `stop_beat`."""
    beat = beat_seconds(bpm)
    length = phrase.bars * BEATS_PER_BAR
    loops = max(1, math.ceil((stop_beat - start_beat) / length)) if loop else 1
    events = []
    for n in range(loops):
        for note in phrase.notes:
            on = start_beat + n * length + note.start_beat
            if on >= stop_beat:
                continue
            off = min(on + note.beats, stop_beat)
            events.append(NoteEvent(on * beat, off * beat, note.midi, note.velocity))
    return events


def melody_notes(
    arrangement: Arrangement, melody: Melody, mode: Literal["replay", "all", "off"]
) -> tuple[list[NoteEvent], list[NoteEvent]]:
    """(replayed, under speech) notes. "replay": sections with a `melody_phrase` loop
    that clip's phrase. "all": also each clip's own phrase under its speech, once."""
    if mode == "off":
        return [], []
    phrases = {p.clip_id: p for p in melody.clips}
    bpm = arrangement.bpm
    replay: list[NoteEvent] = []
    under: list[NoteEvent] = []
    for section in arrangement.sections:
        start = section.start_bar * BEATS_PER_BAR
        stop = start + section.bars * BEATS_PER_BAR
        phrase = phrases.get(section.melody_phrase or "")
        if phrase is not None:
            replay += _phrase_notes(phrase, start, stop, bpm, loop=True)
        own = phrases.get(section.clip_id or "")
        if mode == "all" and own is not None:
            clip_beat = start + section.clip_offset_beats
            under += _phrase_notes(own, clip_beat, stop, bpm, loop=False)
    return replay, under
