"""Mixing logic on arrays: where clips go, the exact speech stem, the processed speech
bus, the music's shaping, and the melody layer's notes. No file I/O.

The speech stem is the clips copied verbatim onto silence (the exactness invariant).
Everything else (gain, high-pass, compression, reverb and delay sends) happens on the
speech bus, which only the mix hears. Each clip's sends are rendered on their own and
faded out before the next clip starts, so tails never run into the next line.

The music is shaped toward the arrangement: sections whose level strays far from their
energy are pulled back toward it, and gaps (the silent sections, with no music of their
own) become a lift into the next section: the build's tail rising, with a reverse swell
of it. A passage played alone keeps its quiet bed until its last phrase: the music stops
just before that phrase and its reverb dies away, and the music returns a beat or two
after the last word through a reverse swell. (M7 arrangements had no music at all
under such a passage: a stop on the bar line, back on the next downbeat.) Under a speech
passage the music is set once, at the passage's bar lines, never following the voice.
The last chord rings out through a long reverb when the music would otherwise stop
abruptly.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from speech2song.arrangement import (
    BEATS_PER_BAR,
    bar_seconds,
    beat_seconds,
    clip_start_s,
    passages,
)
from speech2song.audio.dsp import (
    band_filter,
    channel_levels_db,
    db_to_gain,
    loudness_lufs,
    word_level_db,
)
from speech2song.models import (
    Arrangement,
    Clip,
    ClipMelody,
    Melody,
    PlacedClip,
    Section,
    Word,
)

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


WORD_TAIL_S = 0.15  # ASR word ends come early: speech and room tone ring on after them


@dataclass(frozen=True)
class WordSpan:
    clip_id: str
    section_id: str
    word: str
    start: int  # mix samples
    end: int


def word_spans(
    placed: Sequence[PlacedClip], clips: dict[str, Clip], words: Sequence[Word], sr: int
) -> list[WordSpan]:
    """Where each spoken word of the placed clips lands in the mix (transcript times,
    extended by WORD_TAIL_S and kept inside the clip). A word belongs to the clip that
    holds its midpoint: the next sentence's first word, which a cut can graze, does not."""
    spans = []
    for p in placed:
        clip = clips[p.clip_id]
        for word in words:
            if clip.start_s <= (word.start + word.end) / 2 < clip.end_s:
                start = p.start_sample + round((word.start - clip.start_s) * sr)
                end = p.start_sample + round((word.end + WORD_TAIL_S - clip.start_s) * sr)
                spans.append(WordSpan(p.clip_id, p.section_id, word.w, max(start, p.start_sample),
                                      min(end, p.end_sample)))  # fmt: skip
    return spans


PASSAGE_RAMP_S = 0.5  # the passage level eases in and out over this long
PASSAGE_WORD_GAP_S = 0.1  # ...and is in place this long before the first word


@dataclass(frozen=True)
class PassageLevel:
    """The music under one speech passage: where the passage runs (samples, bar line to
    bar line), the words' span inside it, how far the music sat under the speech there
    (speech band) and the gain the mix sets for it."""

    sections: list[str]
    start: int
    end: int
    words: tuple[int, int]
    margin_db: float
    gain_db: float


def passage_levels(
    music: np.ndarray, speech: np.ndarray, arrangement: Arrangement, words: Sequence[WordSpan],
    sr: int, *, margin_db: float, max_cut_db: float,
) -> list[PassageLevel]:  # fmt: skip
    """One level per passage with music beneath it: the music comes down only as far as
    it takes to sit `margin_db` under the speech over the passage's words (speech band,
    stereo energies averaged), and at most `max_cut_db` (negative). A quiet bed is left
    alone."""
    spoken = heard = None
    levels = []
    for run in passages(arrangement):
        if run[0].rest:
            continue
        ids = {section.id for section in run}
        mine = [w for w in words if w.section_id in ids]
        if not mine:
            continue
        if spoken is None:
            spoken, heard = band_filter(speech, sr), band_filter(music, sr)
        a, b = min(w.start for w in mine), max(w.end for w in mine)
        speech_db = word_level_db(spoken, a, b, sr)
        music_db = float(channel_levels_db(heard, np.array([(a + b) // 2]), max(b - a, 1))[0])
        margin = speech_db - music_db
        gain = -min(max(0.0, margin_db - margin), -max_cut_db)
        bar = bar_seconds(arrangement.bpm) * sr
        start = round(run[0].start_bar * bar)
        end = round((run[-1].start_bar + run[-1].bars) * bar)
        levels.append(PassageLevel([s.id for s in run], start, end, (a, b), round(margin, 2),
                                   round(gain, 2)))  # fmt: skip
    return levels


def passage_curve(levels: Sequence[PassageLevel], frames: int, sr: int) -> np.ndarray:
    """Per-sample linear gain: each passage's gain from its bar line to the next, easing
    in over PASSAGE_RAMP_S right after the passage starts (in place before its first
    word) and out over PASSAGE_RAMP_S before it ends (after its last word). Ramps are
    raised cosines in dB."""
    gain_db = np.zeros(frames, dtype=np.float32)
    ramp, gap = round(PASSAGE_RAMP_S * sr), round(PASSAGE_WORD_GAP_S * sr)
    for level in levels:
        if level.gain_db >= 0:
            continue
        down_end = min(level.start + ramp, level.words[0] - gap)
        up_start = max(level.end - ramp, level.words[1] + gap)
        a, b = max(0, down_end - ramp), min(frames, up_start + ramp)
        if b <= a:
            continue
        shape = np.ones(b - a, dtype=np.float32)
        t = np.arange(a, b)
        down = t < down_end
        shape[down] = 0.5 - 0.5 * np.cos(np.pi * (t[down] - (down_end - ramp)) / ramp)
        up = t >= up_start
        shape[up] = 0.5 + 0.5 * np.cos(np.pi * (t[up] - up_start) / ramp)
        gain_db[a:b] = np.minimum(gain_db[a:b], level.gain_db * shape)
    return (10 ** (gain_db / 20)).astype(np.float32)


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
GAP_SOURCE_S = 1.0  # how much of the build's tail feeds a gap's swell
GAP_ROOM = 0.9
GAP_LEAD_S = 0.05  # the swell's source fades in over this long
GAP_END_FADE_S = 0.005  # the swell's last samples fade out, so it never clicks
RING_SOURCE_S = 1.5  # how much of the last audible music feeds the ring-out
RING_ROOM = 1.0  # the longest reverb: about -8 dB per second after the first second
RING_ROOM_DAMPING = 0.1
RING_HOLD = 0.4  # the ring keeps its natural decay for this fraction of its length, then tapers
RING_WINDOW_S = 0.25  # level frames for finding where the music stops being audible
RING_WITHIN_DB = 12.0  # "audible": within this of the last section's loudest frame
RING_HANDOVER_S = 0.4  # the generated music fades out under the ring over this long


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
    """Sections right after a gap (or a passage played alone) that open near-silent and
    reach their full level within `max_bars` bars: generated drops often open with a
    silent bar and a riser, which after the mix's own gap leaves seconds of
    near-silence. Soft (not near-silent) or longer openings are left alone as
    deliberate."""
    bar = bar_seconds(arrangement.bpm) * sr
    spans = section_spans(arrangement, sr)
    fixes = []
    for i, (section, start, end) in enumerate(spans):
        before = spans[i - 1][0] if i else None
        if before is None or not (before.silent or before.rest) or section.silent \
                or section.rest or max_bars <= 0:  # fmt: skip
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
    max_boost_db: float | None = None,
) -> tuple[list[float], list[float | None]]:
    """(gain dB, target level) per section. The targets lie on a line `range_db` steeper
    from energy 0 to 1, through the median section. Only the part of a deviation beyond
    `tolerance_db` is corrected, by at most `max_db` (`max_boost_db` upward, if given).
    Sections without a level (silent, or too short to measure) get no target and no
    gain."""
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
        limit = max_boost_db if target > level and max_boost_db is not None else max_db
        gains.append(math.copysign(min(excess, limit), target - level) if excess > 0 else 0.0)
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


def _rms(block: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(block)))) if len(block) else 0.0


def reverb_wash(
    music: np.ndarray, start: int, length: int, sr: int, *, source_s: float, room: float,
    damping: float = 0.4,
) -> np.ndarray:  # fmt: skip
    """What a reverb rings on with after `music` stops at `start`: `length` samples
    (stereo) of the room's response to the `source_s` of music before `start`, at the
    level the room returns it (not rescaled)."""
    out = np.zeros((max(length, 0), 2), dtype=np.float32)
    source = min(start, round(source_s * sr))
    if length <= 0 or source <= 0:
        return out
    import pedalboard

    feed = stereo(music[start - source : start]).astype(np.float32)
    lead = min(source, round(GAP_LEAD_S * sr))
    feed[:lead] *= np.linspace(0, 1, lead, dtype=np.float32)[:, None]
    padded = np.concatenate([feed, np.zeros((length, 2), dtype=np.float32)])
    reverb = pedalboard.Reverb(room_size=room, damping=damping, wet_level=1.0, dry_level=0.0,
                               width=1.0)  # fmt: skip
    return reverb(padded.T, sr).T[source:].astype(np.float32)


def gap_lift(
    music: np.ndarray, start: int, end: int, sr: int, *, swell: float, lift_db: float
) -> np.ndarray:
    """The music for a gap, `start` to `end`, as a lift into the next section. What the
    take has there (the build running on) rises by `lift_db` over the gap, and a reverse
    swell of the build's tail, at `swell` times that tail's level, grows to its peak on
    the downbeat. Returns the whole replacement block (stereo)."""
    end = min(end, len(music))
    block = stereo(music[start:end]).astype(np.float32)
    length = len(block)
    if length == 0:
        return block
    t = np.linspace(0, 1, length, dtype=np.float32)
    rise = 0.5 - 0.5 * np.cos(np.pi * t)
    block *= np.float32(10 ** (lift_db / 20)) ** rise[:, None]
    source = min(start, round(GAP_SOURCE_S * sr))
    if swell <= 0 or source <= 0:
        return block
    wash = reverb_wash(music, start, length, sr, source_s=GAP_SOURCE_S, room=GAP_ROOM)
    near = _rms(wash[: round(0.25 * sr)])
    if near <= 0:
        return block
    level = _rms(stereo(music[start - source : start])) / near * swell
    envelope = np.sin(0.5 * np.pi * t) ** 2  # grows to the downbeat
    fade = min(length, round(GAP_END_FADE_S * sr))
    if fade:
        envelope[length - fade :] *= np.linspace(1, 0, fade, dtype=np.float32)
    return block + wash[::-1] * (envelope * np.float32(level))[:, None]


def audible_end(music: np.ndarray, sr: int, start: int, within_db: float) -> int | None:
    """The sample where the music, from `start` on, last is within `within_db` of its
    loudest 250 ms frame (the end of the frame); None when it is silent."""
    window = max(1, round(RING_WINDOW_S * sr))
    count = (len(music) - start) // window
    if count <= 0:
        return None
    frames = music[start : start + count * window].reshape(count, window, -1)
    levels = 10 * np.log10(np.mean(np.square(frames), axis=(1, 2)) + 1e-12)
    loud = np.flatnonzero(levels >= levels.max() - within_db)
    if levels.max() < -80:
        return None
    return start + int(loud[-1] + 1) * window


def ring_tail(
    music: np.ndarray, sr: int, stop: int, seconds: float, *, hold: float
) -> np.ndarray | None:
    """The reverb ring of the RING_SOURCE_S of music before `stop`, at the level that
    music was heard at, `seconds` long: it keeps its natural decay for `hold` of its
    length, then tapers to nothing. None when there is nothing to ring."""
    length = round(seconds * sr)
    if length <= 0 or stop <= 0:
        return None
    wash = reverb_wash(music, stop, length, sr, source_s=RING_SOURCE_S, room=RING_ROOM,
                       damping=RING_ROOM_DAMPING)  # fmt: skip
    near = _rms(wash[: round(0.5 * sr)])
    heard = _rms(stereo(music[max(0, stop - round(RING_SOURCE_S * sr)) : stop]))
    if near <= 0 or heard <= 0:
        return None
    t = np.linspace(0, 1, length, dtype=np.float32)
    taper = np.clip((t - hold) / (1 - hold), 0, 1)
    envelope = (0.5 + 0.5 * np.cos(np.pi * taper)) * np.minimum(1, t * seconds / 0.05)
    return wash * (envelope * np.float32(heard / near))[:, None]


def ring_out(
    music: np.ndarray, sr: int, start: int, seconds: float, hold: float = RING_HOLD
) -> tuple[np.ndarray, int]:
    """The last chord rings out. The music after the point where it stops being audible
    (see `audible_end`, searched from `start`, the last music section's first sample)
    fades out over RING_HANDOVER_S, and the last RING_SOURCE_S before that point go
    through a long reverb at the same level, which keeps its natural decay for `hold` of
    `seconds` and then dies away. Returns the music (longer, when the ring runs past its
    end) and the sample where the ring begins; the music is returned unchanged, with the
    end of the music, when `seconds` is 0 or it is silent."""
    stop = audible_end(music, sr, start, RING_WITHIN_DB) if seconds > 0 else None
    if stop is None:
        return music, len(music)
    ring = ring_tail(music, sr, stop, seconds, hold=hold)
    if ring is None:
        return music, len(music)
    out = np.zeros((max(len(music), stop + len(ring)), 2), dtype=np.float32)
    out[: len(music)] = music
    hand = min(len(music) - stop, round(RING_HANDOVER_S * sr))
    fade = np.zeros(len(out) - stop, dtype=np.float32)
    fade[:hand] = 0.5 + 0.5 * np.cos(np.pi * np.arange(hand) / max(hand, 1))
    out[stop:] *= fade[:, None]
    out[stop : stop + len(ring)] += ring
    return out, stop


DEAD_LU = 25.0  # a closing section this far under the music's loudness has died away


def live_start(
    music: np.ndarray, spans: Sequence[tuple[int, int]], sr: int, reference_lufs: float | None
) -> int:
    """Where the song's last audible music begins: the start of the last span (sections
    with music, in order) whose loudness is within DEAD_LU of `reference_lufs`. The
    music model tends to make closing sections near-silent; the ending is rung out of
    the last music that is actually heard. The last span's start when none qualifies."""
    if not spans:
        return 0
    for start, end in reversed(spans):
        level = loudness_lufs(music[start:end], sr)
        if level is not None and (reference_lufs is None or level >= reference_lufs - DEAD_LU):
            return start
    return spans[-1][0]


ABRUPT_DB = 20.0  # music still this close to its loudest in its last half second stops abruptly
ABRUPT_TAIL_S = 0.5
SETTLED_DB = 50.0  # the music has died away once it stays this far under its loudest


def ends_abruptly(music: np.ndarray, sr: int, start: int, end: int) -> bool:
    """Whether the music from `start` to `end` (the last music section) is still sounding
    in its last half second, within ABRUPT_DB of its loudest 250 ms: it would stop dead
    rather than die away."""
    window = max(1, round(RING_WINDOW_S * sr))
    count = (end - start) // window
    if count <= 0:
        return False
    frames = music[start : start + count * window].reshape(count, window, -1)
    levels = 10 * np.log10(np.mean(np.square(frames), axis=(1, 2)) + 1e-12)
    if levels.max() < -80:
        return False
    last = music[max(start, end - round(ABRUPT_TAIL_S * sr)) : end]
    tail = 10 * np.log10(np.mean(np.square(last)) + 1e-12)
    return bool(tail >= levels.max() - ABRUPT_DB)


def settled_at(music: np.ndarray, sr: int, start: int) -> int:
    """The sample after which the music (from `start`) stays SETTLED_DB under the whole
    song's loudest 250 ms: where a natural ending has died away. (Measured against the
    song, not the last section: a model that fades out early leaves a last section that
    is quiet throughout, and its "ending" would otherwise keep seconds of near-silence.)"""
    window = max(1, round(RING_WINDOW_S * sr))
    count = len(music) // window
    if count <= 0 or start >= len(music):
        return min(start, len(music))
    frames = music[: count * window].reshape(count, window, -1)
    levels = 10 * np.log10(np.mean(np.square(frames), axis=(1, 2)) + 1e-12)
    first = start // window
    loud = np.flatnonzero(levels[first:] >= levels.max() - SETTLED_DB)
    return start if len(loud) == 0 else (first + int(loud[-1]) + 1) * window


ALONE_RETURN_S = 1.5  # the longest reverse swell into the music's return
ALONE_RETURN_MIN_S = 0.3  # shorter room than this after the last word: no swell
ALONE_WORD_GAP_S = 0.15  # the swell starts at least this long after the last word
SWELL_SOURCE_S = 1.0  # how much of the returning music feeds its reverse swell


def reverse_swell(music: np.ndarray, at: int, length: int, sr: int) -> np.ndarray:
    """A reverse reverb of the music that starts at `at`: the music after `at`, reversed,
    through a reverb, reversed back, so its reverb grows toward the downbeat. `length`
    samples (stereo) ending at `at`, at the level the room returns (not rescaled)."""
    source = stereo(music[at : at + round(SWELL_SOURCE_S * sr)])[::-1]
    if length <= 0 or len(source) == 0:
        return np.zeros((max(length, 0), 2), dtype=np.float32)
    feed = np.concatenate([source, np.zeros((length, 2), dtype=np.float32)])
    return reverb_wash(feed, len(source), length, sr, source_s=SWELL_SOURCE_S,
                       room=GAP_ROOM)[::-1].astype(np.float32)  # fmt: skip


def _return_swell(
    out: np.ndarray, music: np.ndarray, back: int, last_word_end: int | None, floor: int,
    sr: int, swell: float,
) -> None:  # fmt: skip
    """Adds (in place) a reverse swell of the music that returns at `back`, growing into
    it from after the last word (or `floor`), at `swell` times that music's level."""
    room = back - (last_word_end if last_word_end is not None else floor) \
        - round(ALONE_WORD_GAP_S * sr)  # fmt: skip
    length = min(round(ALONE_RETURN_S * sr), room, back)
    after = stereo(music[back : back + round(SWELL_SOURCE_S * sr)])
    if swell <= 0 or length < round(ALONE_RETURN_MIN_S * sr) or _rms(after) <= 0:
        return
    wash = reverse_swell(music, back, length, sr)
    near = _rms(wash[-round(0.25 * sr) :])
    if near <= 0:
        return
    t = np.linspace(0, 1, length, dtype=np.float32)
    envelope = np.sin(0.5 * np.pi * t) ** 2
    fade = min(length, round(GAP_END_FADE_S * sr))
    envelope[length - fade :] *= np.linspace(1, 0, fade, dtype=np.float32)
    out[back - length : back] += wash * (envelope * np.float32(_rms(after) / near * swell))[:, None]


def alone_break(
    music: np.ndarray, start: int, end: int, last_word_end: int | None, sr: int, *,
    ring_s: float, swell: float,
) -> np.ndarray:  # fmt: skip
    """The music around a passage played alone in an M7 arrangement (`start` to `end`,
    bar lines; the take is already silent there). From `start`, the reverb of the music
    before it dies away over `ring_s` (tapering at once, so the speaker is soon alone);
    before `end`, if the passage leaves room after its last word, a reverse swell of the
    returning music grows into the downbeat at `swell` times that music's level. Returns
    the music."""
    out = music.copy()
    ring = ring_tail(music, sr, start, ring_s, hold=0.0) if ring_s > 0 else None
    if ring is not None:
        stop = min(len(out), start + len(ring))
        out[start:stop] += ring[: stop - start]
    _return_swell(out, music, end, last_word_end, start, sr, swell)
    return out


PHRASE_MIN_S = 2.0  # a passage's last phrase, heard alone, is at least this long...
PHRASE_MAX_S = 6.0  # ...and at most this long (where its words allow)
PHRASE_ENDS = (",", ";", ":", ".", "!", "?", "\u2014", "\u2013", "\u2026")  # dashes, ellipsis
CLOSING_MARKS = "\"')\u201d\u2019\u00bb"  # quotes and brackets after the mark
BREAK_LEAD_S = 0.12  # the music stops this long before the last phrase's first word...
BREAK_FADE_S = 0.08  # ...fading out over this long
BREATH_BEATS = 1.0  # it returns on the first beat at least this long after the last word...
RETURN_MAX_BEATS = 2.0  # ...or on the next section's downbeat, if that comes within this
RETURN_FADE_S = 0.003


def _ends_phrase(word: str) -> bool:
    return word.rstrip(CLOSING_MARKS).endswith(PHRASE_ENDS)


def last_phrase(words: Sequence[WordSpan], sr: int) -> int | None:
    """Index into `words` (a passage's words, in order) of the first word of its last
    phrase: the latest punctuation mark (or the start of the passage's next clip) that
    leaves PHRASE_MIN_S to PHRASE_MAX_S of speech after it; failing that, the longest
    pause in that range. None when the passage is too short to split."""
    if len(words) < 2:
        return None
    end = words[-1].end
    fits = [i for i in range(1, len(words))
            if PHRASE_MIN_S <= (end - words[i].start) / sr <= PHRASE_MAX_S]  # fmt: skip
    marked = [i for i in fits if _ends_phrase(words[i - 1].word)
              or words[i].clip_id != words[i - 1].clip_id]  # fmt: skip
    if marked:
        return marked[-1]
    if fits:
        return max(fits, key=lambda i: (words[i].start - words[i - 1].end, i))
    return None


def break_stop(words: Sequence[WordSpan], phrase: int | None, floor: int, sr: int) -> int:
    """Where the music stops for a passage's last phrase (`phrase`, from `last_phrase`):
    BREAK_LEAD_S before its first word, in the pause before it where there is one (never
    before `floor`, the passage's first sample). Without a phrase, before the first word."""
    first = words[phrase or 0].start
    stop = first - round(BREAK_LEAD_S * sr)
    if phrase:
        stop = max(stop, min(words[phrase - 1].end, first))
    return max(floor, stop)


def return_point(last_word_end: int, passage_end: int, bpm: float, sr: int) -> int:
    """Where the music comes back after a passage's last phrase: on the next section's
    downbeat (`passage_end`) when that comes within RETURN_MAX_BEATS of the last word,
    else on the first beat at least BREATH_BEATS after it, so the silence after the
    words lasts a beat or two."""
    beat = beat_seconds(bpm) * sr
    if passage_end - last_word_end <= RETURN_MAX_BEATS * beat:
        return passage_end
    return min(passage_end, round(math.ceil((last_word_end + BREATH_BEATS * beat) / beat
                                            - 1e-9) * beat))  # fmt: skip


def cut_music(music: np.ndarray, stop: int, back: int | None, sr: int) -> np.ndarray:
    """The music silenced from `stop` (fading out over BREAK_FADE_S before it) to `back`
    (where it returns, with a 3 ms fade in), or to the end when `back` is None."""
    out = music.copy()
    fade = min(round(BREAK_FADE_S * sr), stop)
    out[stop - fade : stop] *= np.linspace(1, 0, fade, dtype=np.float32)[:, None]
    end = len(out) if back is None else min(back, len(out))
    out[stop:end] = 0
    if back is not None:
        rise = min(round(RETURN_FADE_S * sr), len(out) - back)
        out[back : back + rise] *= np.linspace(0, 1, rise, dtype=np.float32)[:, None]
    return out


def phrase_break(
    music: np.ndarray, stop: int, back: int, last_word_end: int, sr: int, *, ring_s: float,
    swell: float,
) -> np.ndarray:  # fmt: skip
    """A passage's last phrase lands alone: the music stops at `stop` and its reverb dies
    away over `ring_s` (tapering at once), and it comes back at `back` with a reverse
    swell growing into it from after the last word (at `swell` times its level). Returns
    the music."""
    out = cut_music(music, stop, back, sr)
    ring = ring_tail(music, sr, stop, ring_s, hold=0.0) if ring_s > 0 else None
    if ring is not None:
        n = min(len(ring), len(out) - stop)
        out[stop : stop + n] += ring[:n]
    _return_swell(out, music, back, last_word_end, stop, sr, swell)
    return out


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
