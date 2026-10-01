"""Mixing logic on arrays: where clips go, the exact speech stem, the processed speech
bus, and the melody layer's notes. No file I/O.

The speech stem is the clips copied verbatim onto silence (the exactness invariant).
Everything else (gain, high-pass, compression, reverb and delay sends) happens on the
speech bus, which only the mix hears. Each clip's sends are rendered on their own and
faded out before the next clip starts, so tails never run into the next line.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from speech2song.arrangement import BEATS_PER_BAR, beat_seconds, clip_start_s
from speech2song.models import Arrangement, ClipMelody, Melody, PlacedClip

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
