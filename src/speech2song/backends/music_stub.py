"""The stub music backend: a placeholder track synthesized locally, for free.

It follows the arrangement's tempo, chords and energy closely enough to develop and test
the mixer: chord pads that brighten with energy, a sub bass on chord roots, half-time
drums when the energy is high, a noise riser in rising sections, near silence in gaps.
It is deliberately simple; it is not meant to sound finished. Takes differ by seed (pad
voicing, drum fills, riser texture).
"""

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from speech2song.arrangement import (
    BEATS_PER_BAR,
    SONG_TAIL_S,
    beat_seconds,
    energy_bounds,
    music_bars,
    music_total_bars,
)
from speech2song.audio.synth import write_wav
from speech2song.audio.theory import parse_chord
from speech2song.backends.music_base import (
    MelodyFiles,
    Take,
    next_take_number,
    remove_stub_takes,
)
from speech2song.config import SAMPLE_RATE, Preset
from speech2song.costs import SpendEstimate
from speech2song.llm.claude import request_digest
from speech2song.manifest import write_json
from speech2song.models import Arrangement, TakeMeta
from speech2song.music_plan import grid_ms, tail_ms

STUB_VERSION = 1  # bump when the sound changes
BASE_SEED = 1234
PEAK_DBFS = -3.0
PAD_LOW = 55  # pad voices sit in G3-F#4 (MIDI)
BASS_LOW = 31  # bass roots sit in G1-F#2
PAD_HARMONICS = 8
TABLE_SIZE = 2048  # samples in one wavetable cycle
EDGE_S = 0.25  # pad crossfade between chords
TWO_PI = 2 * np.pi


def _midi_hz(note: float) -> float:
    return 440.0 * 2 ** ((note - 69) / 12)


def _place(pitch_class: int, low: int) -> int:
    return low + (pitch_class - low) % 12


def energy_curve(request: dict[str, Any], frames: int, sr: int) -> np.ndarray:
    """Per-sample energy (0-1): linear within each section, from its start to end value."""
    curve = np.zeros(frames, dtype=np.float32)
    bar = BEATS_PER_BAR * beat_seconds(request["bpm"])
    for section in request["sections"]:
        a = round(section["start_bar"] * bar * sr)
        b = min(frames, round((section["start_bar"] + section["bars"]) * bar * sr))
        if b > a:
            curve[a:b] = np.linspace(section["energy_start"], section["energy_end"], b - a)
    return curve


def _pad_chord(
    pitch_classes: tuple[int, ...],
    seconds: float,
    sr: int,
    brightness: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Stereo pad: chord tones plus a root an octave down, slightly detuned left and
    right, with stronger harmonics when brighter (one wavetable per voice). Fades in
    and out over EDGE_S."""
    n = round((seconds + EDGE_S) * sr)
    t = np.arange(n) / sr
    out = np.zeros((n, 2), dtype=np.float64)
    voices = [_place(pc, PAD_LOW) for pc in pitch_classes]
    voices.append(_place(pitch_classes[0], PAD_LOW - 12))
    rolloff = 2.2 - 1.4 * brightness  # higher = darker
    cycle = np.arange(TABLE_SIZE) / TABLE_SIZE
    for voice in voices:
        f0 = _midi_hz(voice)
        harmonics = [h for h in range(1, PAD_HARMONICS + 1) if f0 * h < sr / 2.5]
        table = sum(np.sin(TWO_PI * h * cycle) / h**rolloff for h in harmonics)
        for channel, cents in enumerate((-4.0, 4.0)):
            phase = (f0 * 2 ** (cents / 1200) * t + rng.uniform()) % 1.0
            out[:, channel] += np.interp(phase, cycle, table, period=1.0)
    edge = min(round(EDGE_S * sr), n // 2)
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(edge) / edge)
    out[:edge] *= ramp[:, None]
    out[n - edge :] *= ramp[::-1, None]
    return out / len(voices)


def _kick(sr: int) -> np.ndarray:
    n = round(0.35 * sr)
    t = np.arange(n) / sr
    freq = 45 + 75 * np.exp(-t / 0.04)
    return np.sin(TWO_PI * np.cumsum(freq) / sr) * np.exp(-t / 0.12)


def _snare(sr: int, rng: np.random.Generator) -> np.ndarray:
    n = round(0.25 * sr)
    t = np.arange(n) / sr
    noise = np.diff(rng.standard_normal(n + 1)) * 0.5
    return (0.7 * noise + 0.5 * np.sin(TWO_PI * 185 * t)) * np.exp(-t / 0.07)


def _hat(sr: int, rng: np.random.Generator) -> np.ndarray:
    n = round(0.06 * sr)
    t = np.arange(n) / sr
    noise = np.diff(np.diff(rng.standard_normal(n + 2))) * 0.25
    return noise * np.exp(-t / 0.015)


def _add(out: np.ndarray, sound: np.ndarray, start: int, gain: float, pan: float = 0.0) -> None:
    end = min(len(out), start + len(sound))
    if end <= start or start < 0:
        return
    left, right = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
    out[start:end, 0] += gain * left * math.sqrt(2) * sound[: end - start]
    out[start:end, 1] += gain * right * math.sqrt(2) * sound[: end - start]


def synthesize(request: dict[str, Any], seed: int, sr: int = SAMPLE_RATE) -> np.ndarray:
    """The placeholder track for `request` (see StubBackend.request), stereo float32."""
    import pedalboard

    rng = np.random.default_rng(seed)
    beat = beat_seconds(request["bpm"])
    bar = BEATS_PER_BAR * beat
    frames = round((request["total_bars"] * bar + max(SONG_TAIL_S, request.get("tail_s", 0))) * sr)
    energy = energy_curve(request, frames, sr)
    pads = np.zeros((frames, 2))
    bass = np.zeros(frames)
    drums = np.zeros((frames, 2))
    riser = np.zeros((frames, 2))
    kick, snare, hat = _kick(sr), _snare(sr, rng), _hat(sr, rng)
    fills = max(1, request["total_bars"] // 12)  # bars that end with a snare pickup
    fill_bars = {int(b) for b in rng.choice(request["total_bars"], size=fills)}

    for section in request["sections"]:
        start_bar, bars = section["start_bar"], section["bars"]
        for i in range(bars):
            chord = parse_chord(section["chords"][i % len(section["chords"])])
            if rng.random() < 0.5:  # take-to-take variety: inversions
                chord = (*chord[1:], chord[0])
            a = round((start_bar + i) * bar * sr)
            level = float(energy[min(a, frames - 1)])
            if level <= 0:
                continue
            tone = _pad_chord(chord, bar, sr, brightness=level, rng=rng)
            end = min(frames, a + len(tone))
            pads[a:end] += tone[: end - a]
            if level >= 0.5:  # sub bass with an eighth-note wobble
                n = min(frames - a, round(bar * sr))
                t = np.arange(n) / sr
                f = _midi_hz(_place(chord[0], BASS_LOW))
                wobble = 0.65 + 0.35 * np.cos(TWO_PI * t / (beat / 2))
                bass[a : a + n] += np.sin(TWO_PI * f * t) * wobble * np.minimum(1, t / 0.02)
            for step in range(8):  # eighth notes; half-time: kick on 1, snare on 3
                at = a + round(step * beat / 2 * sr)
                fill = (start_bar + i) in fill_bars and step >= 6
                if level >= 0.5 and (step == 0 or (level >= 0.8 and step == 5)):
                    _add(drums, kick, at, 0.9)
                if level >= 0.6 and (step == 4 or (fill and step == 7)):
                    _add(drums, snare, at, 0.5)
                if level >= 0.8:
                    _add(drums, hat, at, 0.25 if step % 2 else 0.15, pan=0.3)
        if section["shape"] == "rise":
            a = round(start_bar * bar * sr)
            n = min(frames - a, round(bars * bar * sr))
            noise = np.diff(rng.standard_normal((n + 1, 2)), axis=0) * 0.35
            ramp = np.linspace(0, 1, n) ** 2
            riser[a : a + n] += noise * ramp[:, None]

    level = 0.25 + 0.75 * energy
    pads *= (level * (energy > 0))[:, None]
    space = pedalboard.Pedalboard([pedalboard.Reverb(room_size=0.8, wet_level=0.35, dry_level=0.8)])
    pads = space(pads.T.astype(np.float32), sr).T
    mix = 0.6 * pads + 0.5 * bass[:, None] + drums + 0.4 * riser
    peak = float(np.abs(mix).max()) or 1.0
    return (mix * (10 ** (PEAK_DBFS / 20) / peak)).astype(np.float32)


class StubBackend:
    name = "stub"
    paid = False

    def request(
        self, arrangement: Arrangement, preset: Preset, melody: MelodyFiles
    ) -> dict[str, Any]:
        """Sections in music time: a passage played alone has no music (the take stage
        splices it open)."""
        bounds = energy_bounds(arrangement.sections)
        starts = music_bars(arrangement)
        request = {
            "backend": self.name,
            "version": STUB_VERSION,
            "bpm": arrangement.bpm,
            "key": arrangement.key,
            "total_bars": music_total_bars(arrangement),
            "grid_ms": grid_ms(arrangement),
            "sections": [
                {"role": s.role, "start_bar": start, "bars": s.bars, "shape": s.shape,
                 "energy_start": a, "energy_end": b, "chords": s.chords}
                for s, (a, b), start in zip(arrangement.sections, bounds, starts, strict=True)
                if not s.rest
            ],
        }  # fmt: skip
        if tail_ms(arrangement):  # music past the last bar, as the paid plan has
            request["tail_s"] = tail_ms(arrangement) / 1000
        return request

    def estimate(self, request: dict[str, Any], takes: int) -> list[SpendEstimate]:
        return []

    def generate(
        self,
        request: dict[str, Any],
        out_dir: Path,
        takes: int,
        run_dir: Path,
        say: Callable[[str], None] = print,
        fresh: bool = False,
    ) -> list[Take]:
        out_dir.mkdir(parents=True, exist_ok=True)
        digest = request_digest(request)
        remove_stub_takes(out_dir, run_dir)
        first = next_take_number(out_dir)
        results = []
        for number in range(first, first + takes):
            seed = BASE_SEED + number
            audio = synthesize(request, seed)
            path = out_dir / f"take_{number:03d}.wav"
            write_wav(path, audio, SAMPLE_RATE)
            meta = TakeMeta(
                take=number,
                backend=self.name,
                file=path.relative_to(run_dir).as_posix(),
                sample_rate=SAMPLE_RATE,
                channels=2,
                seconds=round(len(audio) / SAMPLE_RATE, 4),
                seed=seed,
                request_sha256=digest,
                grid_ms=request["grid_ms"],
            )
            write_json(path.with_suffix(".meta.json"), meta)
            results.append(Take(path, meta))
        return results
