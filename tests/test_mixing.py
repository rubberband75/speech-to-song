"""Mixing on arrays: clip placement, the exact speech stem, tails, the melody layer."""

import numpy as np
import pytest

from speech2song.arrangement import ClipInfo, build_arrangement, default_parts
from speech2song.audio.mixing import (
    TAIL_MAX_S,
    melody_notes,
    place_clips,
    speech_bus,
    speech_stem,
    tail_limits,
)
from speech2song.config import load_preset
from speech2song.models import Arrangement, BarChord, ClipMelody, Melody, MelodyNote, PlacedClip

from .conftest import PRESETS_DIR
from .fixtures.synth import SR, noise_burst

PRESET = load_preset("cinematic_future_bass", PRESETS_DIR)
BPM = 120.0  # a beat is 0.5 s, a bar 2 s


def _phrase(cid: str, bars: int = 1) -> ClipMelody:
    notes = [
        MelodyNote(start_beat=b, beats=1.0, midi=60 + b, pitch=60 + b, speech_pitch=48,
                   start_s=b / 2, end_s=b / 2 + 0.4, velocity=80)
        for b in range(3)
    ]  # fmt: skip
    chord = BarChord(bar=0, degree=1, name="Am", pitch_classes=[9, 0, 4])
    return ClipMelody(clip_id=cid, file=f"clips/{cid}.wav", duration_s=1.5, bars=bars,
                      voiced_ratio=0.5, median_speech_pitch=48, notes=notes,
                      chords=[chord])  # fmt: skip


def _setup() -> tuple[Arrangement, Melody, dict[str, np.ndarray]]:
    clips = [ClipInfo("a", "hook", "a", 1.5, 1.2), ClipInfo("b", "build", "b", 1.5, 1.2),
             ClipInfo("c", "outro", "c", 1.5, 1.2)]  # fmt: skip
    melody = Melody(key="A minor", tonic=9, mode="minor", key_source="detected",
                    tuning_offset=0, bpm=BPM, tempo_cost=0, grid="1/8", snap_strength=0.8,
                    octave_shift=0, loop_phrase_count=3, instrument="soft_piano",
                    main_clip="a", clips=[_phrase(c.id) for c in clips])  # fmt: skip
    arrangement = build_arrangement(default_parts(PRESET, clips), clips, melody, PRESET)
    audio = {c.id: np.stack([noise_burst(1.5, seed=i), noise_burst(1.5, seed=i + 9)], axis=1)
             for i, c in enumerate(clips)}  # fmt: skip
    return arrangement, melody, audio


def _placed(arrangement: Arrangement, audio: dict[str, np.ndarray]) -> list[PlacedClip]:
    return place_clips(arrangement, {k: f"clips/{k}.wav" for k in audio},
                       {k: len(v) for k, v in audio.items()}, SR)  # fmt: skip


def test_clips_land_on_their_section_starts() -> None:
    arrangement, _, audio = _setup()
    placed = _placed(arrangement, audio)
    beds = [s for s in arrangement.sections if s.clip_id]
    assert [p.clip_id for p in placed] == ["a", "b", "c"]
    for p, bed in zip(placed, beds, strict=True):
        assert p.start_sample == round(bed.start_bar * 2.0 * SR)
        assert p.end_sample - p.start_sample == len(audio[p.clip_id])
    shifted = arrangement.model_copy(deep=True)
    next(s for s in shifted.sections if s.clip_id == "a").clip_offset_beats = 1.0
    assert _placed(shifted, audio)[0].start_sample == placed[0].start_sample + SR // 2


def test_speech_stem_is_the_clips_verbatim_on_silence() -> None:
    arrangement, _, audio = _setup()
    placed = _placed(arrangement, audio)
    frames = round((arrangement.total_seconds + 2) * SR)
    stem = speech_stem(placed, audio, frames, 2)
    covered = np.zeros(frames, dtype=bool)
    for p in placed:
        assert np.array_equal(stem[p.start_sample : p.end_sample], audio[p.clip_id])
        covered[p.start_sample : p.end_sample] = True
    assert not stem[~covered].any()
    with pytest.raises(ValueError, match="past the end"):
        speech_stem(placed, audio, placed[-1].end_sample - 1, 2)


def test_tails_stop_before_the_next_clip() -> None:
    arrangement, _, audio = _setup()
    placed = _placed(arrangement, audio)
    frames = round((arrangement.total_seconds + 2) * SR)
    limits = tail_limits(placed, frames, SR)
    assert limits[0] == min(placed[1].start_sample - placed[0].end_sample, TAIL_MAX_S * SR)
    bus = speech_bus(placed, audio, frames, SR, gain_db=0, highpass_hz=90, threshold_db=-20,
                     reverb_send=0.5, delay_send=0.5, delay_s=0.375)  # fmt: skip
    for p, limit in zip(placed, limits, strict=True):
        tail_end = p.end_sample + limit
        assert np.abs(bus.wet[p.end_sample : tail_end]).max() > 0  # tails ring...
        if limit < TAIL_MAX_S * SR:  # ...and are silent by the time the next clip starts
            assert np.abs(bus.wet[tail_end - 10 : tail_end]).max() < 1e-3
    outside = np.ones(frames, dtype=bool)
    for p in placed:
        outside[p.start_sample : p.end_sample] = False
    assert not bus.dry[outside].any()  # the processed speech stays inside the clips


def test_melody_layer_modes() -> None:
    arrangement, melody, _ = _setup()
    assert melody_notes(arrangement, melody, "off") == ([], [])
    replay, under = melody_notes(arrangement, melody, "replay")
    assert under == []
    drop = next(s for s in arrangement.sections if s.role == "drop")
    start, stop = drop.start_bar * 2.0, (drop.start_bar + drop.bars) * 2.0
    in_drop = [n for n in replay if start <= n.start_s < stop]
    assert len(in_drop) == 3 * drop.bars  # the 1-bar phrase loops through the section
    assert all(n.end_s <= stop + 1e-9 for n in in_drop)
    assert in_drop[0].start_s == pytest.approx(start)
    _, under = melody_notes(arrangement, melody, "all")
    bed = next(s for s in arrangement.sections if s.clip_id == "a")
    first = [n for n in under if bed.start_bar * 2.0 <= n.start_s < (bed.start_bar + bed.bars) * 2]
    assert len(first) == 3  # played once under its clip
    assert first[0].start_s == pytest.approx(bed.start_bar * 2.0)
