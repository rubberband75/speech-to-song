"""Stage 4: turn the clips' speech pitch into a melody (the speech-to-song illusion).

Per clip: pYIN pitch track, notes at syllables. Across clips: the speaker's tuning
offset, the key, scale snapping, an octave that suits a melody, and the tempo that best
fits the clips. Then rhythm quantizing, chords per bar, MIDI, and a short rendered
reference (two loops of the main phrase) for the music generator.
"""

import math
from pathlib import Path
from typing import ClassVar

import numpy as np
import soundfile as sf
from rich.markup import escape

from speech2song.audio.pitch import Segment, segment_notes, track_pitch
from speech2song.audio.synth import build_midi, find_soundfont, render, write_midi
from speech2song.audio.theory import (
    PITCH_CLASS,
    Key,
    TimedPitch,
    choose_chords,
    estimate_tuning,
    grid_beats,
    parse_key,
    pitch_class_histogram,
    place_octaves,
    quantize,
    rank_keys,
    search_tempo,
    snap_to_scale,
)
from speech2song.config import Preset, load_preset
from speech2song.errors import S2SError, StageError
from speech2song.manifest import write_json
from speech2song.models import (
    BarChord,
    Clip,
    ClipMelody,
    ClipSet,
    Melody,
    MelodyNote,
    Transcript,
)
from speech2song.pipeline import Context, Stage, StagePlan, StageResult
from speech2song.stages.align import TRANSCRIPT
from speech2song.stages.select_clips import CLIPS

MELODY = "04_melody.json"
MELODY_MIDI = "04_melody.mid"
REFERENCE = "04_melody_reference.wav"
REFERENCE_LOOPS = 2  # the reference is short: it guides music generation
ANALYSIS_VERSION = 1  # bump when pitch tracking or segmentation changes


def kept_clips(clip_set: ClipSet) -> list[Clip]:
    by_id = {c.id: c for c in clip_set.clips}
    return [by_id[i] for i in clip_set.order]


def clip_words(clip: Clip, transcript: Transcript) -> list[tuple[str, float, float]]:
    """Transcript words inside the clip, with clip-relative times."""
    return [
        (w.w, w.start - clip.start_s, w.end - clip.start_s)
        for w in transcript.words
        if clip.start_s <= (w.start + w.end) / 2 <= clip.end_s
    ]


def resolve_key(
    preset: Preset, override: str | None, pitches: list[float], weights: list[float]
) -> tuple[Key, str, float | None]:
    """(key, source, confidence): --key wins, then a fixed preset tonic, then detection
    (with the preset's mode as a prior), then the preset's fallback tonic."""
    if override:
        try:
            return parse_key(override), "override", None
        except ValueError as exc:
            raise S2SError(str(exc)) from exc
    if preset.key.tonic != "auto":
        return Key(PITCH_CLASS[preset.key.tonic.upper()], preset.key.mode), "preset", None
    ranked = rank_keys(pitch_class_histogram(pitches, weights), prefer_mode=preset.key.mode)
    if ranked:
        score, key = ranked[0]
        return key, "detected", round(score, 3)
    return Key(PITCH_CLASS[preset.key.fallback.upper()], preset.key.mode), "fallback", None


def velocities(segments: list[Segment]) -> list[int]:
    """Louder syllables play louder: 50-100 over the clip's top 30 dB."""
    if not segments:
        return []
    top = max(s.level_db for s in segments)
    return [round(50 + 50 * min(1.0, max(0.0, (s.level_db - top + 30) / 30))) for s in segments]


def bar_weights(notes: list[MelodyNote], bars: int) -> list[np.ndarray]:
    """Per bar: how many beats each pitch class sounds."""
    weights = [np.zeros(12) for _ in range(bars)]
    for note in notes:
        start, end = note.start_beat, note.start_beat + note.beats
        for bar in range(int(start // 4), min(bars, math.ceil(end / 4))):
            overlap = min(end, (bar + 1) * 4) - max(start, bar * 4)
            if overlap > 0:
                weights[bar][note.midi % 12] += overlap
    return weights


def build_melody(
    clips: list[Clip],
    segments: dict[str, list[Segment]],
    stats: dict[str, tuple[float, float | None]],
    preset: Preset,
    key_override: str | None,
) -> Melody:
    """Everything after pitch tracking (pure): key, snapping, tempo, rhythm, chords."""
    all_segments = [s for c in clips for s in segments[c.id]]
    raw = [s.pitch for s in all_segments]
    weights = [s.end - s.start for s in all_segments]
    tuning = estimate_tuning(raw, weights)
    tuned = [p - tuning for p in raw]
    key, key_source, confidence = resolve_key(preset, key_override, tuned, weights)
    strength = preset.melody.scale_snap_strength
    shift, placed = place_octaves([snap_to_scale(p, key, strength) for p in tuned])

    grid = grid_beats(preset.melody.quantize_grid)
    bpm, cost = search_tempo(
        preset.tempo.bpm,
        preset.tempo.tolerance_bpm,
        [c.duration_s for c in clips],
        [s.start for s in all_segments],
        grid,
    )
    beat = 60.0 / bpm
    pitch_of = dict(zip(map(id, all_segments), placed, strict=True))
    melodies = []
    for clip in clips:
        clip_segments = segments[clip.id]
        timed = [
            TimedPitch(s.start, s.end, pitch_of[id(s)], v)
            for s, v in zip(clip_segments, velocities(clip_segments), strict=True)
        ]
        notes = []
        for g in quantize(timed, bpm, grid):
            seg = clip_segments[g.source]
            notes.append(
                MelodyNote(
                    start_beat=g.start_beat,
                    beats=g.beats,
                    midi=round(g.pitch),
                    pitch=round(g.pitch, 3),
                    speech_pitch=round(seg.pitch, 3),
                    start_s=seg.start,
                    end_s=seg.end,
                    velocity=g.velocity,
                    word=seg.word,
                )
            )
        last_beat = max((n.start_beat + n.beats for n in notes), default=0.0)
        bars = max(1, math.ceil(max(clip.duration_s / beat, last_beat) / 4 - 1e-9))
        chords = [
            BarChord(bar=i, degree=t.degree, name=t.name, pitch_classes=list(t.pitch_classes))
            for i, t in enumerate(choose_chords(bar_weights(notes, bars), key))
        ]
        voiced, median = stats[clip.id]
        melodies.append(
            ClipMelody(
                clip_id=clip.id,
                file=clip.file,
                duration_s=clip.duration_s,
                bars=bars,
                voiced_ratio=voiced,
                median_speech_pitch=median,
                notes=notes,
                chords=chords,
            )
        )
    main = next((c.id for c in clips if c.role == "hook"), clips[0].id)
    return Melody(
        key=key.name,
        tonic=key.tonic,
        mode=key.mode,
        key_source=key_source,  # type: ignore[arg-type]
        key_confidence=confidence,
        tuning_offset=round(tuning, 3),
        bpm=bpm,
        tempo_cost=round(cost, 4),
        grid=preset.melody.quantize_grid,
        snap_strength=strength,
        octave_shift=shift,
        loop_phrase_count=preset.melody.loop_phrase_count,
        instrument=preset.melody.render_instrument,
        main_clip=main,
        clips=melodies,
    )


class MelodyStage(Stage):
    name: ClassVar[str] = "melody"
    version: ClassVar[int] = 3

    def plan(self, ctx: Context) -> StagePlan:
        preset = load_preset(ctx.run.manifest.preset, ctx.config.presets_dir)
        clips_path = ctx.run.path(CLIPS)
        inputs: dict[str, Path] = {"clips": clips_path, "transcript": ctx.run.path(TRANSCRIPT)}
        if clips_path.exists():  # the clip files themselves are inputs too
            clip_set = ClipSet.model_validate_json(clips_path.read_text(encoding="utf-8"))
            for clip in kept_clips(clip_set):
                inputs[f"clip {clip.id}"] = ctx.run.path(clip.file)
        params = {
            # The melody-layer fields only matter to the mix.
            "melody": preset.melody.model_dump(exclude={"layer", "layer_roles"}),
            "key": preset.key.model_dump(),
            "tempo": preset.tempo.model_dump(),
            "key_override": ctx.run.manifest.options.key,
            "soundfont": str(ctx.config.soundfont) if ctx.config.soundfont else "auto",
            "analysis": ANALYSIS_VERSION,
        }
        return StagePlan(inputs, params, [MELODY, MELODY_MIDI, REFERENCE])

    def run(self, ctx: Context, plan: StagePlan) -> StageResult:
        preset = load_preset(ctx.run.manifest.preset, ctx.config.presets_dir)
        clip_set = ClipSet.model_validate_json(plan.inputs["clips"].read_text(encoding="utf-8"))
        transcript = Transcript.model_validate_json(plan.inputs["transcript"].read_text())
        clips = kept_clips(clip_set)
        if not clips:
            raise StageError("No clips to build a melody from (all were dropped).")
        soundfont = find_soundfont(ctx.config.soundfont)

        segments: dict[str, list[Segment]] = {}
        stats: dict[str, tuple[float, float | None]] = {}
        for clip in clips:
            audio, rate = sf.read(str(ctx.run.path(clip.file)), dtype="float32", always_2d=True)
            track = track_pitch(audio.mean(axis=1), rate)
            segments[clip.id] = segment_notes(track, clip_words(clip, transcript))
            voiced = np.isfinite(track.midi)
            median = float(np.median(track.midi[voiced])) if voiced.any() else None
            stats[clip.id] = (round(float(voiced.mean()), 3), median and round(median, 2))
            ctx.say(f"  {clip.id}: {len(segments[clip.id])} syllable notes, "
                    f"{voiced.mean():.0%} voiced")  # fmt: skip
        if not any(segments.values()):
            raise StageError("No pitched speech found in the clips.")

        melody = build_melody(clips, segments, stats, preset, ctx.run.manifest.options.key)
        write_json(ctx.run.path(MELODY), melody)
        loops = melody.loop_phrase_count
        full = build_midi(melody.clips, bpm=melody.bpm, instrument_name=melody.instrument,
                          loops=loops)  # fmt: skip
        write_midi(full, ctx.run.path(MELODY_MIDI))
        main = next(c for c in melody.clips if c.clip_id == melody.main_clip)
        reference = build_midi([main], bpm=melody.bpm, instrument_name=melody.instrument,
                               loops=REFERENCE_LOOPS)  # fmt: skip
        seconds = render(reference, ctx.run.path(REFERENCE), soundfont)

        confidence = f", confidence {melody.key_confidence:.2f}" if melody.key_confidence else ""
        ctx.say(
            f"  key {melody.key} ({melody.key_source}{confidence}), tuning offset "
            f"{melody.tuning_offset:+.2f} st, octave shift {melody.octave_shift:+d} st"
        )
        ctx.say(f"  tempo {melody.bpm:g} BPM (grid misalignment {melody.tempo_cost:.3f})")
        for clip in melody.clips:
            chords = " ".join(c.name for c in clip.chords)
            ctx.say(f"  {clip.clip_id}: {len(clip.notes)} notes over {clip.bars} bars | "
                    f"{escape(chords)}")  # fmt: skip
        ctx.say(f"  reference: {melody.main_clip} x{REFERENCE_LOOPS}, {seconds:.1f} s "
                f"({melody.instrument}, {soundfont.name})")  # fmt: skip
        return StageResult(
            summary={
                "key": melody.key,
                "bpm": melody.bpm,
                "notes": sum(len(c.notes) for c in melody.clips),
                "main_clip": melody.main_clip,
            }
        )
