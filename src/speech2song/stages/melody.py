"""Stage 4: turn the clips' speech pitch into a melody (the speech-to-song illusion).

Per clip: pYIN pitch track, notes at syllables. Across clips: the speaker's tuning
offset, the key, scale snapping, an octave that suits a melody, and the tempo that best
fits the clips. Then rhythm quantizing, chords per bar, MIDI, a short rendered reference
(two loops of the main phrase), and every quote's own melody rendered into one file
(04_quote_melodies.wav, indexed by 04_quote_melodies.json) for the music generator to be
conditioned on.

M8: the first length of a run whose melody is made is its anchor. Its tempo, key, the
speaker's tuning offset and the melody's octave are kept in the manifest (`shared`), and
the run's other lengths use them instead of finding their own, so all lengths of a talk
share them (and a quote in two lengths gets the same notes and chords).
"""

import math
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
import soundfile as sf
from rich.markup import escape

from speech2song.audio.pitch import Segment, segment_notes, track_pitch
from speech2song.audio.synth import (
    build_midi,
    find_soundfont,
    normalize_peak,
    render,
    render_array,
    write_midi,
    write_wav,
)
from speech2song.audio.theory import (
    PITCH_CLASS,
    Key,
    TimedPitch,
    alignment_cost,
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
from speech2song.manifest import Song, shared_from_melody, write_json
from speech2song.models import (
    BarChord,
    Clip,
    ClipMelody,
    ClipSet,
    Melody,
    MelodyNote,
    QuoteMelodies,
    SharedMusic,
    Transcript,
)
from speech2song.pipeline import Context, Stage, StagePlan, StageResult
from speech2song.stages.align import TRANSCRIPT
from speech2song.stages.select_clips import CLIPS

MELODY = "04_melody.json"
MELODY_MIDI = "04_melody.mid"
REFERENCE = "04_melody_reference.wav"
REFERENCE_LOOPS = 2  # the reference is short: it guides music generation
QUOTE_MELODIES = "04_quote_melodies.wav"
QUOTE_INDEX = "04_quote_melodies.json"
QUOTE_MIN_S = 8.0  # each quote's melody loops until it lasts at least this long...
QUOTE_MAX_S = 30.0  # ...and no longer than a conditioning reference may be
QUOTE_GAP_S = 0.5  # silence between the quotes in the file
PEAK_DB = -3.0
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
    shared: SharedMusic | None = None,
) -> Melody:
    """Everything after pitch tracking (pure): key, snapping, tempo, rhythm, chords.
    With `shared` (the run's anchor length's), its tuning offset, key, octave and tempo
    are used instead of being found from these clips (`--key` still wins)."""
    all_segments = [s for c in clips for s in segments[c.id]]
    raw = [s.pitch for s in all_segments]
    weights = [s.end - s.start for s in all_segments]
    tuning = estimate_tuning(raw, weights) if shared is None else shared.tuning_offset
    tuned = [p - tuning for p in raw]
    if shared is None or key_override:
        key, key_source, confidence = resolve_key(preset, key_override, tuned, weights)
    else:
        key, key_source, confidence = (Key(shared.tonic, shared.mode), shared.key_source,
                                       shared.key_confidence)  # fmt: skip
    strength = preset.melody.scale_snap_strength
    snapped = [snap_to_scale(p, key, strength) for p in tuned]
    shift, placed = place_octaves(snapped, shift=shared.octave_shift if shared else None)

    grid = grid_beats(preset.melody.quantize_grid)
    durations = [c.duration_s for c in clips]
    onsets = [s.start for s in all_segments]
    if shared is None:
        bpm, cost = search_tempo(preset.tempo.bpm, preset.tempo.tolerance_bpm, durations,
                                 onsets, grid)  # fmt: skip
    else:
        bpm, cost = shared.bpm, alignment_cost(shared.bpm, durations, onsets, grid)
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


def quote_loops(phrase: ClipMelody, bpm: float) -> int:
    """How many times a quote's phrase plays in its rendered melody."""
    seconds = phrase.bars * 4 * 60.0 / bpm
    loops = max(1, math.ceil(QUOTE_MIN_S / seconds - 1e-9))
    return max(1, min(loops, math.floor(QUOTE_MAX_S / seconds)))


def render_quote_melodies(
    melody: Melody, soundfont: Path, sample_rate: int
) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    """Every quote's melody (with its chords), one after another with a short silence
    between, peak-normalized together; and where each one sits (ms)."""
    blocks: list[np.ndarray] = []
    index: dict[str, tuple[int, int]] = {}
    gap = np.zeros((round(QUOTE_GAP_S * sample_rate), 2), dtype=np.float32)
    cursor = 0
    for phrase in melody.clips:
        if not phrase.notes:
            continue
        midi = build_midi([phrase], bpm=melody.bpm, instrument_name=melody.instrument,
                          loops=quote_loops(phrase, melody.bpm), gap_bars=0)  # fmt: skip
        audio = render_array(midi, soundfont, sample_rate)
        start = cursor
        cursor += len(audio)
        end = min(cursor, start + round(QUOTE_MAX_S * sample_rate))
        index[phrase.clip_id] = (round(start / sample_rate * 1000),
                                 round(end / sample_rate * 1000))  # fmt: skip
        blocks += [audio, gap]
        cursor += len(gap)
    if not blocks:
        return np.zeros((0, 2), dtype=np.float32), {}
    return normalize_peak(np.concatenate(blocks), PEAK_DB), index


def shared_music(ctx: Context) -> SharedMusic | None:
    """The tempo, key and octave this length takes from the run's anchor (None for the
    anchor itself, or before any length's melody was made)."""
    shared = ctx.run.manifest.shared
    return shared if shared is not None and shared.anchor != ctx.length else None


def anchor_song(ctx: Context) -> Song | None:
    """The run's anchor length, when the context's length is another one."""
    shared = shared_music(ctx)
    return ctx.run.song(shared.anchor) if shared is not None else None


class MelodyStage(Stage):
    name: ClassVar[str] = "melody"
    version: ClassVar[int] = 5  # 4: whole-number tempos; 5 (M7): each quote's melody rendered
    scope: ClassVar[Literal["run", "song"]] = "song"

    def plan(self, ctx: Context) -> StagePlan:
        preset = load_preset(ctx.run.manifest.preset, ctx.config.presets_dir)
        clips_path = ctx.song.path(CLIPS)
        inputs: dict[str, Path] = {"clips": clips_path, "transcript": ctx.run.path(TRANSCRIPT)}
        if clips_path.exists():  # the clip files themselves are inputs too
            clip_set = ClipSet.model_validate_json(clips_path.read_text(encoding="utf-8"))
            for clip in kept_clips(clip_set):
                inputs[f"clip {clip.id}"] = ctx.song.path(clip.file)
        params = {
            # The melody-layer fields only matter to the mix.
            "melody": preset.melody.model_dump(exclude={"layer", "layer_roles"}),
            "key": preset.key.model_dump(),
            "tempo": preset.tempo.model_dump(),
            "key_override": ctx.run.manifest.options.key,
            "soundfont": str(ctx.config.soundfont) if ctx.config.soundfont else "auto",
            "analysis": ANALYSIS_VERSION,
        }
        shared = shared_music(ctx)
        if shared is not None:  # (the anchor's own request stays as it was before M8)
            params["shared"] = shared.model_dump()
        return StagePlan(inputs, params,
                         [MELODY, MELODY_MIDI, REFERENCE, QUOTE_MELODIES, QUOTE_INDEX])  # fmt: skip

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
            audio, rate = sf.read(str(ctx.song.path(clip.file)), dtype="float32", always_2d=True)
            track = track_pitch(audio.mean(axis=1), rate)
            segments[clip.id] = segment_notes(track, clip_words(clip, transcript))
            voiced = np.isfinite(track.midi)
            median = float(np.median(track.midi[voiced])) if voiced.any() else None
            stats[clip.id] = (round(float(voiced.mean()), 3), median and round(median, 2))
            ctx.say(f"  {clip.id}: {len(segments[clip.id])} syllable notes, "
                    f"{voiced.mean():.0%} voiced")  # fmt: skip
        if not any(segments.values()):
            raise StageError("No pitched speech found in the clips.")

        shared = shared_music(ctx)
        melody = build_melody(clips, segments, stats, preset, ctx.run.manifest.options.key,
                              shared)  # fmt: skip
        write_json(ctx.song.path(MELODY), melody)
        if shared is None:  # the run's anchor: its tempo, key and octave are the run's
            ctx.run.manifest.shared = shared_from_melody(ctx.length, melody)
        loops = melody.loop_phrase_count
        full = build_midi(melody.clips, bpm=melody.bpm, instrument_name=melody.instrument,
                          loops=loops)  # fmt: skip
        write_midi(full, ctx.song.path(MELODY_MIDI))
        main = next(c for c in melody.clips if c.clip_id == melody.main_clip)
        reference = build_midi([main], bpm=melody.bpm, instrument_name=melody.instrument,
                               loops=REFERENCE_LOOPS)  # fmt: skip
        seconds = render(reference, ctx.song.path(REFERENCE), soundfont)
        rate = clip_set.sample_rate
        quotes, index = render_quote_melodies(melody, soundfont, rate)
        write_wav(ctx.song.path(QUOTE_MELODIES), quotes, rate)
        write_json(
            ctx.song.path(QUOTE_INDEX),
            QuoteMelodies(
                file=QUOTE_MELODIES, sample_rate=rate, instrument=melody.instrument, clips=index
            ),
        )

        confidence = f", confidence {melody.key_confidence:.2f}" if melody.key_confidence else ""
        ctx.say(
            f"  key {melody.key} ({melody.key_source}{confidence}), tuning offset "
            f"{melody.tuning_offset:+.2f} st, octave shift {melody.octave_shift:+d} st"
        )
        if shared is not None:
            ctx.say(f"  tempo, key and octave shared with {shared.anchor} (the run's anchor)")
        ctx.say(f"  tempo {melody.bpm:g} BPM (grid misalignment {melody.tempo_cost:.3f})")
        for clip in melody.clips:
            chords = " ".join(c.name for c in clip.chords)
            ctx.say(f"  {clip.clip_id}: {len(clip.notes)} notes over {clip.bars} bars | "
                    f"{escape(chords)}")  # fmt: skip
        ctx.say(f"  reference: {melody.main_clip} x{REFERENCE_LOOPS}, {seconds:.1f} s "
                f"({melody.instrument}, {soundfont.name})")  # fmt: skip
        ctx.say(f"  quote melodies: {len(index)} in {QUOTE_MELODIES} "
                f"({len(quotes) / rate:.1f} s)")  # fmt: skip
        return StageResult(
            summary={
                "key": melody.key,
                "bpm": melody.bpm,
                "notes": sum(len(c.notes) for c in melody.clips),
                "main_clip": melody.main_clip,
            }
        )
