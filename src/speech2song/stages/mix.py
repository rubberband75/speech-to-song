"""Stage 7: mix and master (free).

The speech stem holds the clips verbatim (sample-exact apart from their edge fades). The
mix hears the speech bus instead: the clips set to the music's loudness plus
`speech_level_lu`, high-passed, gently compressed, with reverb and delay sends. The music
is first shaped toward the arrangement's energy, its gaps become lifts into the next
section, and a passage played alone keeps its bed until its last phrase, which lands in
silence (the music rings away just before it and swells back a beat or two after the
last word; a song closing on such a passage ends there). It rings out at the end when it
would otherwise stop abruptly. Under each speech passage the music and melody layer are set
once, at the passage's bar lines and only as far as needed, and a guard keeps every word
clear of them. The master is normalized to the preset's loudness with true peaks at or
below -1 dBTP.
"""

import subprocess
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
import soundfile as sf
from rich.markup import escape

from speech2song.arrangement import energy_bounds, passages
from speech2song.audio.dsp import (
    db_to_gain,
    dips,
    loudness_lufs,
    master,
    speech_guard,
)
from speech2song.audio.io import require_tool
from speech2song.audio.mixing import (
    COMP_ABOVE_LU,
    RING_HOLD,
    EntryFix,
    NoteEvent,
    WordSpan,
    alone_break,
    apply_entry_fix,
    break_stop,
    cut_music,
    delay_seconds,
    ends_abruptly,
    energy_gains,
    gain_curve,
    gap_lift,
    last_phrase,
    late_entries,
    live_start,
    melody_notes,
    passage_curve,
    passage_levels,
    phrase_break,
    place_clips,
    return_point,
    ring_out,
    section_spans,
    settled_at,
    speech_bus,
    speech_stem,
    word_spans,
)
from speech2song.audio.synth import find_soundfont, instrument, render_array, write_wav
from speech2song.config import SAMPLE_RATE, MixSpec, load_preset
from speech2song.errors import S2SError, StageError
from speech2song.manifest import temp_path_for, write_json
from speech2song.models import (
    Arrangement,
    ClipSet,
    Melody,
    MixReport,
    SectionLevel,
    Transcript,
)
from speech2song.pipeline import Context, Stage, StagePlan, StageResult
from speech2song.stages.align import TRANSCRIPT
from speech2song.stages.arrange import ARRANGEMENT, usable_arrangement
from speech2song.stages.generate_music import SELECTED, song_frames
from speech2song.stages.melody import MELODY, kept_clips
from speech2song.stages.select_clips import CLIPS

MIX_DIR = "07_mix"
SPEECH_STEM = "07_mix/stems/speech.wav"
MUSIC_STEM = "07_mix/stems/music.wav"
MELODY_STEM = "07_mix/stems/melody_layer.wav"
MASTER_WAV = "07_mix/master.wav"
MASTER_MP3 = "07_mix/master.mp3"
MIX_REPORT = "07_mix/mix.json"
CEILING_DBTP = -1.0
MP3_BITRATE = "320k"
FALLBACK_MUSIC_LUFS = -20.0  # when the music is silent (nothing to measure)
RENDER_TAIL_S = 1.5
END_MARGIN_S = 0.5  # the file ends this long after the ring-out has died away
SPEECH_TAIL_KEEP_S = 3.0  # and at least this long after the last line

MelodyLayer = Literal["replay", "all", "off"]


def melody_layer_mode(ctx: Context) -> MelodyLayer:
    preset = load_preset(ctx.run.manifest.preset, ctx.config.presets_dir)
    return ctx.run.manifest.options.melody_layer or preset.melody.layer


def render_notes(
    notes: list[NoteEvent], bpm: float, instrument_name: str, soundfont: Path, frames: int
) -> np.ndarray:
    """Render note events with fluidsynth onto a stereo buffer of `frames`."""
    import pretty_midi

    out = np.zeros((frames, 2), dtype=np.float32)
    if not notes:
        return out
    program, base_velocity = instrument(instrument_name)
    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    part = pretty_midi.Instrument(program=program, name="melody layer")
    for n in notes:
        velocity = max(1, min(127, round(base_velocity * n.velocity / 100)))
        part.notes.append(pretty_midi.Note(velocity=velocity, pitch=n.midi, start=n.start_s,
                                           end=max(n.end_s, n.start_s + 0.01)))  # fmt: skip
    end = max(n.end_s for n in notes)
    part.control_changes.append(pretty_midi.ControlChange(number=7, value=100,
                                                          time=end + RENDER_TAIL_S))  # fmt: skip
    midi.instruments = [part]
    audio = render_array(midi, soundfont, SAMPLE_RATE)
    length = min(frames, len(audio))
    out[:length] = audio[:length]
    return out


def shape_music(
    music: np.ndarray, arrangement: Arrangement, spec: MixSpec, sr: int
) -> tuple[np.ndarray, list[SectionLevel], list[str], list[EntryFix]]:
    """The music pulled toward the arrangement's energy, with late entries after gaps
    moved onto their downbeat and each gap turned into a lift. Returns it, each section's
    levels and gain, the IDs of the lifted gaps, and the entry fixes."""
    fixes = late_entries(music, arrangement, sr, spec.late_entry_max_bars)
    for fix in fixes:
        music = apply_entry_fix(music, fix, sr)
    spans = section_spans(arrangement, sr)
    energies = [(a + b) / 2 for a, b in energy_bounds(arrangement.sections)]
    levels = [None if section.silent or section.rest else loudness_lufs(music[start:end], sr)
              for section, start, end in spans]  # fmt: skip
    gains, targets = energy_gains(levels, energies, range_db=spec.energy_range_db,
                                  tolerance_db=spec.energy_tolerance_db,
                                  max_db=spec.energy_max_db,
                                  max_boost_db=spec.energy_max_boost_db)  # fmt: skip
    for i, (section, _, _) in enumerate(spans):  # change gain inside the gap, not before it
        if (section.silent or section.rest) and i:
            gains[i] = gains[i - 1]
    frames = len(music)
    curve = gain_curve([(a, b) for _, a, b in spans], gains, frames, sr)
    shaped = music * curve[:, None]
    gaps = [(start, end) for section, start, end in spans if section.silent]
    for start, end in gaps:  # a lift into the next section, from the build's tail
        end = min(end, frames)
        shaped[start:end] = gap_lift(shaped, start, end, sr, swell=spec.gap_swell,
                                     lift_db=spec.gap_lift_db)  # fmt: skip
    report = [
        SectionLevel(section_id=section.id, role=section.role, energy=round(energy, 3),
                     lufs=None if level is None else round(level, 2),
                     target_lufs=None if target is None else round(target, 2),
                     gain_db=0.0 if section.silent or section.rest else round(gain, 2))
        for (section, _, _), energy, level, target, gain
        in zip(spans, energies, levels, targets, gains, strict=True)
    ]  # fmt: skip
    lifted = [s.id for s, _, _ in spans if s.silent]
    return shaped.astype(np.float32), report, lifted, fixes


def write_mp3(wav: Path, mp3: Path) -> None:
    ffmpeg = require_tool("ffmpeg")
    tmp = temp_path_for(mp3)
    args = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav),
            "-c:a", "libmp3lame", "-b:a", MP3_BITRATE, "-fflags", "+bitexact",
            "-flags:a", "+bitexact", str(tmp)]  # fmt: skip
    try:
        subprocess.run(args, capture_output=True, text=True, check=True)
        tmp.replace(mp3)
    except subprocess.CalledProcessError as exc:
        raise S2SError(f"ffmpeg could not write the MP3: {exc.stderr.strip()[-300:]}") from exc
    finally:
        tmp.unlink(missing_ok=True)


def _db(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}"


class MixStage(Stage):
    name: ClassVar[str] = "mix"
    # 2: energy shaping, gaps; 3: late entries; 4: speech guard; 5: guard words by midpoint;
    # 6 (M6): gaps lift instead of muting, a duck per quote, a gentler guard, a ring-out;
    # 7 (M7): a level per passage at its bar lines, stops for passages played alone, a
    # ring-out only where the music would stop abruptly; 8 (M7.1): a passage played
    # alone loses its music only for its last phrase, quiet sections can be lifted more
    version: ClassVar[int] = 8
    scope: ClassVar[Literal["run", "song"]] = "song"

    def plan(self, ctx: Context) -> StagePlan:
        preset = load_preset(ctx.run.manifest.preset, ctx.config.presets_dir)
        mode = melody_layer_mode(ctx)
        inputs: dict[str, Path] = {
            "arrangement": ctx.song.path(ARRANGEMENT),
            "music": ctx.song.path(SELECTED),
            "clips": ctx.song.path(CLIPS),
            "transcript": ctx.run.path(TRANSCRIPT),  # word timings for the speech guard
        }
        if mode != "off":
            inputs["melody"] = ctx.song.path(MELODY)
        if ctx.song.path(CLIPS).exists():  # the clip files the mix plays are inputs too
            clip_set = ClipSet.model_validate_json(ctx.song.path(CLIPS).read_text())
            playing = {clip.id for clip in kept_clips(clip_set)}
            if ctx.song.path(ARRANGEMENT).exists():
                arrangement = Arrangement.model_validate_json(
                    ctx.song.path(ARRANGEMENT).read_text()
                )
                playing |= {s.clip_id for s in arrangement.sections if s.clip_id}
            for clip in clip_set.clips:
                if clip.id in playing:
                    inputs[f"clip {clip.id}"] = ctx.song.path(clip.file)
        params = {
            "mix": preset.mix.model_dump(),
            "melody_layer": mode,
            "instrument": preset.melody.render_instrument,
            "soundfont": str(ctx.config.soundfont) if ctx.config.soundfont else "auto",
            "ceiling_dbtp": CEILING_DBTP,
        }
        outputs = [SPEECH_STEM, MUSIC_STEM, MASTER_WAV, MASTER_MP3, MIX_REPORT]
        if mode != "off":
            outputs.append(MELODY_STEM)
        return StagePlan(inputs, params, outputs)

    def run(self, ctx: Context, plan: StagePlan) -> StageResult:
        preset = load_preset(ctx.run.manifest.preset, ctx.config.presets_dir)
        spec = preset.mix
        mode: MelodyLayer = plan.params["melody_layer"]
        sr = SAMPLE_RATE
        arrangement = usable_arrangement(ctx)
        clip_set = ClipSet.model_validate_json(plan.inputs["clips"].read_text())
        clips = {c.id: c for c in clip_set.clips}
        if clip_set.sample_rate != sr:
            raise StageError(f"clips are {clip_set.sample_rate} Hz; the mix runs at {sr} Hz")

        frames = song_frames(arrangement, sr)
        music, rate = sf.read(str(plan.inputs["music"]), dtype="float32", always_2d=True)
        if rate != sr or len(music) != frames:
            raise StageError(f"{SELECTED} doesn't match the arrangement; re-run `generate`")
        audio = {}
        for clip_id in {s.clip_id for s in arrangement.sections if s.clip_id}:
            block, _ = sf.read(str(ctx.song.path(clips[clip_id].file)), dtype="float32",
                               always_2d=True)  # fmt: skip
            audio[clip_id] = block
        placed = place_clips(arrangement, {k: c.file for k, c in clips.items()},
                             {k: len(a) for k, a in audio.items()}, sr)  # fmt: skip
        warnings: list[str] = []
        music, levels, lifted, fixes = shape_music(music, arrangement, spec, sr)
        transcript = Transcript.model_validate_json(plan.inputs["transcript"].read_text())
        words = word_spans(placed, clips, transcript.words, sr)
        spans = section_spans(arrangement, sr)
        alone, breaks = [], []
        sections = arrangement.sections
        last_music = max((i for i, s in enumerate(sections) if not s.rest), default=-1)
        alone_words: set[WordSpan] = set()  # the last phrases, heard without music
        closing_stop = None
        for run in passages(arrangement):
            ids = {section.id for section in run}
            start = next(a for s, a, _ in spans if s.id == run[0].id)
            end = next(b for s, _, b in spans if s.id == run[-1].id)
            mine = [w for w in words if w.section_id in ids]
            last_word = max((w.end for w in mine), default=None)
            if run[0].rest:  # an M7 passage played alone: the music stops on its bar line
                closing = sections.index(run[0]) > last_music  # the ending rings it
                music = alone_break(music, start, end, last_word, sr,
                                    ring_s=0.0 if closing else spec.alone_ring_s,
                                    swell=spec.gap_swell)  # fmt: skip
                alone += [section.id for section in run]
                continue
            if run[0].treatment != "break" or not mine:
                continue
            phrase = last_phrase(mine, sr)
            stop = break_stop(mine, phrase, start, sr)
            alone_words |= set(mine[phrase or 0 :])
            closing = sections.index(run[-1]) == last_music
            back = None if closing else return_point(last_word, end, arrangement.bpm, sr)
            if back is None:  # the song ends on this line: the ending is made from here
                music, closing_stop = cut_music(music, stop, None, sr), stop
            else:
                music = phrase_break(music, stop, back, last_word, sr,
                                     ring_s=spec.alone_ring_s, swell=spec.gap_swell)  # fmt: skip
            breaks.append({"sections": [section.id for section in run],
                           "phrase": " ".join(w.word for w in mine[phrase or 0 :]),
                           "stop_s": round(stop / sr, 3),
                           "back_s": None if back is None else round(back / sr, 3)})  # fmt: skip
        music_spans = [(a, b) for s, a, b in spans if not s.rest]
        if closing_stop is not None:
            music_spans = [(a, min(b, closing_stop)) for a, b in music_spans if a < closing_stop]
        music_end = music_spans[-1][1] if music_spans else len(music)
        ended, ring_s = None, spec.ring_out_s
        if arrangement.ending is None:  # an M5 arrangement: the last chord always rings out
            start = music_spans[-1][0] if music_spans else 0
            music, ring_at = ring_out(music, sr, start, ring_s)
        else:  # rung out of the last music that is actually heard
            start = live_start(music, music_spans, sr, loudness_lufs(music, sr))
            if arrangement.ending == "stop":  # a crisp stop on the last bar
                ring_s = spec.alone_ring_s
                music, ring_at = ring_out(music, sr, start, ring_s, hold=0.0)
            elif arrangement.ending == "held_chord" or ends_abruptly(music, sr, start,
                                                                      music_end):  # fmt: skip
                hold = RING_HOLD
                if closing_stop is not None and arrangement.ending == "fade":
                    ring_s, hold = spec.alone_ring_s, 0.0  # the closing line lands alone
                music, ring_at = ring_out(music, sr, start, ring_s, hold=hold)
            else:  # the music fades away by itself: leave its ending as it is
                ring_at, ended = len(music), "natural"
            ended = ended or ("ring" if ring_at < len(music) else None)
        keep = None
        if ring_at < len(music):  # the song ends where the ring dies away, not 4 s of silence later
            keep = ring_at + round((ring_s + END_MARGIN_S) * sr)
        elif ended == "natural":
            keep = settled_at(music, sr, start) + round(END_MARGIN_S * sr)
        if keep is not None:
            if placed:  # but never before the last line's own tail
                keep = max(keep, placed[-1].end_sample + round(SPEECH_TAIL_KEEP_S * sr))
            music = music[:keep]
            if ended == "natural":
                fade = min(len(music), round(END_MARGIN_S * sr))
                music[len(music) - fade :] *= np.linspace(1, 0, fade, dtype=np.float32)[:, None]
        frames = len(music)  # the ring-out may also run past the song's end
        dry = speech_stem(placed, audio, frames, clip_set.channels)

        # Levels: speech sits speech_level_lu above the music's loudness.
        music_lufs = loudness_lufs(music, sr)
        reference = music_lufs if music_lufs is not None else FALLBACK_MUSIC_LUFS
        speech_target = reference + spec.speech_level_lu
        speech_lufs = loudness_lufs(dry, sr)
        if speech_lufs is None:
            warnings.append("the speech stem is silent")
        gain_db = speech_target - speech_lufs if speech_lufs is not None else 0.0
        bus = speech_bus(
            placed, audio, frames, sr, gain_db=gain_db, highpass_hz=spec.speech_highpass_hz,
            threshold_db=speech_target + COMP_ABOVE_LU, reverb_send=spec.speech_reverb_send,
            delay_send=spec.speech_delay_send, delay_s=delay_seconds(arrangement.bpm),
        )  # fmt: skip
        processed = loudness_lufs(bus.dry, sr)
        if processed is not None:  # make up for what the high-pass and compressor took
            makeup = np.float32(db_to_gain(speech_target - processed))
            bus = type(bus)(bus.dry * makeup, bus.wet * makeup)
            gain_db += speech_target - processed

        # Under each passage the music and melody layer are set once, at its bar lines
        # (measured under the words the music plays beneath: not a last phrase alone).
        beds = passage_levels(music, bus.dry, arrangement,
                              [w for w in words if w not in alone_words], sr,
                               margin_db=spec.bed_margin_db,
                               max_cut_db=spec.sidechain_duck_db)  # fmt: skip
        duck = passage_curve(beds, frames, sr)[:, None]

        layer = np.zeros((frames, 2), dtype=np.float32)
        melody_gain_db = None
        if mode != "off":
            melody = Melody.model_validate_json(plan.inputs["melody"].read_text())
            replay, under = melody_notes(arrangement, melody, mode)
            soundfont = find_soundfont(ctx.config.soundfont)
            tracks = [render_notes(notes, arrangement.bpm, melody.instrument, soundfont, frames)
                      for notes in (replay, under)]  # fmt: skip
            measured = loudness_lufs(tracks[0], sr) or loudness_lufs(tracks[1], sr)
            if measured is not None:
                melody_gain_db = reference + spec.melody_layer_lu - measured
                under_gain = melody_gain_db + spec.melody_under_speech_db
                layer = (tracks[0] * np.float32(db_to_gain(melody_gain_db))
                         + tracks[1] * np.float32(db_to_gain(under_gain)))  # fmt: skip
            else:
                warnings.append(f"melody layer '{mode}' has no notes to play here")

        music_bus = music * duck
        layer_bus = layer * duck
        # The guard: under every word, the music stays speech_margin_db below it.
        guard, margins = speech_guard(bus.dry, music_bus + layer_bus, sr,
                                      [(w.start, w.end) for w in words],
                                      margin_db=spec.speech_margin_db)  # fmt: skip
        music_bus *= guard[:, None]
        layer_bus *= guard[:, None]
        guarded = [
            {"clip_id": w.clip_id, "word": w.word, "at_s": round(w.start / sr, 3),
             "margin_db": round(m, 1)}
            for w, m in zip(words, margins, strict=True) if m < spec.speech_margin_db
        ]  # fmt: skip
        mastered = master(music_bus + layer_bus + bus.dry + bus.wet, sr, spec.target_lufs,
                          CEILING_DBTP)  # fmt: skip

        stems = ctx.song.path(SPEECH_STEM).parent
        stems.mkdir(parents=True, exist_ok=True)
        write_wav(ctx.song.path(SPEECH_STEM), dry, sr)
        write_wav(ctx.song.path(MUSIC_STEM), music_bus, sr)
        if mode != "off":
            write_wav(ctx.song.path(MELODY_STEM), layer_bus, sr)
        else:
            ctx.song.path(MELODY_STEM).unlink(missing_ok=True)
        master_wav = ctx.song.path(MASTER_WAV)
        tmp = temp_path_for(master_wav)
        try:
            sf.write(str(tmp), mastered.audio, sr, subtype="PCM_24")
            tmp.replace(master_wav)
        finally:
            tmp.unlink(missing_ok=True)
        write_mp3(master_wav, ctx.song.path(MASTER_MP3))

        if mastered.lufs is None:
            raise StageError("the mix is silent")
        report = MixReport(
            sample_rate=sr, frames=frames, seconds=round(frames / sr, 4), take=SELECTED,
            clips=placed, melody_layer=mode, music_lufs=music_lufs, speech_lufs=speech_lufs,
            speech_gain_db=round(gain_db, 3), duck_db=spec.sidechain_duck_db,
            melody_gain_db=None if melody_gain_db is None else round(melody_gain_db, 3),
            target_lufs=spec.target_lufs, master_gain_db=round(mastered.gain_db, 3),
            integrated_lufs=round(mastered.lufs, 3),
            true_peak_dbtp=round(mastered.true_peak_db, 3), section_levels=levels,
            lifted=lifted, ring_out_at_s=round(ring_at / sr, 3) if ring_at < frames else None,
            ending=arrangement.ending, ended=ended, alone=alone, breaks=breaks,
            passages=[{"sections": p.sections, "start_s": round(p.start / sr, 3),
                       "end_s": round(p.end / sr, 3), "margin_db": p.margin_db,
                       "gain_db": p.gain_db} for p in beds],
            warnings=warnings,
            late_entries=[{"section_id": f.section_id, "bars": f.bars, "mode": f.mode}
                          for f in fixes],
            speech_margin_db=spec.speech_margin_db, guarded_words=guarded,
            guard_dips=[{"start_s": round(a, 3), "end_s": round(b, 3), "db": round(d, 2)}
                        for a, b, d in dips(guard, sr)],
        )  # fmt: skip
        write_json(ctx.song.path(MIX_REPORT), report)
        for warning in warnings:
            ctx.say(f"[yellow]  {warning}[/]")
        changed = [f"{level.section_id} {level.gain_db:+.1f} dB" for level in levels
                   if level.gain_db]  # fmt: skip
        for fix in fixes:
            how = ("taken from that much later, with its spill into the next section"
                   if fix.mode == "shift" else "filled with the bars after them")  # fmt: skip
            ctx.say(f"  {fix.section_id} came in {fix.bars} bar(s) late after the gap: "
                    f"{how}")  # fmt: skip
        ctx.say(f"  music shaped toward the arrangement: {', '.join(changed) or 'no changes'}"
                + (f"; lifts into {', '.join(lifted)}" if lifted else ""))  # fmt: skip
        if alone:
            ctx.say(f"  the music stops for the passages played alone ({', '.join(alone)}) "
                    "and swells back after them")  # fmt: skip
        for item in breaks:
            back = ("the song ends there" if item["back_s"] is None
                    else f"back at {item['back_s']:.1f} s")  # fmt: skip
            ctx.say(f"  {item['sections'][0]}: the music drops out at {item['stop_s']:.1f} s for "
                    f"\"{escape(item['phrase'])}\" ({back})")  # fmt: skip
        if ring_at < frames:
            ctx.say(f"  the last chord rings out from {ring_at / sr:.1f} s, "
                    f"over {ring_s:g} s")  # fmt: skip
        elif ended == "natural":
            ctx.say(f"  the music's own ending ({arrangement.ending.replace('_', ' ')}) dies "
                    "away by itself; no ring-out")  # fmt: skip
        if beds:
            shown = ", ".join(f"{p.sections[0]}" + (f"-{p.sections[-1]}" if len(p.sections) > 1
                                                    else "") + f" {p.gain_db:+.1f} dB"
                              for p in beds)  # fmt: skip
            ctx.say(f"  music under the passages (set at their bar lines): {shown}")
        if guarded:
            worst = min(guarded, key=lambda w: w["margin_db"])
            ctx.say(f"  speech guard: {len(guarded)} of {len(words)} words were less than "
                    f"{spec.speech_margin_db:g} dB clear of the music, which now dips under "
                    f"them (the closest: '{worst['word']}' at {worst['at_s']:.1f} s, "
                    f"{worst['margin_db']:+.1f} dB)")  # fmt: skip
        ctx.say(f"  music {_db(music_lufs)} LUFS · speech {_db(speech_lufs)} LUFS dry, "
                f"{gain_db:+.1f} dB to sit {spec.speech_level_lu:+g} LU above the music · "
                f"melody layer: {mode}")  # fmt: skip
        minutes, seconds = divmod(round(frames / sr), 60)
        ctx.say(f"  master {mastered.lufs:.1f} LUFS, true peak {mastered.true_peak_db:.1f} dBTP, "
                f"{minutes}:{seconds:02d}")  # fmt: skip
        return StageResult(
            outputs=[*plan.outputs],
            summary={"lufs": round(mastered.lufs, 2), "true_peak_dbtp": round(
                mastered.true_peak_db, 2), "seconds": round(frames / sr, 2)},
        )  # fmt: skip
