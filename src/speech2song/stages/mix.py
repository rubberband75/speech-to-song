"""Stage 7: mix and master (free).

The speech stem holds the clips verbatim (sample-exact apart from their edge fades). The
mix hears the speech bus instead: the clips set to the music's loudness plus
`speech_level_lu`, high-passed, gently compressed, with reverb and delay sends. The music
and melody layer are ducked under the speech. The master is normalized to the preset's
loudness with true peaks at or below -1 dBTP.
"""

import subprocess
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
import soundfile as sf

from speech2song.arrangement import ClipInfo, clip_info, validate_arrangement
from speech2song.audio.dsp import db_to_gain, duck_gain, loudness_lufs, master
from speech2song.audio.io import require_tool
from speech2song.audio.mixing import (
    COMP_ABOVE_LU,
    NoteEvent,
    delay_seconds,
    melody_notes,
    place_clips,
    speech_bus,
    speech_stem,
)
from speech2song.audio.synth import find_soundfont, instrument, render_array, write_wav
from speech2song.config import SAMPLE_RATE, load_preset
from speech2song.errors import S2SError, StageError
from speech2song.manifest import temp_path_for, write_json
from speech2song.models import Arrangement, ClipSet, Melody, MixReport
from speech2song.pipeline import Context, Stage, StagePlan, StageResult
from speech2song.stages.arrange import ARRANGEMENT
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
DUCK_FLOOR_LU = -30.0  # speech this far below its loudness doesn't duck the music
DUCK_FULL_LU = -15.0  # from here up the music is fully ducked
MP3_BITRATE = "320k"
FALLBACK_MUSIC_LUFS = -20.0  # when the music is silent (nothing to measure)
RENDER_TAIL_S = 1.5

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
    version: ClassVar[int] = 1

    def plan(self, ctx: Context) -> StagePlan:
        preset = load_preset(ctx.run.manifest.preset, ctx.config.presets_dir)
        mode = melody_layer_mode(ctx)
        inputs: dict[str, Path] = {
            "arrangement": ctx.run.path(ARRANGEMENT),
            "music": ctx.run.path(SELECTED),
            "clips": ctx.run.path(CLIPS),
        }
        if mode != "off":
            inputs["melody"] = ctx.run.path(MELODY)
        if ctx.run.path(CLIPS).exists():  # the clip files the mix plays are inputs too
            clip_set = ClipSet.model_validate_json(ctx.run.path(CLIPS).read_text())
            playing = {clip.id for clip in kept_clips(clip_set)}
            if ctx.run.path(ARRANGEMENT).exists():
                arrangement = Arrangement.model_validate_json(ctx.run.path(ARRANGEMENT).read_text())
                playing |= {s.clip_id for s in arrangement.sections if s.clip_id}
            for clip in clip_set.clips:
                if clip.id in playing:
                    inputs[f"clip {clip.id}"] = ctx.run.path(clip.file)
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
        arrangement = Arrangement.model_validate_json(plan.inputs["arrangement"].read_text())
        clip_set = ClipSet.model_validate_json(plan.inputs["clips"].read_text())
        clips = {c.id: c for c in clip_set.clips}
        infos: dict[str, ClipInfo] = {c.id: clip_info(c) for c in clip_set.clips}
        problems = validate_arrangement(arrangement, infos)
        if problems:
            raise StageError(f"{ARRANGEMENT} can't be mixed: " + "; ".join(problems))
        if clip_set.sample_rate != sr:
            raise StageError(f"clips are {clip_set.sample_rate} Hz; the mix runs at {sr} Hz")

        frames = song_frames(arrangement, sr)
        music, rate = sf.read(str(plan.inputs["music"]), dtype="float32", always_2d=True)
        if rate != sr or len(music) != frames:
            raise StageError(f"{SELECTED} doesn't match the arrangement; re-run `generate`")
        audio = {}
        for clip_id in {s.clip_id for s in arrangement.sections if s.clip_id}:
            block, _ = sf.read(str(ctx.run.path(clips[clip_id].file)), dtype="float32",
                               always_2d=True)  # fmt: skip
            audio[clip_id] = block
        placed = place_clips(arrangement, {k: c.file for k, c in clips.items()},
                             {k: len(a) for k, a in audio.items()}, sr)  # fmt: skip
        dry = speech_stem(placed, audio, frames, clip_set.channels)
        warnings: list[str] = []

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

        # Ducking, from the dry speech at its mix level.
        mono = dry.mean(axis=1) * np.float32(db_to_gain(gain_db))
        duck = duck_gain(mono, sr, frames, depth_db=spec.sidechain_duck_db,
                         attack_ms=spec.duck_attack_ms, release_ms=spec.duck_release_ms,
                         floor_db=speech_target + DUCK_FLOOR_LU,
                         full_db=speech_target + DUCK_FULL_LU)[:, None]  # fmt: skip

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
        mastered = master(music_bus + layer_bus + bus.dry + bus.wet, sr, spec.target_lufs,
                          CEILING_DBTP)  # fmt: skip

        stems = ctx.run.path(SPEECH_STEM).parent
        stems.mkdir(parents=True, exist_ok=True)
        write_wav(ctx.run.path(SPEECH_STEM), dry, sr)
        write_wav(ctx.run.path(MUSIC_STEM), music_bus, sr)
        if mode != "off":
            write_wav(ctx.run.path(MELODY_STEM), layer_bus, sr)
        else:
            ctx.run.path(MELODY_STEM).unlink(missing_ok=True)
        master_wav = ctx.run.path(MASTER_WAV)
        tmp = temp_path_for(master_wav)
        try:
            sf.write(str(tmp), mastered.audio, sr, subtype="PCM_24")
            tmp.replace(master_wav)
        finally:
            tmp.unlink(missing_ok=True)
        write_mp3(master_wav, ctx.run.path(MASTER_MP3))

        if mastered.lufs is None:
            raise StageError("the mix is silent")
        report = MixReport(
            sample_rate=sr, frames=frames, seconds=round(frames / sr, 4), take=SELECTED,
            clips=placed, melody_layer=mode, music_lufs=music_lufs, speech_lufs=speech_lufs,
            speech_gain_db=round(gain_db, 3), duck_db=spec.sidechain_duck_db,
            melody_gain_db=None if melody_gain_db is None else round(melody_gain_db, 3),
            target_lufs=spec.target_lufs, master_gain_db=round(mastered.gain_db, 3),
            integrated_lufs=round(mastered.lufs, 3),
            true_peak_dbtp=round(mastered.true_peak_db, 3), warnings=warnings,
        )  # fmt: skip
        write_json(ctx.run.path(MIX_REPORT), report)
        for warning in warnings:
            ctx.say(f"[yellow]  {warning}[/]")
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
