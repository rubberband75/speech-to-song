"""MIDI for the speech melody (pretty_midi) and rendering with the fluidsynth CLI."""

import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import soundfile as sf

from speech2song.errors import S2SError
from speech2song.manifest import temp_path_for
from speech2song.models import ClipMelody

# General MIDI program and melody velocity for each preset `render_instrument`.
INSTRUMENTS = {
    "soft_piano": (0, 56),
    "piano": (0, 80),
    "electric_piano": (4, 64),
    "celesta": (8, 60),
    "bells": (14, 60),
    "strings": (48, 60),
    "pad": (89, 64),
}
CHORD_VELOCITY = 0.55  # chords sit under the melody
CHORD_LOW = 48  # chord roots go in C3-B3
SOUNDFONTS = (
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/usr/share/sounds/sf2/default-GM.sf2",
    "/usr/share/soundfonts/FluidR3_GM.sf2",
    "/usr/share/soundfonts/default.sf2",
)
RENDER_TAIL_S = 1.5


def find_soundfont(configured: Path | None) -> Path:
    if configured is not None:
        if not configured.is_file():
            raise S2SError(f"Soundfont not found: {configured}")
        return configured
    for candidate in SOUNDFONTS:
        if Path(candidate).is_file():
            return Path(candidate)
    raise S2SError(
        "No General MIDI soundfont found; install fluid-soundfont-gm or set `soundfont` "
        "in config.yaml."
    )


def instrument(name: str) -> tuple[int, int]:
    if name not in INSTRUMENTS:
        raise S2SError(f"Unknown render_instrument {name!r}; choose from {sorted(INSTRUMENTS)}")
    return INSTRUMENTS[name]


def build_midi(
    phrases: Sequence[ClipMelody],
    *,
    bpm: float,
    instrument_name: str,
    loops: int,
    gap_bars: int = 1,
    chords: bool = True,
    tail_s: float = RENDER_TAIL_S,
):  # -> pretty_midi.PrettyMIDI
    """Each phrase played `loops` times in a row, with `gap_bars` of rest between
    phrases; melody and (quieter) block chords on the same instrument."""
    import pretty_midi

    program, velocity = instrument(instrument_name)
    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    melody = pretty_midi.Instrument(program=program, name="melody")
    harmony = pretty_midi.Instrument(program=program, name="chords")
    beat = 60.0 / bpm
    bar_cursor = 0
    for phrase in phrases:
        for loop in range(loops):
            offset = (bar_cursor + loop * phrase.bars) * 4
            for note in phrase.notes:
                start = (offset + note.start_beat) * beat
                melody.notes.append(
                    pretty_midi.Note(
                        velocity=round(velocity * note.velocity / 100),
                        pitch=note.midi,
                        start=start,
                        end=start + note.beats * beat,
                    )
                )
            if chords:
                for chord in phrase.chords:
                    start = (offset + chord.bar * 4) * beat
                    root = chord.pitch_classes[0]
                    for pc in chord.pitch_classes:
                        pitch = CHORD_LOW + root + (pc - root) % 12
                        harmony.notes.append(
                            pretty_midi.Note(
                                velocity=round(velocity * CHORD_VELOCITY),
                                pitch=pitch,
                                start=start,
                                end=start + 4 * beat,
                            )
                        )
        bar_cursor += phrase.bars * loops + gap_bars
    end = max((n.end for n in melody.notes + harmony.notes), default=0.0)
    # A silent controller event after the last note makes fluidsynth render the release.
    harmony.control_changes.append(pretty_midi.ControlChange(number=7, value=100,
                                                             time=end + tail_s))  # fmt: skip
    midi.instruments = [melody, harmony]
    return midi


def write_midi(midi, path: Path) -> None:  # midi: pretty_midi.PrettyMIDI
    tmp = temp_path_for(path)
    try:
        midi.write(str(tmp))
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def render_array(midi, soundfont: Path, sample_rate: int = 44100) -> np.ndarray:
    """Render with fluidsynth; (frames, 2) float32, trimmed after the release tail."""
    fluidsynth = shutil.which("fluidsynth")
    if fluidsynth is None:
        raise S2SError("`fluidsynth` was not found on PATH; install it to render the melody.")
    with tempfile.TemporaryDirectory() as tmpdir:
        mid = Path(tmpdir) / "in.mid"
        raw = Path(tmpdir) / "out.wav"
        midi.write(str(mid))
        args = [
            fluidsynth, "-ni", "-q", "-R", "1", "-C", "0", "-g", "0.6",
            "-r", str(sample_rate), "-T", "wav", "-F", str(raw), str(soundfont), str(mid),
        ]  # fmt: skip
        try:
            subprocess.run(args, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as exc:
            raise S2SError(f"fluidsynth failed: {exc.stderr.strip()[-300:]}") from exc
        audio, rate = sf.read(str(raw), dtype="float32", always_2d=True)
    if rate != sample_rate:
        raise S2SError(f"fluidsynth rendered at {rate} Hz instead of {sample_rate} Hz")
    return audio[: min(len(audio), round(midi.get_end_time() * rate))]


def normalize_peak(audio: np.ndarray, peak_db: float) -> np.ndarray:
    peak = float(np.abs(audio).max()) if len(audio) else 0.0
    return audio * np.float32(10 ** (peak_db / 20) / peak) if peak > 0 else audio


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """Atomically write a 32-bit float WAV with reproducible bytes.

    libsndfile stamps float WAVs with a PEAK chunk holding the current time, so the
    same samples hash differently on every write and needlessly invalidate the stages
    downstream. scipy writes plain IEEE-float WAVs, which soundfile reads back exactly.
    """
    from scipy.io import wavfile

    tmp = temp_path_for(path)
    try:
        wavfile.write(str(tmp), sample_rate, np.ascontiguousarray(audio, dtype=np.float32))
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def render(
    midi, wav_path: Path, soundfont: Path, sample_rate: int = 44100, peak_db: float = -3.0
) -> float:
    """Render to a float WAV, peak-normalized to `peak_db` (synthesized music, never
    speech). Returns the duration in seconds."""
    audio = normalize_peak(render_array(midi, soundfont, sample_rate), peak_db)
    write_wav(wav_path, audio, sample_rate)
    return len(audio) / sample_rate


def peak_dbfs(path: Path) -> float:
    audio, _ = sf.read(str(path), dtype="float32")
    return float(20 * np.log10(np.abs(audio).max() + 1e-12))
