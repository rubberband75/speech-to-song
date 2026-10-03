"""Listening tests for the speech-to-song melody: WAVs to compare by ear.

Per clip: the speech alone, the melody alone, the two together, and an "illusion" take
where the speech repeats while the melody fades in (repetition is what makes speech
start to sound sung). These files are for listening only; nothing downstream uses them.
"""

from pathlib import Path

import numpy as np
import soundfile as sf

from speech2song.audio.synth import build_midi, normalize_peak, render_array, write_wav
from speech2song.manifest import Song
from speech2song.models import Melody

LISTEN_DIR = "04_listen"
MELODY_UNDER_SPEECH_DB = -8.0
ILLUSION_GAINS = (0.0, 0.35, 0.7, 1.0)  # melody level on each repeat of the speech


def _stereo(audio: np.ndarray) -> np.ndarray:
    return audio if audio.shape[1] == 2 else np.repeat(audio[:, :1], 2, axis=1)


def _mix(parts: list[tuple[np.ndarray, int, float]], length: int) -> np.ndarray:
    """Sum (audio, start sample, gain) parts into a stereo buffer of `length` frames."""
    out = np.zeros((length, 2), dtype=np.float32)
    for audio, start, gain in parts:
        end = min(length, start + len(audio))
        out[start:end] += gain * audio[: end - start]
    return out


def write_listening_set(song: Song, soundfont: Path) -> list[Path]:
    """The listening set for one length of a run, in its 04_listen/."""
    melody = Melody.model_validate_json(song.path("04_melody.json").read_text())
    out_dir = song.path(LISTEN_DIR)
    out_dir.mkdir(exist_ok=True)
    written = []
    for clip in melody.clips:
        speech, rate = sf.read(str(song.path(clip.file)), dtype="float32", always_2d=True)
        speech = _stereo(speech)
        one_loop = build_midi([clip], bpm=melody.bpm, instrument_name=melody.instrument,
                              loops=1, tail_s=0.5)  # fmt: skip
        tune = normalize_peak(render_array(one_loop, soundfont, rate), -3.0)
        phrase = round(clip.bars * 4 * 60 / melody.bpm * rate)
        under = 10 ** (MELODY_UNDER_SPEECH_DB / 20)
        repeats = len(ILLUSION_GAINS)
        takes = {
            "1_speech": speech,
            "2_melody": tune,
            "3_overlay": _mix([(speech, 0, 1.0), (tune, 0, under)], max(len(speech), len(tune))),
            "4_illusion": _mix(
                [(speech, i * phrase, 1.0) for i in range(repeats)]
                + [(tune, i * phrase, under * g) for i, g in enumerate(ILLUSION_GAINS) if g],
                (repeats - 1) * phrase + max(len(speech), len(tune)),
            ),
        }
        for name, audio in takes.items():
            path = out_dir / f"{clip.clip_id}_{name}.wav"
            level = -1.0 if name != "1_speech" else None
            write_wav(path, normalize_peak(audio, level) if level else audio, rate)
            written.append(path)
    return written
