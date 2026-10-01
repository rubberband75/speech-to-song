"""Synthetic test material: generated signals and fake ASR output. No real speech or music."""

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from speech2song.models import AsrRef, AsrResult, AsrSegment, AsrWord, Sentence, Transcript, Word

SR = 44100
FIXTURES = Path(__file__).parent


def sine(freq: float, seconds: float, sr: int = SR, amp: float = 0.5) -> np.ndarray:
    t = np.arange(round(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def sweep(f0: float, f1: float, seconds: float, sr: int = SR, amp: float = 0.5) -> np.ndarray:
    """Exponential sine sweep from f0 to f1 Hz."""
    t = np.arange(round(seconds * sr)) / sr
    k = np.log(f1 / f0) / seconds
    return (amp * np.sin(2 * np.pi * f0 * (np.exp(k * t) - 1) / k)).astype(np.float32)


def noise_burst(seconds: float, sr: int = SR, amp: float = 0.3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (amp * rng.standard_normal(round(seconds * sr))).clip(-1, 1).astype(np.float32)


def speechlike(
    syllables: list[tuple[float, float, float]], gap_s: float = 0.12, sr: int = SR
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Harmonic 'syllables' (f0 glides with smooth envelopes) separated by silence.

    `syllables` holds (f0_start_hz, f0_end_hz, seconds). Returns the mono signal and the
    (start, end) time of each syllable, which serve as ground-truth word times.
    """
    pieces: list[np.ndarray] = []
    times: list[tuple[float, float]] = []
    t = 0.0
    for f_start, f_end, seconds in syllables:
        n = round(seconds * sr)
        f0 = np.linspace(f_start, f_end, n)
        phase = 2 * np.pi * np.cumsum(f0) / sr
        tone = sum(np.sin(h * phase) / h for h in range(1, 6))
        envelope = np.sin(np.linspace(0, np.pi, n)) ** 2
        pieces.append((0.3 * tone * envelope).astype(np.float32))
        pieces.append(np.zeros(round(gap_s * sr), dtype=np.float32))
        times.append((t, t + n / sr))
        t += n / sr + gap_s
    return np.concatenate(pieces), times


def write_wav(path: Path, data: np.ndarray, sr: int = SR, subtype: str = "FLOAT") -> Path:
    sf.write(str(path), data, sr, subtype=subtype)
    return path


def make_video(path: Path, seconds: float = 1.0) -> Path:
    """A tiny MP4 with a test-pattern video stream and a 48 kHz stereo sine audio stream."""
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"testsrc=size=64x64:rate=10:duration={seconds}",
            "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={seconds}",
            "-ac", "2", "-c:v", "mpeg4", "-c:a", "aac", "-shortest", str(path),
        ],
        check=True,
    )  # fmt: skip
    return path


def fake_asr_words(
    text: str,
    *,
    start: float = 0.5,
    word_s: float = 0.3,
    gap_s: float = 0.05,
    sentence_gap_s: float = 0.6,
    conf: float = 0.9,
) -> list[AsrWord]:
    """Timed words for `text`, one per whitespace token, with pauses after sentences."""
    words = []
    t = start
    for token in text.split():
        words.append(AsrWord(w=token, start=round(t, 3), end=round(t + word_s, 3), conf=conf))
        t += word_s + (sentence_gap_s if token.endswith((".", "?", "!")) else gap_s)
    return words


def asr_result(words: list[AsrWord], duration_s: float | None = None) -> AsrResult:
    end = words[-1].end if words else 0.0
    return AsrResult(
        backend="fake",
        model="fake-model",
        audio="01_clean.wav",
        language="en",
        language_probability=1.0,
        duration_s=duration_s if duration_s is not None else end + 1.0,
        segments=[
            AsrSegment(
                id=0,
                start=words[0].start if words else 0.0,
                end=end,
                text=" ".join(w.w for w in words),
                words=words,
            )
        ],
    )


def transcript_from(
    sentences: list[list[tuple[str, float, float]]], duration_s: float | None = None
) -> Transcript:
    """A Transcript from sentences given as lists of (word, start, end)."""
    words: list[Word] = []
    rows: list[Sentence] = []
    for number, spec in enumerate(sentences, start=1):
        first = len(words)
        words += [Word(w=w, start=a, end=b, conf=0.9, flag="matched") for w, a, b in spec]
        rows.append(
            Sentence(
                id=number,
                text=" ".join(w for w, _, _ in spec),
                start=spec[0][1],
                end=spec[-1][2],
                word_start=first,
                word_end=len(words),
                source="official",
                avg_conf=0.9,
            )
        )
    end = duration_s if duration_s is not None else (words[-1].end + 1.0 if words else 1.0)
    return Transcript(
        source="talk.wav",
        duration_s=end,
        language="en",
        words=words,
        sentences=rows,
        official_transcript_used=True,
        asr=AsrRef(backend="fake", model="fake"),
    )


def synthetic_talk(
    n_sentences: int = 8,
    words_per_sentence: int = 8,
    *,
    word_s: float = 0.35,
    gap_s: float = 0.1,
    pause_s: float = 0.8,
    lead_s: float = 0.6,
    jitter_s: float = 0.04,
    seed: int = 7,
) -> tuple[np.ndarray, Transcript, list[tuple[float, float]]]:
    """Mono 'speech' of syllable-words with pauses between sentences.

    Returns the audio, a transcript whose word times are jittered by up to `jitter_s`
    (like real ASR timestamps), and the true silent pauses before and after each
    sentence as (start, end) seconds, for checking cut points.
    """
    rng = np.random.default_rng(seed)
    pieces = [np.zeros(round(lead_s * SR), dtype=np.float32)]
    t = lead_s
    sentences = []
    silences = [(0.0, lead_s)]
    for s in range(n_sentences):
        syllables = [(150 + 12 * k, 190 - 5 * k, word_s) for k in range(words_per_sentence)]
        voice, times = speechlike(syllables, gap_s=gap_s)
        voice = voice[: round((times[-1][1]) * SR)]  # drop the trailing gap
        pieces.append(voice)
        spec = []
        for k, (a, b) in enumerate(times):
            ja, jb = rng.uniform(-jitter_s, jitter_s, size=2)
            word = f"s{s + 1}w{k + 1}" + ("." if k == len(times) - 1 else "")
            spec.append((word, round(t + a + ja, 3), round(t + b + jb, 3)))
        sentences.append(spec)
        t += len(voice) / SR
        pieces.append(np.zeros(round(pause_s * SR), dtype=np.float32))
        silences.append((t, t + pause_s))
        t += pause_s
    audio = np.concatenate(pieces)
    return audio, transcript_from(sentences, duration_s=len(audio) / SR), silences
