"""Local transcription with faster-whisper (word timestamps, VAD)."""

import functools
import logging
import os
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from speech2song.audio.io import decode_mono
from speech2song.backends.transcribe_base import ProgressFn
from speech2song.config import WhisperConfig
from speech2song.models import AsrRepair, AsrResult, AsrSegment, AsrWord

log = logging.getLogger(__name__)

WHISPER_SAMPLE_RATE = 16000
REPAIR_MAX_EXTEND_S = 30.5  # at most one batched chunk on either side of a gap


class _WordLike(Protocol):
    start: float
    end: float
    word: str
    probability: float


class _SegmentLike(Protocol):
    id: int
    start: float
    end: float
    text: str
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    words: list[_WordLike] | None


def physical_cores() -> int:
    """Best guess without extra dependencies: logical CPUs / 2 (SMT), at least 1."""
    return max(1, (os.cpu_count() or 2) // 2)


def segments_to_asr(
    segments: Iterable[_SegmentLike],
    *,
    language: str,
    language_probability: float | None,
    duration_s: float,
    backend: str,
    model: str,
    params: dict[str, Any],
    audio: str,
) -> AsrResult:
    """Convert faster-whisper segments, cleaning word timings.

    Words are stripped, empty ones dropped, and times made non-decreasing and
    non-overlapping (start >= previous end, end >= start), rounded to milliseconds.
    """
    out: list[AsrSegment] = []
    prev_end = 0.0
    for seg in segments:
        words: list[AsrWord] = []
        for word in seg.words or []:
            text = word.word.strip()
            if not text:
                continue
            start = max(round(word.start, 3), prev_end)
            end = max(round(word.end, 3), start)
            words.append(AsrWord(w=text, start=start, end=end, conf=round(word.probability, 3)))
            prev_end = end
        out.append(
            AsrSegment(
                id=seg.id,
                start=round(seg.start, 3),
                end=round(seg.end, 3),
                text=seg.text.strip(),
                avg_logprob=seg.avg_logprob,
                no_speech_prob=seg.no_speech_prob,
                compression_ratio=seg.compression_ratio,
                words=words,
            )
        )
    return AsrResult(
        backend=backend,
        model=model,
        params=params,
        audio=audio,
        language=language,
        language_probability=language_probability,
        duration_s=round(duration_s, 3),
        segments=out,
    )


def find_holes(
    words: Sequence[AsrWord],
    speech: Sequence[tuple[float, float]],
    duration_s: float,
    *,
    min_gap_s: float,
    min_speech_s: float,
) -> list[tuple[float, float, float]]:
    """(start, end, speech seconds) of word gaps, including either end, that contain
    detected speech. `speech` holds (start, end) seconds from a voice activity detector."""
    edges = [0.0, *(t for w in words for t in (w.start, w.end)), duration_s]
    holes = []
    for lo, hi in zip(edges[::2], edges[1::2], strict=True):
        if hi - lo < min_gap_s:
            continue
        speech_s = sum(max(0.0, min(hi, end) - max(lo, start)) for start, end in speech)
        if speech_s >= min_speech_s:
            holes.append((lo, hi, round(speech_s, 3)))
    return holes


def repair_windows(
    segments: Sequence[AsrSegment],
    holes: Sequence[tuple[float, float, float]],
    *,
    max_extend_s: float,
) -> list[AsrRepair]:
    """Widen each hole to the whole segments that touch it, then merge overlaps.

    The words next to a dropped chunk are often damaged too (truncated, badly timed),
    and segment boundaries fall in pauses, so re-transcribing whole neighbouring
    segments avoids cutting the audio mid-sentence.
    """
    windows: list[AsrRepair] = []
    for gap_start, gap_end, speech_s in holes:
        start, end = gap_start, gap_end
        for seg in segments:
            if not seg.words:
                continue
            touches_start = seg.words[0].start <= gap_start <= seg.words[-1].end
            touches_end = seg.words[0].start <= gap_end <= seg.words[-1].end
            if touches_start and gap_start - seg.start <= max_extend_s:
                start = min(start, seg.start)
            if touches_end and seg.end - gap_end <= max_extend_s:
                end = max(end, seg.end)
        windows.append(
            AsrRepair(gap_start=gap_start, gap_end=gap_end, speech_s=speech_s, start=start,
                      end=end)
        )  # fmt: skip
    merged: list[AsrRepair] = []
    for window in sorted(windows, key=lambda w: w.start):
        if merged and window.start <= merged[-1].end:
            last = merged[-1]
            merged[-1] = last.model_copy(
                update={
                    "end": max(last.end, window.end),
                    "gap_end": max(last.gap_end, window.gap_end),
                    "speech_s": last.speech_s + window.speech_s,
                }
            )
        else:
            merged.append(window)
    return merged


def apply_repairs(
    result: AsrResult, repaired: Sequence[tuple[AsrRepair, Sequence[AsrSegment]]]
) -> AsrResult:
    """Replace every word inside each window with the re-transcribed words (absolute
    times). A word belongs to a window when its midpoint does."""

    def inside(word: AsrWord, window: AsrRepair) -> bool:
        return window.start <= (word.start + word.end) / 2 <= window.end

    segments = list(result.segments)
    repairs = []
    for window, new_segments in repaired:
        kept, removed = [], 0
        for seg in segments:
            remaining = [w for w in seg.words if not inside(w, window)]
            removed += len(seg.words) - len(remaining)
            if remaining or (not seg.words and not window.start <= seg.start <= window.end):
                kept.append(seg.model_copy(update={"words": remaining}))
        added = 0
        for seg in new_segments:
            words = []
            for word in seg.words:
                if inside(word, window):
                    start = min(max(word.start, window.start), window.end)
                    end = max(start, min(word.end, window.end))
                    words.append(word.model_copy(update={"start": start, "end": end}))
            if words:
                kept.append(seg.model_copy(update={"words": words, "repaired": True}))
                added += len(words)
        segments = kept
        repairs.append(window.model_copy(update={"words_before": removed, "words_after": added}))
    segments.sort(key=lambda seg: seg.words[0].start if seg.words else seg.start)
    prev_end = 0.0
    numbered = []
    for number, seg in enumerate(segments, start=1):
        words = []
        for word in seg.words:  # keep words non-overlapping across window edges
            start = max(word.start, prev_end)
            words.append(word.model_copy(update={"start": start, "end": max(word.end, start)}))
            prev_end = words[-1].end
        numbered.append(seg.model_copy(update={"id": number, "words": words}))
    return result.model_copy(update={"segments": numbered, "repairs": repairs})


def shift_segments(segments: Sequence[AsrSegment], offset: float) -> list[AsrSegment]:
    def moved(t: float) -> float:
        return round(t + offset, 3)

    return [
        seg.model_copy(
            update={
                "start": moved(seg.start),
                "end": moved(seg.end),
                "words": [
                    w.model_copy(update={"start": moved(w.start), "end": moved(w.end)})
                    for w in seg.words
                ],
            }
        )
        for seg in segments
    ]


class WhisperTranscriber:
    name = "whisper"

    def __init__(self, model: str, language: str | None, settings: WhisperConfig) -> None:
        self.model = model
        self.language = language
        self.settings = settings

    def params(self) -> dict[str, Any]:
        s = self.settings
        return {
            "backend": "faster-whisper",
            "model": self.model,
            "language": self.language,
            "compute_type": s.compute_type,
            "beam_size": s.beam_size,
            "vad_filter": s.vad_filter,
            "batch_size": s.batch_size,
            "condition_on_previous_text": s.condition_on_previous_text,
            "repair_gaps": s.repair_gaps,
            "repair_min_gap_s": s.repair_min_gap_s,
            "repair_min_speech_s": s.repair_min_speech_s,
        }

    def _device(self) -> tuple[str, str]:
        import ctranslate2

        device = self.settings.device
        if device == "auto":
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        compute_type = self.settings.compute_type
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"
        return device, compute_type

    def transcribe(self, audio: Path, *, progress: ProgressFn | None = None) -> AsrResult:
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        # Decode ourselves: faster-whisper's own decoder relies on a PyAV API that
        # PyAV 19 removed, and ffmpeg/soxr is what the rest of the pipeline uses.
        samples = decode_mono(audio, WHISPER_SAMPLE_RATE)
        device, compute_type = self._device()
        threads = self.settings.cpu_threads or physical_cores()
        log.info("Loading faster-whisper %s (%s, %s, %d threads)", self.model, device, compute_type,
                 threads)  # fmt: skip
        model = WhisperModel(self.model, device=device, compute_type=compute_type,
                             cpu_threads=threads)  # fmt: skip
        options: dict[str, Any] = {
            "language": self.language,
            "beam_size": self.settings.beam_size,
            "word_timestamps": True,
            "vad_filter": self.settings.vad_filter,
            "condition_on_previous_text": self.settings.condition_on_previous_text,
        }
        if self.settings.batch_size > 0:
            pipeline = BatchedInferencePipeline(model)
            segments, info = pipeline.transcribe(
                samples, batch_size=self.settings.batch_size, **options
            )
        else:
            segments, info = model.transcribe(samples, **options)
        collected = []
        for segment in segments:  # a generator: decoding happens while iterating
            collected.append(segment)
            if progress is not None and info.duration > 0:
                progress(min(1.0, segment.end / info.duration))
        convert = functools.partial(
            segments_to_asr,
            language=info.language,
            language_probability=info.language_probability,
            backend="faster-whisper",
            model=self.model,
            params=self.params() | {"device": device, "resolved_compute_type": compute_type},
            audio=audio.name,
        )
        result = convert(collected, duration_s=info.duration)
        if self.settings.repair_gaps:
            result = self._repair(model, samples, result, convert, options | {
                "language": info.language})  # fmt: skip
        return result

    def _repair(
        self,
        model: Any,
        samples: np.ndarray,
        result: AsrResult,
        convert: Callable[..., AsrResult],
        options: dict[str, Any],
    ) -> AsrResult:
        """Re-transcribe around word gaps where the voice activity detector hears speech."""
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        rate = WHISPER_SAMPLE_RATE
        vad = VadOptions(min_silence_duration_ms=500, speech_pad_ms=0)
        speech = [(t["start"] / rate, t["end"] / rate) for t in get_speech_timestamps(samples, vad)]
        holes = find_holes(
            result.words(),
            speech,
            result.duration_s,
            min_gap_s=self.settings.repair_min_gap_s,
            min_speech_s=self.settings.repair_min_speech_s,
        )
        if not holes:
            return result
        repaired = []
        for window in repair_windows(result.segments, holes, max_extend_s=REPAIR_MAX_EXTEND_S):
            log.warning(
                "No words for %.1f-%.1f s although %.1f s of it is speech; "
                "re-transcribing %.1f-%.1f s",
                window.gap_start, window.gap_end, window.speech_s, window.start, window.end,
            )  # fmt: skip
            part = samples[int(window.start * rate) : int(window.end * rate)]
            segments, _ = model.transcribe(part, **options)  # sequential decoding
            local = convert(list(segments), duration_s=window.end - window.start)
            repaired.append((window, shift_segments(local.segments, window.start)))
        return apply_repairs(result, repaired)
