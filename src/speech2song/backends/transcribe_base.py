"""Transcription backend interface."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from speech2song.config import AppConfig
from speech2song.errors import ConfigError
from speech2song.models import AsrResult, RunOptions

ProgressFn = Callable[[float], None]  # fraction of the audio done, 0..1


class Transcriber(Protocol):
    name: str

    def params(self) -> dict[str, Any]:
        """Settings that affect the output (they go into the stage fingerprint)."""
        ...

    def transcribe(self, audio: Path, *, progress: ProgressFn | None = None) -> AsrResult: ...


def make_transcriber(config: AppConfig, options: RunOptions) -> Transcriber:
    """Build the configured backend; per-run options override config defaults."""
    if config.transcribe_backend == "whisper":
        from speech2song.backends.transcribe_whisper import WhisperTranscriber

        return WhisperTranscriber(
            model=options.whisper_model or config.whisper_model,
            language=options.language or config.language,
            settings=config.whisper,
        )
    raise ConfigError(f"Unknown transcribe_backend: {config.transcribe_backend}")
