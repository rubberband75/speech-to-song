"""The music backend interface (spec section 6, stage 6).

A backend builds a request from the arrangement (pure: its digest is the generate
stage's cache key, so a paid backend repeats a call only when the request changes),
then generates takes from it. Takes are kept exactly as generated; selecting,
converting and conforming one happens in the free `take` stage.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from speech2song.config import AppConfig, Preset
from speech2song.costs import CostLog, SpendEstimate
from speech2song.errors import S2SError
from speech2song.models import Arrangement, RunOptions, TakeMeta


@dataclass
class Take:
    path: Path  # the audio file, as the backend wrote it
    meta: TakeMeta


class MusicBackend(Protocol):
    name: str
    paid: bool

    def request(
        self, arrangement: Arrangement, preset: Preset, melody_reference: Path | None
    ) -> dict[str, Any]:
        """Everything the backend would send or use to generate (JSON-serializable)."""
        ...

    def estimate(self, request: dict[str, Any], takes: int) -> list[SpendEstimate]: ...

    def generate(
        self,
        request: dict[str, Any],
        out_dir: Path,
        takes: int,
        run_dir: Path,
        say: Callable[[str], None] = print,
        fresh: bool = False,  # paid backends: make new takes even if this request has some
    ) -> list[Take]: ...


def backend_name(config: AppConfig, options: RunOptions) -> str:
    return options.music_backend or config.music_backend


def make_backend(
    config: AppConfig, options: RunOptions, cost_log: CostLog | None = None, run_id: str = ""
) -> MusicBackend:
    name = backend_name(config, options)
    if name == "stub":
        from speech2song.backends.music_stub import StubBackend

        return StubBackend()
    if name == "elevenlabs":
        from speech2song.backends.music_elevenlabs import ElevenLabsBackend

        return ElevenLabsBackend(config, cost_log, run_id)
    raise S2SError(f"Unknown music backend {name!r} (use stub or elevenlabs).")
