"""The music backend interface (spec section 6, stage 6).

A backend builds a request from the arrangement (pure: its digest is the generate
stage's cache key, so a paid backend repeats a call only when the request changes),
then generates takes from it. Takes are kept exactly as generated; selecting,
converting and conforming one happens in the free `take` stage.

Take numbers are never reused within a run. A take made for an older request stays
in 06_music/ (and can be chosen with `--take N`) as long as its grid of chunk lengths
still fits the arrangement; otherwise a paid take moves to 06_music/archive/.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from speech2song.config import AppConfig, Preset
from speech2song.costs import CostLog, SpendEstimate
from speech2song.errors import S2SError
from speech2song.models import Arrangement, QuoteMelodies, RunOptions, TakeMeta


@dataclass
class Take:
    path: Path  # the audio file, as the backend wrote it
    meta: TakeMeta


@dataclass(frozen=True)
class MelodyFiles:
    """The melody stage's renders a backend may condition the music on."""

    reference: Path | None = None  # the main phrase, looped (M5 plans)
    quotes: Path | None = None  # every quote's melody (M7 plans)
    index: QuoteMelodies | None = None  # where each quote sits in `quotes`


class MusicBackend(Protocol):
    name: str
    paid: bool

    def request(
        self, arrangement: Arrangement, preset: Preset, melody: MelodyFiles
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


def take_metas(out_dir: Path) -> list[TakeMeta]:
    """The takes in 06_music/ (not the archive), by number."""
    metas = [TakeMeta.model_validate_json(path.read_text(encoding="utf-8"))
             for path in out_dir.glob("take_*.meta.json")]  # fmt: skip
    return sorted(metas, key=lambda meta: meta.take)


def next_take_number(out_dir: Path) -> int:
    """One past every take number used in this run, archived ones included."""
    paths = [*out_dir.glob("take_*.meta.json"), *out_dir.glob("archive/*/take_*.meta.json")]
    numbers = [TakeMeta.model_validate_json(p.read_text(encoding="utf-8")).take for p in paths]
    return max(numbers, default=0) + 1


def take_grid(meta: TakeMeta, run_dir: Path) -> list[int] | None:
    """The chunk lengths a take was generated with. Takes from before `grid_ms` existed
    are read from the plan in the API's response for their first version."""
    if meta.grid_ms is not None:
        return meta.grid_ms
    history = meta.params.get("history") or []
    first = history[0]["file"] if history else meta.file
    response = (run_dir / first).with_suffix(".response.json")
    if not response.exists():
        return None
    chunks = json.loads(response.read_text()).get("composition_plan", {}).get("chunks", [])
    if not chunks or any("duration_ms" not in chunk for chunk in chunks):
        return None
    return [int(chunk["duration_ms"]) for chunk in chunks]


def take_fits(meta: TakeMeta, run_dir: Path, grid: list[int], digest: str) -> bool:
    """Whether a take lines up with the current arrangement."""
    found = take_grid(meta, run_dir)
    return found == grid if found is not None else meta.request_sha256 == digest


def remove_stub_takes(out_dir: Path, run_dir: Path) -> None:
    """Stub takes are free to remake; they make way for the next generation."""
    for meta in take_metas(out_dir):
        if meta.backend == "stub":
            (run_dir / meta.file).unlink(missing_ok=True)
            (out_dir / f"take_{meta.take:03d}.meta.json").unlink(missing_ok=True)


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
