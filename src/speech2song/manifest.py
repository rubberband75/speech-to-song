"""Run directories: run IDs, manifest I/O, atomic writes, file hashing and per-run logs.

A run holds one talk and songs of several lengths (M8). The talk's files (source audio,
transcript) sit at the run root; each length's files sit in <run>/<length>/, and every
path inside them (clip files, takes) is relative to that folder.
"""

import hashlib
import json
import logging
import os
import re
import unicodedata
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from speech2song import __version__
from speech2song.config import SUMMARY
from speech2song.errors import RunNotFoundError
from speech2song.models import (
    FileRef,
    HashMemo,
    Manifest,
    Melody,
    RunOptions,
    SharedMusic,
    SongOptions,
    SongState,
    StageRecord,
)

MANIFEST_NAME = "manifest.json"
COSTS_NAME = "costs.json"
LOG_NAME = "log.txt"
# Before M8 a run held one song at its root; it becomes the run's summary length.
SONG_STAGES = ("select", "clips", "melody", "arc", "arrange", "generate", "take", "mix")
SONG_FILE_PREFIXES = ("03_", "04_", "05_", "06_", "07_")
SONG_DIRS = ("clips",)
SONG_OPTIONS = ("clips", "take")

log = logging.getLogger(__name__)


def slugify(text: str, max_len: int = 32) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")
    return slug[:max_len].rstrip("-") or "input"


def make_run_id(input_path: Path, now: datetime) -> str:
    return f"{now:%Y%m%d-%H%M%S}-{slugify(input_path.stem)}"


def atomic_write_text(path: Path, text: str) -> None:
    """Write through a temp file in the same directory, then rename over the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create of a unique name (not mkstemp, whose files are always 0600).
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with tmp.open("x", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_json(path: Path, data: BaseModel | dict[str, Any] | list[Any]) -> None:
    if isinstance(data, BaseModel):
        text = data.model_dump_json(indent=2)
    else:
        text = json.dumps(data, indent=2, ensure_ascii=False)
    atomic_write_text(path, text + "\n")


def temp_path_for(path: Path) -> Path:
    """Sibling temp path that keeps the suffix (tools like ffmpeg pick formats from it)."""
    return path.with_name(f".{path.stem}.tmp{path.suffix}")


def sha256_file(path: Path) -> str:
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def list_run_dirs(runs_dir: Path) -> list[Path]:
    """Run directories, oldest first (run IDs start with a timestamp)."""
    if not runs_dir.is_dir():
        return []
    return sorted(d for d in runs_dir.iterdir() if (d / MANIFEST_NAME).is_file())


def resolve_run_dir(runs_dir: Path, ref: str | None) -> Path:
    """Find a run by ID, unique ID prefix, path, or 'latest' (the default)."""
    runs = list_run_dirs(runs_dir)
    if ref is None or ref == "latest":
        if not runs:
            raise RunNotFoundError(
                f"No runs in {runs_dir} yet. Start one with `speech2song ingest INPUT`."
            )
        return runs[-1]
    as_path = Path(ref)
    if (as_path / MANIFEST_NAME).is_file():
        return as_path
    if (runs_dir / ref / MANIFEST_NAME).is_file():
        return runs_dir / ref
    matches = [d for d in runs if d.name.startswith(ref)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise RunNotFoundError(f"No run in {runs_dir} matches {ref!r}.")
    names = ", ".join(d.name for d in matches)
    raise RunNotFoundError(f"{ref!r} matches several runs: {names}. Use a longer prefix.")


def migrate_v1(root: Path, data: dict[str, Any]) -> dict[str, Any]:
    """A schema 1 manifest (one song at the run root) as schema 2: the song becomes the
    run's summary length. Its files move into summary/ unchanged (the paths inside them
    are relative to the song's folder, so they stay valid), and so do its stage records
    and options; the hash memo follows the moved files (same inode and mtime)."""
    song_dir = root / SUMMARY
    memo = data.setdefault("hash_memo", {})
    for item in sorted(root.iterdir()):
        if not (item.name.startswith(SONG_FILE_PREFIXES) or item.name in SONG_DIRS):
            continue
        target = song_dir / item.name
        if target.exists():  # left by an interrupted move: never overwrite
            continue
        song_dir.mkdir(exist_ok=True)
        folder = item.is_dir()
        files = [p for p in item.rglob("*") if p.is_file()] if folder else [item]
        keys = {p: str(p.resolve()) for p in files}
        item.rename(target)
        for path, key in keys.items():
            if key in memo:
                moved = target / path.relative_to(item) if folder else target
                memo[str(moved.resolve())] = memo.pop(key)
    stages = data.setdefault("stages", {})
    song_stages = {name: stages.pop(name) for name in list(stages) if name in SONG_STAGES}
    options = data.setdefault("options", {})
    song_options = {k: options.pop(k) for k in SONG_OPTIONS if k in options}
    song_options = {k: v for k, v in song_options.items() if v is not None}
    if song_stages or song_options or song_dir.exists():
        data.setdefault("songs", {})[SUMMARY] = {"options": song_options, "stages": song_stages}
    data["schema_version"] = 2
    data["length"] = SUMMARY
    return data


def shared_from_melody(anchor: str, melody: Melody) -> SharedMusic:
    return SharedMusic(anchor=anchor, bpm=melody.bpm, key=melody.key, tonic=melody.tonic,
                       mode=melody.mode, key_source=melody.key_source,
                       key_confidence=melody.key_confidence,
                       tuning_offset=melody.tuning_offset,
                       octave_shift=melody.octave_shift)  # fmt: skip


class Song:
    """One length of a run: its folder (<run>/<length>/) and its state in the manifest."""

    def __init__(self, run: "Run", length: str) -> None:
        self.run = run
        self.length = length

    @property
    def root(self) -> Path:
        return self.run.root / self.length

    def path(self, rel: str) -> Path:
        return self.root / rel

    @property
    def state(self) -> SongState:
        """The length's state, created when first written to."""
        return self.run.manifest.songs.setdefault(self.length, SongState())

    @property
    def stages(self) -> dict[str, StageRecord]:
        """The length's stage records (read only: empty for a length not made yet)."""
        state = self.run.manifest.songs.get(self.length)
        return state.stages if state is not None else {}

    @property
    def options(self) -> SongOptions:
        state = self.run.manifest.songs.get(self.length)
        return state.options if state is not None else SongOptions()

    def set_options(self, **changes: Any) -> None:
        """Apply sticky choices (None means 'keep')."""
        updates = {key: value for key, value in changes.items() if value is not None}
        if updates:
            self.state.options = self.state.options.model_copy(update=updates)


class Run:
    """A run directory plus its manifest. Mutate `manifest`, then call `save()`."""

    def __init__(self, root: Path, manifest: Manifest) -> None:
        self.root = root
        self.manifest = manifest
        self.migrated = False  # opened from a schema 1 manifest and moved to summary/

    @property
    def id(self) -> str:
        return self.manifest.run_id

    @property
    def costs_path(self) -> Path:
        return self.root / COSTS_NAME

    def path(self, rel: str) -> Path:
        return self.root / rel

    def song(self, length: str | None = None) -> Song:
        """A length of this run (default: the one commands act on)."""
        return Song(self, length or self.manifest.length)

    def made_lengths(self) -> list[str]:
        """Lengths that have stages or a folder, in the order they were first made."""
        return [name for name, state in self.manifest.songs.items()
                if state.stages or (self.root / name).is_dir()]  # fmt: skip

    @classmethod
    def create(
        cls,
        runs_dir: Path,
        input_path: Path,
        *,
        preset: str,
        transcript: Path | None = None,
        options: RunOptions | None = None,
        settings: dict[str, Any] | None = None,
        now: datetime | None = None,
        length: str = SUMMARY,
    ) -> "Run":
        now = now or datetime.now().astimezone()
        base_id = make_run_id(input_path, now)
        run_id, n = base_id, 1
        while (runs_dir / run_id).exists():
            n += 1
            run_id = f"{base_id}-{n}"
        root = runs_dir / run_id
        root.mkdir(parents=True)
        manifest = Manifest(
            run_id=run_id,
            created_at=now,
            tool_version=__version__,
            input=FileRef(path=str(input_path.resolve()), sha256="", size=0),
            preset=preset,
            options=options or RunOptions(),
            length=length,
            settings=settings or {},
        )
        run = cls(root, manifest)
        manifest.input = run.file_ref(input_path)
        if transcript is not None:
            manifest.transcript = run.file_ref(transcript)
        run.save()
        write_json(run.costs_path, [])
        return run

    @classmethod
    def open(cls, runs_dir: Path, ref: str | None = None) -> "Run":
        """Open a run. A run from before M8 is moved to the M8 layout first: its song
        becomes the summary length, which then also anchors the run's other lengths."""
        root = resolve_run_dir(runs_dir, ref)
        data = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
        migrated = data.get("schema_version", 1) < 2
        if migrated:
            data = migrate_v1(root, data)
        run = cls(root, Manifest.model_validate(data))
        if migrated:
            run.migrated = SUMMARY in run.manifest.songs  # it had a song, now the summary
            melody = run.song(SUMMARY).path("04_melody.json")
            if melody.is_file():
                run.manifest.shared = shared_from_melody(
                    SUMMARY, Melody.model_validate_json(melody.read_text(encoding="utf-8"))
                )
            run.save()
        return run

    def save(self) -> None:
        memo = self.manifest.hash_memo
        self.manifest.hash_memo = {k: v for k, v in memo.items() if Path(k).exists()}
        write_json(self.root / MANIFEST_NAME, self.manifest)

    def hash_file(self, path: Path) -> tuple[str, int]:
        """(sha256, size), reusing the memo while inode, size and mtime are unchanged."""
        real = path.resolve()
        st = real.stat()
        key = str(real)
        memo = self.manifest.hash_memo.get(key)
        if (
            memo is not None
            and memo.ino == st.st_ino
            and memo.size == st.st_size
            and memo.mtime_ns == st.st_mtime_ns
        ):
            return memo.sha256, st.st_size
        sha = sha256_file(real)
        self.manifest.hash_memo[key] = HashMemo(
            ino=st.st_ino, size=st.st_size, mtime_ns=st.st_mtime_ns, sha256=sha
        )
        return sha, st.st_size

    def file_ref(self, path: Path) -> FileRef:
        """Run files get run-relative paths (symlinks keep their own name)."""
        sha, size = self.hash_file(path)
        try:
            shown = path.absolute().relative_to(self.root.absolute()).as_posix()
        except ValueError:
            shown = str(path.resolve())
        return FileRef(path=shown, sha256=sha, size=size)

    @contextmanager
    def logging_to_file(self, level: int = logging.INFO) -> Iterator[None]:
        """Mirror the package's log records into the run's log.txt for one command."""
        handler = logging.FileHandler(self.root / LOG_NAME, encoding="utf-8")
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger = logging.getLogger("speech2song")
        logger.addHandler(handler)
        try:
            yield
        finally:
            logger.removeHandler(handler)
            handler.close()
