"""Run directories: run IDs, manifest I/O, atomic writes, file hashing and per-run logs."""

import hashlib
import json
import logging
import os
import re
import tempfile
import unicodedata
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from speech2song import __version__
from speech2song.errors import RunNotFoundError
from speech2song.models import FileRef, HashMemo, Manifest, RunOptions

MANIFEST_NAME = "manifest.json"
COSTS_NAME = "costs.json"
LOG_NAME = "log.txt"

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
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
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


class Run:
    """A run directory plus its manifest. Mutate `manifest`, then call `save()`."""

    def __init__(self, root: Path, manifest: Manifest) -> None:
        self.root = root
        self.manifest = manifest

    @property
    def id(self) -> str:
        return self.manifest.run_id

    @property
    def costs_path(self) -> Path:
        return self.root / COSTS_NAME

    def path(self, rel: str) -> Path:
        return self.root / rel

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
        root = resolve_run_dir(runs_dir, ref)
        manifest = Manifest.model_validate_json((root / MANIFEST_NAME).read_text(encoding="utf-8"))
        return cls(root, manifest)

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
