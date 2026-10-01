import json
import logging
from datetime import datetime
from pathlib import Path

import pytest

from speech2song import manifest as manifest_mod
from speech2song.errors import RunNotFoundError
from speech2song.manifest import (
    LOG_NAME,
    Run,
    atomic_write_text,
    make_run_id,
    sha256_file,
    slugify,
)

NOW = datetime(2026, 9, 30, 23, 15, 0).astimezone()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("My Talk", "my-talk"),
        ("Café Ünïcödé — talk!", "cafe-unicode-talk"),
        ("***", "input"),
        ("a" * 40, "a" * 32),
        ("abc def ghi jkl mno pqr stu vwx-yz", "abc-def-ghi-jkl-mno-pqr-stu-vwx"),
    ],
)
def test_slugify(text: str, expected: str) -> None:
    assert slugify(text) == expected


def test_run_id_format() -> None:
    assert make_run_id(Path("/x/Come Home.mp3"), NOW) == "20260930-231500-come-home"


def test_create_writes_manifest_and_empty_costs(tmp_path: Path, input_file: Path) -> None:
    run = Run.create(tmp_path / "runs", input_file, preset="p", now=NOW)
    assert run.root == tmp_path / "runs" / "20260930-231500-my-talk"
    data = json.loads((run.root / "manifest.json").read_text())
    assert data["run_id"] == run.id
    assert data["input"]["path"] == str(input_file.resolve())
    assert data["input"]["sha256"] == sha256_file(input_file)
    assert json.loads(run.costs_path.read_text()) == []


def test_create_suffixes_colliding_ids(tmp_path: Path, input_file: Path) -> None:
    first = Run.create(tmp_path / "runs", input_file, preset="p", now=NOW)
    second = Run.create(tmp_path / "runs", input_file, preset="p", now=NOW)
    assert second.id == f"{first.id}-2"


def test_open_by_id_prefix_path_and_latest(tmp_path: Path, input_file: Path) -> None:
    runs_dir = tmp_path / "runs"
    older = Run.create(runs_dir, input_file, preset="p", now=NOW)
    newer = Run.create(runs_dir, input_file, preset="p", now=NOW.replace(hour=23, minute=59))
    assert Run.open(runs_dir, older.id).id == older.id
    assert Run.open(runs_dir, "20260930-2359").id == newer.id
    assert Run.open(runs_dir, str(older.root)).id == older.id
    assert Run.open(runs_dir, "latest").id == newer.id
    assert Run.open(runs_dir, None).id == newer.id
    with pytest.raises(RunNotFoundError, match="several runs"):
        Run.open(runs_dir, "20260930")
    with pytest.raises(RunNotFoundError, match="No run"):
        Run.open(runs_dir, "1999")


def test_open_without_runs_explains_how_to_start(tmp_path: Path) -> None:
    with pytest.raises(RunNotFoundError, match="speech2song ingest"):
        Run.open(tmp_path / "runs", None)


def test_atomic_write_keeps_old_content_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "data.json"
    atomic_write_text(target, "old")

    def failing_fsync(fd: int) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(manifest_mod.os, "fsync", failing_fsync)
    with pytest.raises(OSError, match="disk full"):
        atomic_write_text(target, "new")
    assert target.read_text() == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["data.json"]


def test_hash_memo_skips_rehash_until_file_changes(
    run: Run, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = run.path("data.bin")
    atomic_write_text(path, "first")
    calls: list[Path] = []
    real = manifest_mod.sha256_file

    def counting(p: Path) -> str:
        calls.append(p)
        return real(p)

    monkeypatch.setattr(manifest_mod, "sha256_file", counting)
    first = run.hash_file(path)
    assert run.hash_file(path) == first
    assert len(calls) == 1
    atomic_write_text(path, "other")  # same size, new inode
    assert run.hash_file(path) != first
    assert len(calls) == 2
    with path.open("w") as fh:  # in-place rewrite, different size
        fh.write("in place")
    assert run.hash_file(path)[0] == real(path)
    assert len(calls) == 3


def test_file_ref_paths(run: Run, tmp_path: Path) -> None:
    target = run.path("00_source.wav")
    target.write_bytes(b"audio")
    link = run.path("01_clean.wav")
    link.symlink_to("00_source.wav")
    ref = run.file_ref(link)
    assert ref.path == "01_clean.wav"
    assert ref.sha256 == run.file_ref(target).sha256
    outside = tmp_path / "outside.txt"
    outside.write_text("x")
    assert run.file_ref(outside).path == str(outside.resolve())


def test_save_prunes_memo_of_deleted_files(run: Run) -> None:
    path = run.path("temp.bin")
    path.write_bytes(b"x")
    run.hash_file(path)
    path.unlink()
    run.save()
    reopened = Run.open(run.root.parent, run.id)
    assert str(path.resolve()) not in reopened.manifest.hash_memo


def test_logging_to_file_is_scoped_to_the_block(run: Run) -> None:
    logger = logging.getLogger("speech2song.test")
    logging.getLogger("speech2song").setLevel(logging.DEBUG)
    with run.logging_to_file():
        logger.info("inside the block")
    logger.info("after the block")
    text = run.path(LOG_NAME).read_text()
    assert "inside the block" in text
    assert "after the block" not in text
