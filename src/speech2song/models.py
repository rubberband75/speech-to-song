"""Pydantic models for every JSON artifact a run writes."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel

# --- manifest.json -----------------------------------------------------------------------


class FileRef(BaseModel):
    """A file and its content hash. Run files use run-relative paths, others absolute."""

    path: str
    sha256: str
    size: int


class HashMemo(BaseModel):
    """Cached hash of a file, valid while its inode, size and mtime are unchanged.

    The inode catches same-size rewrites within one mtime tick: atomic writes
    (temp file + rename) always produce a new inode.
    """

    ino: int
    size: int
    mtime_ns: int
    sha256: str


StageStatus = Literal["running", "complete", "failed"]


class StageRecord(BaseModel):
    status: StageStatus
    version: int
    fingerprint: str
    inputs: dict[str, FileRef] = {}
    params: dict[str, Any] = {}
    outputs: dict[str, FileRef] = {}
    started_at: datetime
    finished_at: datetime | None = None
    elapsed_s: float | None = None
    summary: dict[str, Any] = {}
    error: str | None = None


class SourceProbe(BaseModel):
    """What ffprobe reported about the original input file."""

    format_name: str
    duration_s: float | None
    has_video: bool
    audio_stream: int  # index among audio streams, as in ffmpeg's `-map 0:a:N`
    codec: str | None
    sample_rate: int
    channels: int
    channel_layout: str | None = None
    bit_rate: int | None = None


class AudioInfo(BaseModel):
    """A WAV written by the pipeline, with its EBU R128 measurements."""

    path: str
    sample_rate: int
    channels: int
    frames: int
    duration_s: float
    subtype: str
    integrated_lufs: float | None = None
    loudness_range_lu: float | None = None
    sample_peak_dbfs: float | None = None
    true_peak_dbtp: float | None = None


class RunOptions(BaseModel):
    """Per-run choices. CLI flags update them, and they stick for later commands."""

    isolate_voice: bool = False
    clips: int | None = None
    music_backend: Literal["stub", "elevenlabs"] | None = None
    whisper_model: str | None = None
    language: str | None = None


class Manifest(BaseModel):
    schema_version: Literal[1] = 1
    run_id: str
    created_at: datetime
    tool_version: str
    input: FileRef
    transcript: FileRef | None = None
    preset: str
    options: RunOptions = RunOptions()
    settings: dict[str, Any] = {}
    source_probe: SourceProbe | None = None
    source: AudioInfo | None = None
    clean: AudioInfo | None = None
    stages: dict[str, StageRecord] = {}
    hash_memo: dict[str, HashMemo] = {}


# --- costs.json --------------------------------------------------------------------------


class CostEntry(BaseModel):
    ts: datetime
    run_id: str
    stage: str
    service: Literal["anthropic", "elevenlabs"]
    operation: str
    model: str
    units: dict[str, float]
    usd: float
    estimated: bool = True
    price_ref: str | None = None
    request_id: str | None = None
    note: str | None = None
