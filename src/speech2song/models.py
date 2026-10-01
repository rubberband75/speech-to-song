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
    claude_model: str | None = None


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


# --- 02_asr.json -------------------------------------------------------------------------


class AsrWord(BaseModel):
    w: str
    start: float
    end: float
    conf: float


class AsrSegment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None
    words: list[AsrWord] = []
    repaired: bool = False  # produced by the gap-repair pass


class AsrRepair(BaseModel):
    """A gap with speech but no words. The window around it (the gap plus the
    segments touching it) was re-transcribed and its words replaced."""

    gap_start: float
    gap_end: float
    speech_s: float  # seconds of detected speech inside the gap
    start: float  # re-transcribed window
    end: float
    words_before: int = 0  # words the window held before the repair
    words_after: int = 0


class AsrResult(BaseModel):
    """Raw speech recognition output, cached so re-aligning never re-runs ASR."""

    schema_version: Literal[1] = 1
    backend: str
    model: str
    params: dict[str, Any] = {}
    audio: str
    language: str
    language_probability: float | None = None
    duration_s: float
    segments: list[AsrSegment] = []
    repairs: list[AsrRepair] = []

    def words(self) -> list[AsrWord]:
        return [word for segment in self.segments for word in segment.words]


# --- 02_transcript.json ------------------------------------------------------------------

# asr: no official transcript. matched/fuzzy: official word timed by ASR. interpolated:
# official word ASR missed (time estimated). asr_only: spoken but not in the official text.
WordFlag = Literal["asr", "matched", "fuzzy", "interpolated", "asr_only"]


class Word(BaseModel):
    w: str
    start: float
    end: float
    conf: float | None = None
    flag: WordFlag = "asr"
    asr: str | None = None  # what ASR heard, when it differs from `w`


class Sentence(BaseModel):
    id: int
    text: str
    start: float
    end: float
    word_start: int  # index into Transcript.words (inclusive)
    word_end: int  # exclusive
    source: Literal["official", "asr"]
    avg_conf: float | None = None


class Span(BaseModel):
    text: str
    start: float | None = None
    end: float | None = None
    line: int | None = None  # 1-based line in the official transcript


class AlignmentReport(BaseModel):
    official_words: int
    matched: int
    fuzzy: int
    interpolated: int
    unspoken: int
    asr_words: int
    asr_only: int
    quality: float  # (matched + fuzzy) / official words, as in the spec
    quality_spoken: float  # same, ignoring official words judged unspoken
    coverage: float  # fraction of ASR words tied to official text
    anchor_coverage: float  # fraction of ASR tokens in runs of 3+ consecutive pairs
    unspoken_spans: list[Span] = []
    asr_only_spans: list[Span] = []
    fallback: bool = False  # official text ignored because it barely matched


class AsrRef(BaseModel):
    backend: str
    model: str


class Transcript(BaseModel):
    schema_version: Literal[1] = 1
    source: str
    duration_s: float
    language: str
    words: list[Word]
    sentences: list[Sentence]
    official_transcript_used: bool
    alignment: AlignmentReport | None = None
    asr: AsrRef


# --- 03_selection.json, 03_clips.json, 03_review.json ---------------------------------------

ClipRole = Literal["hook", "build", "payoff", "breakdown", "outro"]


class SelectedClip(BaseModel):
    """One clip as Claude proposes it: a range of transcript sentence IDs."""

    id: str
    start_sentence: int
    end_sentence: int
    text: str
    score: float
    role: ClipRole
    reason: str


class ClipSelection(BaseModel):
    """The JSON Claude must return (spec section 6, stage 3)."""

    clips: list[SelectedClip]
    suggested_order: list[str]
    notes: str | None = None


class ClipTargets(BaseModel):
    count: int
    min_seconds: float
    max_seconds: float
    total_speech_seconds: float


class SelectionAttempt(BaseModel):
    attempt: int
    model: str  # the model that produced the answer (differs on a refusal fallback)
    request_id: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float | None = None
    answer: ClipSelection | None = None
    problems: list[str] = []


class SelectionResult(BaseModel):
    """Claude's raw answers. Clean-up happens in the free `clips` stage, so changing
    those rules never repeats a paid call."""

    schema_version: Literal[2] = 2
    model: str  # requested model
    targets: ClipTargets
    attempts: list[SelectionAttempt]
    chosen: int  # index of the attempt whose answer is used

    def answer(self) -> ClipSelection:
        answer = self.attempts[self.chosen].answer
        assert answer is not None
        return answer


class Clip(BaseModel):
    id: str
    file: str  # run-relative WAV path
    start_sentence: int
    end_sentence: int
    text: str  # transcript text of the sentences
    score: float
    role: ClipRole
    reason: str
    nominal_start_s: float  # word timestamps
    nominal_end_s: float
    start_s: float  # refined cut points
    end_s: float
    start_sample: int  # in the source's sample rate; end is exclusive
    end_sample: int
    duration_s: float
    cut_level_db: tuple[float, float]  # level around the start and end cuts


class ClipSet(BaseModel):
    schema_version: Literal[1] = 1
    source: str  # the file clips were cut from
    isolated: bool  # True when that file is demucs output rather than the original
    sample_rate: int
    channels: int
    fade_ms: float
    clips: list[Clip]  # every validated clip, in time order
    order: list[str]  # playback order of the kept clips
    dropped: list[str] = []  # removed during review
    notes: str | None = None
    warnings: list[str] = []  # clips or limits the clean-up had to fix


class ClipReview(BaseModel):
    """User edits from --interactive-review, tied to the selection they were made on."""

    selection_sha256: str
    order: list[str]
    dropped: list[str] = []
