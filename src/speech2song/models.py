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
    key: str | None = None  # e.g. "D minor": overrides key detection
    refine_arc: bool = False  # ask Claude to refine the arrangement (a paid call)
    melody_layer: Literal["replay", "all", "off"] | None = None  # None: the preset's
    take: int | None = None  # which music take to mix (1-based); None: the first


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


# --- 04_melody.json ----------------------------------------------------------------------


class MelodyNote(BaseModel):
    start_beat: float  # from the start of the clip's phrase
    beats: float
    midi: int  # the note played
    pitch: float  # tuned, snapped and octave-placed pitch (fractional MIDI)
    speech_pitch: float  # as measured in the speech (fractional MIDI)
    start_s: float  # the syllable in the clip, before quantizing
    end_s: float
    velocity: int
    word: str | None = None


class BarChord(BaseModel):
    bar: int  # 0-based, within the phrase
    degree: int  # 1-7 in the key
    name: str  # e.g. "Am"
    pitch_classes: list[int]


class ClipMelody(BaseModel):
    clip_id: str
    file: str
    duration_s: float
    bars: int  # phrase length (whole bars, so loops line up)
    voiced_ratio: float  # share of frames with a pitch (a confidence hint)
    median_speech_pitch: float | None
    notes: list[MelodyNote]
    chords: list[BarChord]


class Melody(BaseModel):
    schema_version: Literal[1] = 1
    key: str  # e.g. "C minor"
    tonic: int
    mode: Literal["major", "minor"]
    key_source: Literal["detected", "preset", "override", "fallback"]
    key_confidence: float | None = None  # Krumhansl-Schmuckler correlation (detected keys)
    tuning_offset: float  # the speaker's offset from the semitone grid, removed first
    bpm: float
    tempo_cost: float  # grid misalignment at that tempo (0 = perfect)
    time_signature: str = "4/4"
    grid: str
    snap_strength: float
    octave_shift: int  # semitones added to every note to reach a melody range
    loop_phrase_count: int
    instrument: str
    main_clip: str  # the phrase used for the audio reference
    clips: list[ClipMelody]  # in playback order


# --- 05_arc.json (optional: Claude refines the arc) -------------------------------------


class ArcPart(BaseModel):
    """One step of the song: a non-speech section of `bars`, or a speech passage that
    plays `clips` back to back (each in its own speech_bed section)."""

    role: str
    bars: int = 0  # ignored for speech_bed parts (beds are fitted to their clips)
    clips: list[str] = []


class ArcPlan(BaseModel):
    """The JSON Claude returns when asked to refine the arc."""

    parts: list[ArcPart]
    notes: str = ""


class ArcAttempt(BaseModel):
    attempt: int
    model: str
    request_id: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float | None = None
    answer: ArcPlan | None = None
    problems: list[str] = []


class ArcResult(BaseModel):
    """Claude's raw answers. The free `arrange` stage checks the chosen one again and
    falls back to the preset's arc if it can't be used."""

    schema_version: Literal[1] = 1
    model: str
    attempts: list[ArcAttempt]
    chosen: int

    def answer(self) -> ArcPlan | None:
        return self.attempts[self.chosen].answer


# --- 05_arrangement.json -----------------------------------------------------------------


class Section(BaseModel):
    id: str
    role: str
    start_bar: int
    bars: int
    energy: float
    shape: Literal["flat", "rise", "fall"] = "flat"
    styles: list[str] = []
    chords: list[str] = []  # one chord name per bar
    clip_id: str | None = None  # the clip this speech_bed carries
    clip_offset_beats: float = 0.0  # where the clip starts, from the section start
    melody_phrase: str | None = None  # clip whose melody the melody layer replays here
    silent: bool = False  # the mix mutes the music here (a gap); nothing is generated for it
    start_s: float = 0.0  # informational: start_bar at the arrangement's tempo
    seconds: float = 0.0


class Arrangement(BaseModel):
    schema_version: Literal[1] = 1
    bpm: float
    key: str
    time_signature: str = "4/4"
    arc_source: Literal["preset", "claude"] = "preset"
    sections: list[Section]
    total_bars: int
    total_seconds: float
    notes: str | None = None
    warnings: list[str] = []


# --- 06_music ------------------------------------------------------------------------------


class TakeMeta(BaseModel):
    """take_NNN.meta.json: one generated backing track, kept exactly as the backend made it."""

    schema_version: Literal[1] = 1
    take: int  # 1-based
    backend: str
    model: str | None = None
    file: str  # run-relative
    sample_rate: int
    channels: int
    seconds: float
    seed: int | None = None
    usd: float = 0.0
    song_id: str | None = None  # stored song for inpainting, when the backend keeps one
    request_sha256: str
    # Chunk lengths (ms) the music was generated with. The take fits any arrangement with
    # the same grid, so it stays usable when only styles or the plan's wording change.
    grid_ms: list[int] | None = None
    params: dict[str, Any] = {}


class TakeAnalysis(BaseModel):
    """How a take matches its arrangement (06_music/analysis.json)."""

    take: int
    seconds: float
    tempo_bpm: float | None  # as estimated (the music may read as half or double time)
    tempo_ratio: float | None  # 0.5, 1 or 2: the reading closest to the target
    tempo_error: float | None  # relative error at that reading
    key: str | None
    key_relation: Literal["same", "relative", "other"] | None
    energy_correlation: float | None  # section levels vs. arrangement energies
    sections: list[str] = []  # the sections measured (silent ones are left out)
    section_levels_db: list[float] = []
    lufs: float | None = None
    flags: list[str] = []
    score: float  # higher is better; used to pick a take
    current: bool = True  # made for the current music request (else an older one that fits)


class TakeChoice(BaseModel):
    schema_version: Literal[1] = 1
    chosen: int
    reason: Literal["requested", "best score", "only take"]
    takes: list[TakeAnalysis]


# --- 07_mix/mix.json -------------------------------------------------------------------------


class PlacedClip(BaseModel):
    clip_id: str
    section_id: str
    file: str
    start_sample: int  # in the mix; the clip's samples are copied verbatim from here
    end_sample: int
    start_s: float


class SectionLevel(BaseModel):
    """Energy shaping of one section of the music (07_mix/mix.json)."""

    section_id: str
    role: str
    energy: float  # mean over the section (a build rises through it)
    lufs: float | None  # as generated; None when silent or too short to measure
    target_lufs: float | None
    gain_db: float  # applied in the mix


class MixReport(BaseModel):
    schema_version: Literal[1] = 1
    sample_rate: int
    frames: int
    seconds: float
    take: str  # the music file used
    clips: list[PlacedClip]
    melody_layer: Literal["replay", "all", "off"]
    music_lufs: float | None
    speech_lufs: float | None  # dry speech, before its gain
    speech_gain_db: float
    melody_gain_db: float | None
    duck_db: float
    target_lufs: float
    master_gain_db: float
    integrated_lufs: float
    true_peak_dbtp: float
    section_levels: list[SectionLevel] = []
    silenced: list[str] = []  # silent sections (gaps), muted in the mix
    warnings: list[str] = []
