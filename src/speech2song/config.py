"""Settings (config.yaml), secrets (.env / environment) and the preset schema."""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import find_dotenv, load_dotenv
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)

from speech2song.errors import ConfigError

# Fixed by the spec (section 5) and required by the demucs models.
SAMPLE_RATE = 44100

PITCH_CLASSES: dict[str, int] = {
    "C": 0, "B#": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "Fb": 4,
    "F": 5, "E#": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10,
    "Bb": 10, "B": 11, "Cb": 11,
}  # fmt: skip

QUANTIZE_GRIDS = ("1/1", "1/2", "1/4", "1/8", "1/16", "1/32")


class _Strict(BaseModel):
    """Rejects unknown keys so typos in YAML fail loudly."""

    model_config = ConfigDict(extra="forbid")


# --- Preset schema -----------------------------------------------------------------------


class TempoSpec(_Strict):
    bpm: float = Field(gt=30, lt=300)
    feel: Literal["normal", "half-time", "double-time"] = "normal"
    tolerance_bpm: float = Field(default=6.0, ge=0)


class KeySpec(_Strict):
    mode: Literal["major", "minor"]
    tonic: str = "auto"
    fallback: str = "C"

    @model_validator(mode="after")
    def _check_pitch_classes(self) -> "KeySpec":
        if self.tonic != "auto" and self.tonic not in PITCH_CLASSES:
            raise ValueError(f"key.tonic must be 'auto' or a pitch class, got {self.tonic!r}")
        if self.fallback not in PITCH_CLASSES:
            raise ValueError(f"key.fallback must be a pitch class, got {self.fallback!r}")
        return self


SPEECH_ROLE = "speech_bed"  # the role a clip plays under
DEFAULT_SECTION_BARS = 8


class SectionRole(_Strict):
    energy: float = Field(ge=0, le=1)
    styles: list[str] = []
    # Styles this section must avoid (on top of the preset's negative_styles). The model
    # fills silence unless it is told what to leave out.
    negative_styles: list[str] = []
    # When a role appears more than once, its first and last occurrence get these too,
    # so the song develops (a restrained first drop, a climactic last one).
    first_styles: list[str] = []
    last_styles: list[str] = []
    # Length in bars (default 8). A speech_bed is fitted to its clip instead.
    bars: int | None = Field(default=None, ge=1, le=64)
    # Energy over the section: steady, rising into the next section, or fading out.
    shape: Literal["flat", "rise", "fall"] = "flat"
    # A short lift before a drop (the gap). Nothing is generated for it: the music of the
    # section before runs on, and the mix adds a swell of that music's tail on top.
    silent: bool = False


class SpeechInteraction(_Strict):
    music_recedes_during_clips: bool = True
    swell_after_clip_end: bool = True
    align_clip_starts_to: Literal["bar", "beat", "free"] = "bar"
    # Beats after a clip's last word before the next section may start.
    tail_beats: float = Field(default=2.0, ge=0)
    # Music between the parts of a long quote, after a part's last word.
    part_pause_beats: float = Field(default=4.0, ge=0)
    # A passage's music starts this long before its first quote, so the music has settled
    # into the quiet bed when the voice comes in (a bar = 4 beats; 0 = none).
    lead_in_beats: float = Field(default=4.0, ge=0)


class MixSpec(_Strict):
    # The music under a speech passage is set once for the whole passage, at its bar
    # lines (never following the voice): only as far down as it takes to sit
    # `bed_margin_db` under the speech (speech band), and at most `sidechain_duck_db`.
    # A bed the music model already made quiet is left alone.
    sidechain_duck_db: float = Field(default=-9.0, le=0)
    bed_margin_db: float = Field(default=14.0, ge=0, le=40)
    speech_highpass_hz: float = Field(default=90.0, ge=0)
    speech_reverb_send: float = Field(default=0.12, ge=0, le=1)
    speech_delay_send: float = Field(default=0.06, ge=0, le=1)
    target_lufs: float = Field(default=-14.0, lt=0)
    speech_level_lu: float = 2.0  # speech loudness relative to the music's
    melody_layer_lu: float = -10.0  # melody layer loudness relative to the music's
    melody_under_speech_db: float = Field(default=-6.0, le=0)  # extra cut under speech
    # Energy shaping: each section's level is pulled toward a line through the sections'
    # median, `energy_range_db` louder at energy 1 than at energy 0. Only the part of a
    # deviation beyond `energy_tolerance_db` is corrected, by at most `energy_max_db`
    # (0 turns shaping off), or `energy_max_boost_db` when a section is too quiet (None:
    # the same; the music model sometimes makes a quiet bed near-silent).
    energy_range_db: float = Field(default=10.0, ge=0)
    energy_tolerance_db: float = Field(default=3.0, ge=0)
    energy_max_db: float = Field(default=6.0, ge=0)
    energy_max_boost_db: float | None = Field(default=None, ge=0)
    # A gap (silent section) is a lift into the next one: its music rises by `gap_lift_db`
    # and a reverse swell of the build's tail (at `gap_swell` of that music's level) peaks
    # on the downbeat. `gap_swell` 0 and `gap_lift_db` 0 leave the gap as generated.
    gap_swell: float = Field(default=0.8, ge=0, le=1)
    gap_lift_db: float = Field(default=3.0, ge=0)
    # The last chord rings out: the final audible music goes through a reverb that dies
    # away over this long (0 = the music ends as generated). Arrangements with an ending
    # of their own ring out only where the music stops abruptly (or ends on a final hit).
    ring_out_s: float = Field(default=6.0, ge=0)
    # A passage played alone: the music stops just before its last phrase and its reverb
    # dies away over this long, and it comes back a beat or two after the last word
    # through a reverse swell (at `gap_swell` of that music's level).
    alone_ring_s: float = Field(default=1.5, ge=0)
    # A section after a gap whose first 1-N bars are near-silent (a generated drop that
    # opens with a silent bar and a riser) is pulled onto its downbeat (0 turns this off).
    late_entry_max_bars: int = Field(default=4, ge=0, le=8)
    # Wherever a word is spoken, the music in the speech band (200 Hz-5 kHz) stays at
    # least this far under it: on top of the fixed duck, the music dips further under
    # quiet words.
    speech_margin_db: float = Field(default=10.0, ge=0, le=30)


class MelodySpec(_Strict):
    quantize_grid: str = "1/8"
    scale_snap_strength: float = Field(default=0.8, ge=0, le=1)
    loop_phrase_count: int = Field(default=3, ge=1)
    render_instrument: str = "soft_piano"
    # The melody layer in the mix: "replay" plays the line just heard in the sections of
    # `layer_roles`; "all" also plays each clip's melody quietly under the speech.
    layer: Literal["replay", "all", "off"] = "off"
    layer_roles: list[str] = ["breakdown", "drop"]

    @field_validator("layer", mode="before")
    @classmethod
    def _yaml_off(cls, value: object) -> object:
        return "off" if value is False else value  # YAML reads a bare `off` as false

    @model_validator(mode="after")
    def _check_grid(self) -> "MelodySpec":
        if self.quantize_grid not in QUANTIZE_GRIDS:
            raise ValueError(f"melody.quantize_grid must be one of {QUANTIZE_GRIDS}")
        return self


class ClipsSpec(_Strict):
    """Clip-selection targets (an addition to the spec; see docs/DECISIONS.md)."""

    count: int = Field(default=8, ge=1)  # about this many quotes...
    count_tolerance: int = Field(default=2, ge=0)  # ...give or take this, to cover the talk
    min_seconds: float = Field(default=3.0, gt=0)
    max_seconds: float = Field(default=15.0, gt=0)  # the usual longest quote
    long_max_seconds: float = Field(default=40.0, gt=0)  # for quotes essential to the talk
    total_speech_seconds: float = Field(default=150.0, gt=0)
    part_seconds: float = Field(default=15.0, gt=0)  # longer quotes play in parts
    fade_ms: float = Field(default=15.0, ge=0)
    # How far cut points may move from the word timestamps: before a clip's first word,
    # after its last word (speech and room reverb decay well past ASR end times), and
    # into the clip. Never past the middle of a neighbouring word.
    lead_search_ms: float = Field(default=300.0, ge=0)
    tail_search_ms: float = Field(default=400.0, ge=0)
    inner_search_ms: float = Field(default=150.0, ge=0)

    @model_validator(mode="after")
    def _check_bounds(self) -> "ClipsSpec":
        if self.max_seconds < self.min_seconds:
            raise ValueError("clips.max_seconds must be >= clips.min_seconds")
        if self.long_max_seconds < self.max_seconds:
            raise ValueError("clips.long_max_seconds must be >= clips.max_seconds")
        return self


class Preset(_Strict):
    name: str
    description: str
    tempo: TempoSpec
    key: KeySpec
    positive_styles: list[str]
    negative_styles: list[str] = []
    section_roles: dict[str, SectionRole]
    arc: list[str] = Field(min_length=1)
    speech_interaction: SpeechInteraction = SpeechInteraction()
    # How the song's music ends (Claude's arc may choose another): "held_chord" comes home
    # to the tonic and that chord is held and rings away, "stop" ends crisply on the
    # tonic, "fade" lets the music fade out by itself.
    ending: Literal["held_chord", "fade", "stop"] = "held_chord"
    mix: MixSpec = MixSpec()
    melody: MelodySpec = MelodySpec()
    clips: ClipsSpec = ClipsSpec()

    @model_validator(mode="after")
    def _check_roles(self) -> "Preset":
        if SPEECH_ROLE not in self.section_roles:
            raise ValueError("section_roles must define 'speech_bed' (it plays under each clip)")
        if self.section_roles[SPEECH_ROLE].bars is not None:
            raise ValueError("section_roles.speech_bed.bars is not allowed: beds fit their clip")
        if self.section_roles[SPEECH_ROLE].silent:
            raise ValueError("section_roles.speech_bed.silent is not allowed: beds carry music")
        if SPEECH_ROLE not in self.arc:
            raise ValueError("arc must contain at least one speech_bed")
        unknown = sorted({role for role in self.arc if role not in self.section_roles})
        if unknown:
            raise ValueError(f"arc uses roles missing from section_roles: {unknown}")
        unknown = sorted(set(self.melody.layer_roles) - set(self.section_roles))
        if unknown:
            raise ValueError(f"melody.layer_roles uses roles missing from section_roles: {unknown}")
        return self

    def role_bars(self, role: str) -> int:
        bars = self.section_roles[role].bars
        return DEFAULT_SECTION_BARS if bars is None else bars

    def digest(self) -> str:
        """Content hash, used in stage params so preset edits invalidate dependent stages."""
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass
class PresetListing:
    name: str
    path: Path
    preset: Preset | None
    error: str | None


def preset_path(name_or_path: str, presets_dir: Path) -> Path:
    candidate = Path(name_or_path)
    if candidate.suffix in (".yaml", ".yml"):
        return candidate
    return presets_dir / f"{name_or_path}.yaml"


def load_preset(name_or_path: str, presets_dir: Path) -> Preset:
    path = preset_path(name_or_path, presets_dir)
    if not path.is_file():
        available = ", ".join(p.stem for p in sorted(presets_dir.glob("*.yaml"))) or "none"
        raise ConfigError(f"Preset not found: {path} (available in {presets_dir}: {available})")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        preset = Preset.model_validate(data)
    except (yaml.YAMLError, ValidationError) as exc:
        raise ConfigError(f"Invalid preset {path}:\n{exc}") from exc
    if preset.name != path.stem:
        raise ConfigError(f"Preset name {preset.name!r} does not match its file name {path.name}")
    return preset


def list_presets(presets_dir: Path) -> list[PresetListing]:
    listings = []
    for path in sorted(presets_dir.glob("*.yaml")):
        try:
            preset = load_preset(str(path), presets_dir)
            listings.append(PresetListing(path.stem, path, preset, None))
        except ConfigError as exc:
            listings.append(PresetListing(path.stem, path, None, str(exc)))
    return listings


# --- App config (config.yaml) ----------------------------------------------------------


class WhisperConfig(_Strict):
    device: Literal["auto", "cpu", "cuda"] = "auto"
    compute_type: str = "auto"  # auto: int8 on CPU, float16 on CUDA
    beam_size: int = Field(default=5, ge=1)
    vad_filter: bool = True
    batch_size: int = Field(default=8, ge=0)  # 0 = sequential decoding (slower here)
    cpu_threads: int = Field(default=0, ge=0)  # 0 = number of physical cores (estimated)
    condition_on_previous_text: bool = True
    # Re-transcribe gaps between words that contain speech (decoders can drop chunks).
    repair_gaps: bool = True
    repair_min_gap_s: float = Field(default=3.0, gt=0)
    repair_min_speech_s: float = Field(default=1.5, gt=0)


class DemucsConfig(_Strict):
    model: str = "htdemucs"
    device: str = "cpu"
    shifts: int = Field(default=0, ge=0)
    overlap: float = Field(default=0.25, ge=0, lt=1)
    window_s: float = Field(default=120.0, gt=0)
    context_s: float = Field(default=5.0, ge=0)
    crossfade_s: float = Field(default=0.5, ge=0)
    jobs: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _check_crossfade(self) -> "DemucsConfig":
        if self.crossfade_s > self.window_s:
            raise ValueError("demucs.crossfade_s must not exceed demucs.window_s")
        if self.crossfade_s / 2 > self.context_s:
            raise ValueError("demucs.context_s must be at least half of demucs.crossfade_s")
        return self


class AlignConfig(_Strict):
    fuzzy_threshold: float = Field(default=0.6, ge=0, le=1)
    unspoken_min_tokens: int = Field(default=3, ge=1)
    unspoken_line_ratio: float = Field(default=0.25, ge=0, le=1)
    asr_sentence_min_words: int = Field(default=4, ge=1)
    pause_split_s: float = Field(default=1.5, gt=0)
    max_sentence_s: float = Field(default=30.0, gt=0)
    min_coverage: float = Field(default=0.2, ge=0, le=1)


class TokenPrice(_Strict):
    input_usd_per_mtok: float = Field(ge=0)
    output_usd_per_mtok: float = Field(ge=0)
    cache_read_usd_per_mtok: float | None = Field(default=None, ge=0)  # None: 0.1x input
    cache_write_usd_per_mtok: float | None = Field(default=None, ge=0)  # None: 1.25x input
    as_of: str | None = None


class UnitPrice(_Strict):
    unit: str
    usd_per_unit: float = Field(ge=0)
    as_of: str | None = None


# Anthropic first-party list prices from the Claude API reference, as of 2026-09-25.
# Entries under `pricing.anthropic` in config.yaml override these per model. The Opus
# entries price server-side refusal fallbacks, which bill at the serving model's rates.
_PRICES_AS_OF = "2026-09-25"
DEFAULT_ANTHROPIC_PRICES = {
    "claude-sonnet-5-5": TokenPrice(input_usd_per_mtok=2.0, output_usd_per_mtok=10.0,
                                    cache_read_usd_per_mtok=0.2, as_of=_PRICES_AS_OF),
    "claude-opus-5-5": TokenPrice(input_usd_per_mtok=4.0, output_usd_per_mtok=20.0,
                                  cache_read_usd_per_mtok=0.2, as_of=_PRICES_AS_OF),
    "claude-opus-5": TokenPrice(input_usd_per_mtok=5.0, output_usd_per_mtok=25.0,
                                as_of=_PRICES_AS_OF),
    "claude-opus-4-8": TokenPrice(input_usd_per_mtok=5.0, output_usd_per_mtok=25.0,
                                  as_of=_PRICES_AS_OF),
}  # fmt: skip


# ElevenLabs API rate for Eleven Music (elevenlabs.io/pricing/api, 2026-10-01): $0.15 per
# generated minute on every tier. Uploads for inpainting cost the same as generation.
# Subscriptions spend included minutes first; override for your plan in config.yaml.
DEFAULT_ELEVENLABS_PRICES = {
    "music": UnitPrice(unit="minute", usd_per_unit=0.15, as_of="2026-10-01"),
}


class PricingConfig(_Strict):
    """Prices change, so config.yaml can override every entry (spec section 9)."""

    anthropic: dict[str, TokenPrice] = {}
    elevenlabs: dict[str, UnitPrice] = {}

    @model_validator(mode="after")
    def _merge_defaults(self) -> "PricingConfig":
        self.anthropic = {**DEFAULT_ANTHROPIC_PRICES, **self.anthropic}
        self.elevenlabs = {**DEFAULT_ELEVENLABS_PRICES, **self.elevenlabs}
        return self


class ElevenLabsConfig(_Strict):
    """Eleven Music API settings (docs read 2026-10-01; see docs/DECISIONS.md)."""

    output_format: str = "auto"  # the API picks mp3_48000_192 for music_v2 models
    timeout_s: int = Field(default=900, ge=30)  # long songs take minutes; never retried
    context_adherence: Literal["low", "medium", "high"] = "high"
    # M5 plans: condition the first chunk on 04_melody_reference.wav (uploaded, billed
    # like a generation). Off: the docs say references carry feel and palette, not notes.
    melody_reference: bool = False
    # M7 plans: the music around each quote (the song's first chunk and the music right
    # after a speech passage) is conditioned on that quote's rendered melody. The
    # melodies used are uploaded once, as one file (billed like a generation). Off: on
    # the first real run the conditioned sections did not follow the melodies (chroma
    # correlation at chance), but the rendered piano's sound came through.
    melody_conditioning: bool = False
    condition_strength: Literal["low", "medium", "high", "xhigh"] = "low"

    @model_validator(mode="after")
    def _check_format(self) -> "ElevenLabsConfig":
        if not (self.output_format == "auto" or self.output_format.startswith(("mp3_", "opus_"))):
            raise ValueError("elevenlabs.output_format must be auto, mp3_* or opus_* "
                             "(raw PCM is not supported yet)")  # fmt: skip
        return self


class AppConfig(_Strict):
    runs_dir: Path = Path("runs")
    presets_dir: Path = Path("presets")
    downloads_dir: Path = Path("inputs")  # where talks given as a URL are saved
    default_preset: str = "cinematic_future_bass"
    claude_model: str = "claude-sonnet-5-5"  # spec section 8; claude-opus-5-5 for harder picks
    claude_effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"
    claude_max_tokens: int = Field(default=16000, ge=1024)
    claude_fallbacks: bool = True  # server-side retry on another model if Claude declines
    music_model: Literal["music_v2", "music_v2_5"] = "music_v2_5"  # chunked plans need v2+
    music_backend: Literal["stub", "elevenlabs"] = "stub"
    music_takes: int = Field(default=2, ge=1, le=8)  # takes per generation
    transcribe_backend: Literal["whisper"] = "whisper"
    whisper_model: str = "large-v3-turbo"
    language: str | None = None  # None = auto-detect
    soundfont: Path | None = None  # General MIDI .sf2 for rendering; None = find a system one
    whisper: WhisperConfig = WhisperConfig()
    demucs: DemucsConfig = DemucsConfig()
    align: AlignConfig = AlignConfig()
    elevenlabs: ElevenLabsConfig = ElevenLabsConfig()
    pricing: PricingConfig = PricingConfig()

    def snapshot(self) -> dict[str, Any]:
        """Settings recorded in the manifest. Secrets are not part of AppConfig."""
        return self.model_dump(mode="json")


def load_config(path: Path | None = None) -> AppConfig:
    """Load config.yaml (explicit path, else ./config.yaml if present, else defaults).

    Relative paths inside a config file resolve against that file's directory.
    """
    if path is None:
        default = Path("config.yaml")
        if not default.is_file():
            return AppConfig()
        path = default
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        config = AppConfig.model_validate(data)
    except (yaml.YAMLError, ValidationError) as exc:
        raise ConfigError(f"Invalid config {path}:\n{exc}") from exc
    base = path.parent
    return config.model_copy(
        update={
            "runs_dir": base / config.runs_dir,
            "presets_dir": base / config.presets_dir,
            "downloads_dir": base / config.downloads_dir,
        }
    )


# --- Secrets ---------------------------------------------------------------------------


class Secrets(BaseModel):
    """API keys. Kept out of AppConfig so they can never reach the manifest or logs."""

    anthropic_api_key: SecretStr | None = None
    elevenlabs_api_key: SecretStr | None = None


def load_secrets(dotenv_path: Path | None = None) -> Secrets:
    """Read keys from the environment, after loading .env (existing env vars win)."""
    env_file = dotenv_path or find_dotenv(usecwd=True)
    if env_file:
        load_dotenv(env_file, override=False)
    return Secrets(
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY") or None,
        elevenlabs_api_key=os.environ.get("ELEVENLABS_API_KEY") or None,
    )
