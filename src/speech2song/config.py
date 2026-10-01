"""Settings (config.yaml), secrets (.env / environment) and the preset schema."""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, model_validator

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


class SectionRole(_Strict):
    energy: float = Field(ge=0, le=1)
    styles: list[str] = []


class SpeechInteraction(_Strict):
    music_recedes_during_clips: bool = True
    swell_after_clip_end: bool = True
    align_clip_starts_to: Literal["bar", "beat", "free"] = "bar"


class MixSpec(_Strict):
    sidechain_duck_db: float = Field(default=-9.0, le=0)
    duck_attack_ms: float = Field(default=30.0, gt=0)
    duck_release_ms: float = Field(default=400.0, gt=0)
    speech_highpass_hz: float = Field(default=90.0, ge=0)
    speech_reverb_send: float = Field(default=0.12, ge=0, le=1)
    speech_delay_send: float = Field(default=0.06, ge=0, le=1)
    target_lufs: float = Field(default=-14.0, lt=0)


class MelodySpec(_Strict):
    quantize_grid: str = "1/8"
    scale_snap_strength: float = Field(default=0.8, ge=0, le=1)
    loop_phrase_count: int = Field(default=3, ge=1)
    render_instrument: str = "soft_piano"

    @model_validator(mode="after")
    def _check_grid(self) -> "MelodySpec":
        if self.quantize_grid not in QUANTIZE_GRIDS:
            raise ValueError(f"melody.quantize_grid must be one of {QUANTIZE_GRIDS}")
        return self


class ClipsSpec(_Strict):
    """Clip-selection targets (an addition to the spec; see docs/DECISIONS.md)."""

    count: int = Field(default=5, ge=1)
    min_seconds: float = Field(default=3.0, gt=0)
    max_seconds: float = Field(default=15.0, gt=0)
    total_speech_seconds: float = Field(default=60.0, gt=0)
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
    mix: MixSpec = MixSpec()
    melody: MelodySpec = MelodySpec()
    clips: ClipsSpec = ClipsSpec()

    @model_validator(mode="after")
    def _check_roles(self) -> "Preset":
        if "speech_bed" not in self.section_roles:
            raise ValueError("section_roles must define 'speech_bed' (it plays under each clip)")
        unknown = sorted({role for role in self.arc if role not in self.section_roles})
        if unknown:
            raise ValueError(f"arc uses roles missing from section_roles: {unknown}")
        return self

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


class PricingConfig(_Strict):
    """Prices change, so config.yaml can override every entry (spec section 9).

    Claude list prices ship as defaults; ElevenLabs rates are added in M5 after the docs pass.
    """

    anthropic: dict[str, TokenPrice] = {}
    elevenlabs: dict[str, UnitPrice] = {}

    @model_validator(mode="after")
    def _merge_defaults(self) -> "PricingConfig":
        self.anthropic = {**DEFAULT_ANTHROPIC_PRICES, **self.anthropic}
        return self


class AppConfig(_Strict):
    runs_dir: Path = Path("runs")
    presets_dir: Path = Path("presets")
    default_preset: str = "cinematic_future_bass"
    claude_model: str = "claude-sonnet-5-5"  # spec section 8; claude-opus-5-5 for harder picks
    claude_effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"
    claude_max_tokens: int = Field(default=16000, ge=1024)
    claude_fallbacks: bool = True  # server-side retry on another model if Claude declines
    music_model: str = "music_v2_5"  # unverified until the M5 ElevenLabs docs pass
    music_backend: Literal["stub", "elevenlabs"] = "stub"
    transcribe_backend: Literal["whisper"] = "whisper"
    whisper_model: str = "large-v3-turbo"
    language: str | None = None  # None = auto-detect
    whisper: WhisperConfig = WhisperConfig()
    demucs: DemucsConfig = DemucsConfig()
    align: AlignConfig = AlignConfig()
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
