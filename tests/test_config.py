import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from speech2song.config import (
    AppConfig,
    ClipsSpec,
    list_presets,
    load_config,
    load_preset,
    load_secrets,
)
from speech2song.errors import ConfigError

from .conftest import PRESETS_DIR


def _preset_data() -> dict[str, Any]:
    return yaml.safe_load((PRESETS_DIR / "cinematic_future_bass.yaml").read_text())


def _write_preset(directory: Path, data: dict[str, Any], name: str | None = None) -> Path:
    path = directory / f"{name or data['name']}.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_real_preset_loads() -> None:
    preset = load_preset("cinematic_future_bass", PRESETS_DIR)
    assert preset.tempo.bpm == 114
    assert preset.tempo.feel == "half-time"
    assert preset.key.mode == "minor"
    assert preset.key.tonic == "auto"
    assert preset.section_roles["drop"].energy == 1.0
    assert preset.arc[0] == "intro"
    assert preset.melody.quantize_grid == "1/8"
    assert preset.clips == ClipsSpec()  # the optional block falls back to defaults
    assert preset.arc.count("speech_bed") == 3  # the closing line can end the song
    assert preset.role_bars("drop") == 16 and preset.role_bars("gap") == 1
    assert preset.section_roles["build"].shape == "rise"
    assert preset.melody.layer == "off"  # no MIDI in the song unless asked for
    assert preset.section_roles["gap"].silent and not preset.section_roles["build"].silent


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.update(tempoo=1), "tempoo"),
        (lambda d: d["arc"].append("chorus"), "chorus"),
        (lambda d: d["section_roles"]["drop"].update(energy=1.5), "energy"),
        (lambda d: d["key"].update(tonic="H"), "tonic"),
        (lambda d: d["section_roles"].pop("speech_bed"), "speech_bed"),
        (lambda d: d["melody"].update(quantize_grid="1/7"), "quantize_grid"),
        (lambda d: d.update(clips={"min_seconds": 9, "max_seconds": 4}), "max_seconds"),
        (lambda d: d["section_roles"]["speech_bed"].update(bars=4), "beds fit their clip"),
        (lambda d: d["section_roles"]["drop"].update(bars=0), "bars"),
        (lambda d: d["section_roles"]["drop"].update(shape="wobble"), "shape"),
        (lambda d: d.update(arc=["intro", "drop"]), "at least one speech_bed"),
        (lambda d: d["melody"].update(layer_roles=["chorus"]), "layer_roles"),
        (lambda d: d["melody"].update(layer="loud"), "layer"),
        (lambda d: d["section_roles"]["speech_bed"].update(silent=True), "beds carry music"),
        (lambda d: d["mix"].update(energy_max_db=-1), "energy_max_db"),
    ],
)
def test_invalid_presets_are_rejected(tmp_path: Path, mutate: Any, message: str) -> None:
    data = _preset_data()
    mutate(data)
    _write_preset(tmp_path, data)
    with pytest.raises(ConfigError, match=message):
        load_preset(data["name"], tmp_path)


def test_a_bare_yaml_off_turns_the_melody_layer_off(tmp_path: Path) -> None:
    text = yaml.safe_dump(_preset_data()).replace("layer: 'off'", "layer: off")
    assert "layer: off" in text
    (tmp_path / "cinematic_future_bass.yaml").write_text(text)
    assert load_preset("cinematic_future_bass", tmp_path).melody.layer == "off"


def test_sharp_tonic_is_accepted(tmp_path: Path) -> None:
    data = _preset_data()
    data["key"]["tonic"] = "F#"
    _write_preset(tmp_path, data)
    assert load_preset(data["name"], tmp_path).key.tonic == "F#"


def test_preset_name_must_match_file_name(tmp_path: Path) -> None:
    _write_preset(tmp_path, _preset_data(), name="other_name")
    with pytest.raises(ConfigError, match="does not match"):
        load_preset("other_name", tmp_path)


def test_missing_preset_lists_available() -> None:
    with pytest.raises(ConfigError, match="cinematic_future_bass"):
        load_preset("nope", PRESETS_DIR)


def test_preset_digest_tracks_content(tmp_path: Path) -> None:
    data = _preset_data()
    _write_preset(tmp_path, data)
    before = load_preset(data["name"], tmp_path).digest()
    data["mix"]["target_lufs"] = -16
    _write_preset(tmp_path, data)
    assert load_preset(data["name"], tmp_path).digest() != before


def test_list_presets_reports_invalid_files(tmp_path: Path) -> None:
    good = _preset_data()
    _write_preset(tmp_path, good)
    bad = _preset_data()
    bad["name"] = "broken"
    bad["arc"] = ["missing_role"]
    _write_preset(tmp_path, bad)
    listings = {item.name: item for item in list_presets(tmp_path)}
    assert listings["cinematic_future_bass"].preset is not None
    assert listings["broken"].preset is None
    assert "missing_role" in (listings["broken"].error or "")


def test_config_defaults_without_file() -> None:
    config = load_config()
    assert config == AppConfig()
    assert config.whisper_model == "large-v3-turbo"
    assert config.music_backend == "stub"


def test_config_file_resolves_relative_paths(tmp_path: Path) -> None:
    conf_dir = tmp_path / "conf"
    conf_dir.mkdir()
    path = conf_dir / "config.yaml"
    path.write_text("runs_dir: my_runs\nwhisper_model: small\nwhisper:\n  beam_size: 2\n")
    config = load_config(path)
    assert config.runs_dir == conf_dir / "my_runs"
    assert config.whisper_model == "small"
    assert config.whisper.beam_size == 2


def test_config_picks_up_cwd_config_yaml(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text("default_preset: lofi\n")
    assert load_config().default_preset == "lofi"


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("whisper_modle: small\n")
    with pytest.raises(ConfigError, match="whisper_modle"):
        load_config(path)


def test_secrets_load_from_dotenv_without_overriding_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("ANTHROPIC_API_KEY=sk-from-dotenv\nELEVENLABS_API_KEY=el-from-dotenv\n")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "el-from-env")
    secrets = load_secrets(env_file)
    assert secrets.anthropic_api_key is not None
    assert secrets.anthropic_api_key.get_secret_value() == "sk-from-dotenv"
    assert secrets.elevenlabs_api_key is not None
    assert secrets.elevenlabs_api_key.get_secret_value() == "el-from-env"
    assert "sk-from-dotenv" not in repr(secrets)
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-from-dotenv"


def test_config_snapshot_contains_no_secret_fields() -> None:
    snapshot = AppConfig().snapshot()
    assert not [key for key in snapshot if "key" in key.lower() or "secret" in key.lower()]


def test_example_config_documents_the_defaults() -> None:
    from .conftest import REPO_ROOT

    example = load_config(REPO_ROOT / "config.example.yaml")
    paths = {"runs_dir", "presets_dir"}
    assert example.model_dump(exclude=paths) == AppConfig().model_dump(exclude=paths)
