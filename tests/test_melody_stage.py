"""The melody step: the pure melody builder, then select -> melody end to end."""

from collections.abc import Callable
from pathlib import Path

import pretty_midi
import pytest
import soundfile as sf
import yaml
from typer.testing import Result

from speech2song.audio.pitch import Segment
from speech2song.config import load_preset
from speech2song.listen import write_listening_set
from speech2song.models import Clip, Melody
from speech2song.stages.melody import build_melody

from .conftest import PRESETS_DIR, Talk
from .fixtures.fake_claude import clip_answer, response

Cli = Callable[..., Result]
PRESET = load_preset("cinematic_future_bass", PRESETS_DIR)


def _clip(cid: str, duration: float, role: str = "build") -> Clip:
    return Clip(id=cid, file=f"clips/{cid}.wav", start_sentence=1, end_sentence=1, text="t",
                score=0.5, role=role, reason="r", nominal_start_s=0, nominal_end_s=duration,
                start_s=0, end_s=duration, start_sample=0, end_sample=1, duration_s=duration,
                cut_level_db=(-80.0, -80.0))  # fmt: skip


def _segments(pitches: list[float], step: float = 0.3) -> list[Segment]:
    return [Segment(i * step, i * step + 0.25, p, -20.0, f"w{i}") for i, p in enumerate(pitches)]


# D minor arpeggio-ish speech, sung 0.2 semitones sharp, an octave and a half low.
D_MINOR = [38.2, 41.2, 45.2, 38.2, 43.2, 41.2, 38.2, 45.2, 40.2, 38.2]


def _melody(**overrides: object) -> Melody:
    clips = [_clip("a", 3.6), _clip("b", 3.1, role="hook")]
    segments = {"a": _segments(D_MINOR), "b": _segments(D_MINOR[::-1])}
    stats = {"a": (0.6, 41.2), "b": (0.5, 41.2)}
    preset = PRESET.model_copy(update=overrides) if overrides else PRESET
    return build_melody(clips, segments, stats, preset, None)


def test_melody_finds_the_key_and_cleans_up_the_pitch() -> None:
    melody = _melody()
    assert (melody.key, melody.key_source) == ("D minor", "detected")
    assert melody.tuning_offset == pytest.approx(0.2, abs=0.01)
    assert melody.octave_shift == 24
    assert melody.main_clip == "b"  # the hook
    scale = {2, 4, 5, 7, 9, 10, 0}
    for clip in melody.clips:
        assert all(n.midi % 12 in scale for n in clip.notes)  # strength 0.8 lands in key
        assert all(55 <= n.midi <= 84 for n in clip.notes)
        assert all(float(n.start_beat * 2).is_integer() for n in clip.notes)  # 1/8 grid
        assert len(clip.chords) == clip.bars
    assert 108 <= melody.bpm <= 120


def test_key_override_and_fixed_preset_tonic() -> None:
    clips = [_clip("a", 3.6)]
    segments, stats = {"a": _segments(D_MINOR)}, {"a": (0.6, 41.2)}
    melody = build_melody(clips, segments, stats, PRESET, "F major")
    assert (melody.key, melody.key_source) == ("F major", "override")
    fixed = PRESET.model_copy(
        update={"key": PRESET.key.model_copy(update={"tonic": "G"})}
    )  # preset says G (minor)
    melody = build_melody(clips, segments, stats, fixed, None)
    assert (melody.key, melody.key_source) == ("G minor", "preset")


def _select(cli: Cli, talk: Talk, install: Callable) -> None:
    install(response(clip_answer(talk[1])))
    assert cli("select", "--yes").exit_code == 0


def test_melody_step_end_to_end(cli: Cli, talk: Talk, install: Callable) -> None:
    run = talk[0]
    _select(cli, talk, install)
    result = cli("melody")
    assert result.exit_code == 0, result.output
    melody = Melody.model_validate_json(run.path("04_melody.json").read_text())
    assert [c.clip_id for c in melody.clips] == ["c5", "c4", "c3", "c2", "c1"]  # play order
    assert all(c.notes for c in melody.clips)
    assert melody.loop_phrase_count == 3

    midi = pretty_midi.PrettyMIDI(str(run.path("04_melody.mid")))
    assert midi.get_tempo_changes()[1][0] == pytest.approx(melody.bpm, abs=0.01)  # MIDI: whole µs
    melody_track = next(i for i in midi.instruments if i.name == "melody")
    assert len(melody_track.notes) == 3 * sum(len(c.notes) for c in melody.clips)
    assert melody_track.program == 0  # soft piano

    main = next(c for c in melody.clips if c.clip_id == melody.main_clip)
    info = sf.info(str(run.path("04_melody_reference.wav")))
    expected = 2 * main.bars * 4 * 60 / melody.bpm + 1.5
    assert info.duration == pytest.approx(expected, abs=0.05)

    assert "melody: cached" in cli("melody").output
    changed = cli("melody", "--key", "f#m")
    assert "melody (stale)" in changed.output
    melody = Melody.model_validate_json(run.path("04_melody.json").read_text())
    assert (melody.key, melody.key_source) == ("F# minor", "override")


def test_bad_key_is_a_clean_error(cli: Cli, talk: Talk) -> None:
    result = cli("melody", "--key", "H minor")
    assert result.exit_code == 1
    assert "not a key" in result.output


def test_unknown_instrument_is_a_clean_error(
    cli: Cli, talk: Talk, install: Callable, tmp_path: Path
) -> None:
    _select(cli, talk, install)
    presets = tmp_path / "presets"
    presets.mkdir()
    data = yaml.safe_load((PRESETS_DIR / "cinematic_future_bass.yaml").read_text())
    data["melody"]["render_instrument"] = "theremin"
    (presets / "cinematic_future_bass.yaml").write_text(yaml.safe_dump(data))
    (tmp_path / "config.yaml").write_text(f"presets_dir: {presets}\nruns_dir: runs\n")
    result = cli("melody")
    assert result.exit_code == 1
    assert "Unknown render_instrument 'theremin'" in result.output


def test_listening_set(cli: Cli, talk: Talk, install: Callable) -> None:
    from speech2song.audio.synth import find_soundfont

    run = talk[0]
    _select(cli, talk, install)
    cli("melody")
    paths = write_listening_set(run, find_soundfont(None))
    assert len(paths) == 4 * 5
    melody = Melody.model_validate_json(run.path("04_melody.json").read_text())
    clip = melody.clips[0]
    speech = sf.info(str(run.path(clip.file)))
    illusion = sf.info(str(run.path(f"04_listen/{clip.clip_id}_4_illusion.wav")))
    phrase = clip.bars * 4 * 60 / melody.bpm
    assert illusion.duration >= 3 * phrase + speech.duration - 0.01
