"""The stub music backend: length, determinism, energy, tempo and key."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from speech2song.arrangement import SONG_TAIL_S
from speech2song.backends.music_base import make_backend
from speech2song.backends.music_stub import StubBackend, energy_curve, synthesize
from speech2song.config import AppConfig, load_preset
from speech2song.models import Arrangement, RunOptions, Section, TakeMeta

from .conftest import PRESETS_DIR

PRESET = load_preset("cinematic_future_bass", PRESETS_DIR)
SR = 44100


def _section(sid: str, role: str, start: int, bars: int, energy: float, shape: str = "flat"):
    return Section(id=sid, role=role, start_bar=start, bars=bars, energy=energy,
                   shape=shape, chords=["Am", "F", "C", "G"][:bars] or ["Am"])  # type: ignore[arg-type]  # fmt: skip


ARRANGEMENT = Arrangement(
    bpm=120.0, key="A minor", total_bars=12, total_seconds=24.0,
    sections=[_section("s1", "speech_bed", 0, 4, 0.2), _section("s2", "build", 4, 2, 0.6, "rise"),
              _section("s3", "gap", 6, 1, 0.0), _section("s4", "drop", 7, 4, 1.0),
              _section("s5", "outro", 11, 1, 0.1, "fall")],
)  # fmt: skip


@pytest.fixture(scope="module")
def track() -> np.ndarray:
    return synthesize(StubBackend().request(ARRANGEMENT, PRESET, None), seed=1)


def _rms_db(audio: np.ndarray, start_s: float, end_s: float) -> float:
    block = audio[round(start_s * SR) : round(end_s * SR)]
    return float(20 * np.log10(np.sqrt(np.mean(block**2)) + 1e-12))


def test_length_format_and_level(track: np.ndarray) -> None:
    assert track.dtype == np.float32
    assert track.shape == (round((24.0 + SONG_TAIL_S) * SR), 2)
    assert 20 * np.log10(np.abs(track).max()) == pytest.approx(-3.0, abs=0.01)


def test_energy_follows_the_sections(track: np.ndarray) -> None:
    bed, drop = _rms_db(track, 0.5, 7.5), _rms_db(track, 14.5, 21.5)
    gap = _rms_db(track, 12.6, 13.9)  # the gap bar, minus the build's last reverb
    assert drop > bed + 6
    assert gap < bed
    build_start, build_end = _rms_db(track, 8.0, 9.0), _rms_db(track, 11.0, 12.0)
    assert build_end > build_start  # rising
    curve = energy_curve(StubBackend().request(ARRANGEMENT, PRESET, None), 26 * SR, SR)
    assert curve[round(13 * SR)] == 0.0 and curve[round(15 * SR)] == pytest.approx(1.0)


def test_tempo_reads_as_the_arrangement_bpm(track: np.ndarray) -> None:
    import librosa

    drop = track[round(14 * SR) : round(22 * SR)].mean(axis=1)
    tempo, _ = librosa.beat.beat_track(y=drop, sr=SR)
    bpm = float(np.atleast_1d(tempo)[0])
    assert any(abs(bpm - t) / t < 0.04 for t in (60.0, 120.0, 240.0)), bpm


def test_chords_put_the_key_in_the_chroma(track: np.ndarray) -> None:
    import librosa

    chroma = librosa.feature.chroma_cqt(y=track[: 8 * SR].mean(axis=1), sr=SR).mean(axis=1)
    top = set(np.argsort(chroma)[-4:].tolist())
    assert top <= {9, 0, 4, 5, 7, 11, 2}  # A minor's pitch classes (Am F C G)


def test_takes_are_deterministic_and_differ(tmp_path: Path) -> None:
    backend = StubBackend()
    request = backend.request(ARRANGEMENT, PRESET, None)
    takes = backend.generate(request, tmp_path / "06_music", 2, tmp_path)
    assert [t.meta.file for t in takes] == ["06_music/take_001.wav", "06_music/take_002.wav"]
    one, _ = sf.read(str(takes[0].path), dtype="float32")
    two, _ = sf.read(str(takes[1].path), dtype="float32")
    assert not np.array_equal(one, two)
    np.testing.assert_array_equal(one, synthesize(request, takes[0].meta.seed or 0))
    meta = TakeMeta.model_validate_json((tmp_path / "06_music/take_001.meta.json").read_text())
    assert meta.backend == "stub" and meta.usd == 0.0


def test_stub_takes_are_replaced_and_numbered_after_paid_ones(tmp_path: Path) -> None:
    from speech2song.manifest import write_json

    backend = StubBackend()
    request = backend.request(ARRANGEMENT, PRESET, None)
    music = tmp_path / "06_music"
    backend.generate(request, music, 2, tmp_path)
    again = backend.generate(request, music, 2, tmp_path)  # free: remade, not added
    assert [t.meta.take for t in again] == [1, 2]
    assert again[0].meta.grid_ms == request["grid_ms"]
    paid = TakeMeta(take=5, backend="elevenlabs", file="06_music/take_005.mp3", sample_rate=1,
                    channels=2, seconds=1, request_sha256="x")  # fmt: skip
    write_json(music / "take_005.meta.json", paid)
    takes = backend.generate(request, music, 1, tmp_path)
    assert [t.meta.take for t in takes] == [6]
    assert sorted(p.name for p in music.glob("*.meta.json")) == [
        "take_005.meta.json", "take_006.meta.json"]  # fmt: skip


def test_float_wavs_are_byte_reproducible(tmp_path: Path) -> None:
    from speech2song.audio.synth import write_wav

    audio = np.random.default_rng(0).standard_normal((4410, 2)).astype(np.float32)
    write_wav(tmp_path / "a.wav", audio, SR)
    write_wav(tmp_path / "b.wav", audio, SR)  # libsndfile would stamp each with the time
    assert (tmp_path / "a.wav").read_bytes() == (tmp_path / "b.wav").read_bytes()
    back, rate = sf.read(str(tmp_path / "a.wav"), dtype="float32")
    assert rate == SR and np.array_equal(back, audio)
    assert sf.info(str(tmp_path / "a.wav")).subtype == "FLOAT"


def test_backend_choice() -> None:
    assert make_backend(AppConfig(), RunOptions()).name == "stub"
    assert make_backend(AppConfig(), RunOptions(music_backend="elevenlabs")).name == "elevenlabs"
