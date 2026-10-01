"""Windowed voice isolation, using fake separators (the demucs model is integration-only)."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from speech2song.backends.isolate_base import isolate_file, isolate_stream, window_bounds
from speech2song.errors import StageError

from .fixtures.synth import SR, noise_burst, write_wav


def _stereo_noise(frames: int) -> np.ndarray:
    rng = np.random.default_rng(3)
    return (0.3 * rng.standard_normal((frames, 2))).astype(np.float32)


def _run(data: np.ndarray, separate, *, window: int, context: int, crossfade: int):
    written: list[np.ndarray] = []
    progress: list[tuple[int, int]] = []
    isolate_stream(
        lambda a, b: data[a:b],
        written.append,
        len(data),
        separate,
        window=window,
        context=context,
        crossfade=crossfade,
        progress=lambda done, total: progress.append((done, total)),
    )
    return np.concatenate(written), progress


def test_window_bounds_are_equal_and_cover_everything() -> None:
    assert window_bounds(100, 30) == [0, 33, 67, 100]
    assert window_bounds(10, 30) == [0, 10]
    assert window_bounds(0, 30) == [0, 0]


def test_identity_separator_reproduces_the_input() -> None:
    data = _stereo_noise(10_000)
    out, progress = _run(data, lambda b: b, window=1000, context=300, crossfade=200)
    assert out.shape == data.shape
    np.testing.assert_allclose(out, data, atol=1e-6)
    assert progress[-1] == (10_000, 10_000)


def test_gain_separator_applies_exactly() -> None:
    data = _stereo_noise(7_777)
    out, _ = _run(data, lambda b: 0.5 * b, window=1000, context=300, crossfade=200)
    np.testing.assert_allclose(out, 0.5 * data, atol=1e-6)


def test_windows_are_crossfaded_at_their_boundaries() -> None:
    """A separator that adds its call number shows which window produced each frame."""
    data = np.zeros((3_000, 1), dtype=np.float32)
    calls = iter(range(100))
    out, _ = _run(data, lambda b: b + next(calls), window=1000, context=300, crossfade=200)
    flat = out[:, 0]
    assert np.all(flat[:900] == 0)  # window 0 alone
    assert np.all(flat[1100:1900] == 1)  # window 1 alone
    assert np.all(flat[2100:] == 2)  # window 2 alone
    ramp = flat[900:1100]
    assert np.all(np.diff(ramp) > 0)  # a smooth ramp from window 0 to window 1
    assert 0 < ramp[0] < 0.01 and 0.99 < ramp[-1] < 1


def test_short_input_is_a_single_window() -> None:
    data = _stereo_noise(500)
    calls: list[int] = []
    out, _ = _run(
        data, lambda b: calls.append(len(b)) or b, window=1000, context=300, crossfade=200
    )
    assert calls == [500]
    np.testing.assert_array_equal(out, data)


def test_context_must_cover_half_the_crossfade() -> None:
    with pytest.raises(ValueError, match="context"):
        _run(_stereo_noise(5000), lambda b: b, window=1000, context=50, crossfade=200)


def test_shape_changing_separator_is_an_error() -> None:
    with pytest.raises(StageError, match="shape"):
        _run(_stereo_noise(5000), lambda b: b[:, :1], window=1000, context=300, crossfade=200)


@pytest.mark.parametrize("channels", [1, 2])
def test_isolate_file_keeps_length_rate_and_channels(tmp_path: Path, channels: int) -> None:
    mono = noise_burst(3.3, amp=0.2)
    data = np.stack([mono, mono[::-1]], axis=1) if channels == 2 else mono
    src = write_wav(tmp_path / "source.wav", data)
    dst = tmp_path / "clean.wav"
    isolate_file(src, dst, lambda b: 0.5 * b, window_s=1.0, context_s=0.2, crossfade_s=0.1)
    info = sf.info(str(dst))
    assert (info.frames, info.channels, info.samplerate, info.subtype) == (
        len(mono),
        channels,
        SR,
        "FLOAT",
    )
    out, _ = sf.read(str(dst), dtype="float32", always_2d=True)
    expected = data.reshape(len(mono), channels)
    np.testing.assert_allclose(out, 0.5 * expected, atol=1e-6)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["clean.wav", "source.wav"]
