"""ffprobe/ffmpeg wrappers: pure parsing tests, plus real ffmpeg on tiny synthetic files."""

import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from speech2song.audio.io import (
    decode_mono,
    describe_wav,
    extract_args,
    extract_audio,
    measure_loudness,
    parse_ebur128_summary,
    parse_probe,
    probe,
)
from speech2song.errors import S2SError
from speech2song.manifest import sha256_file
from speech2song.models import SourceProbe

from .fixtures.synth import SR, make_video, noise_burst, sine, sweep, write_wav

MP3_WITH_COVER_ART = {
    "streams": [
        {
            "codec_type": "audio",
            "codec_name": "mp3",
            "sample_rate": "48000",
            "channels": 2,
            "channel_layout": "stereo",
            "bit_rate": "128000",
        },
        {"codec_type": "video", "codec_name": "mjpeg", "disposition": {"attached_pic": 1}},
    ],
    "format": {"format_name": "mp3", "duration": "775.008000", "bit_rate": "128009"},
}

SUMMARY = """[Parsed_ebur128_0 @ 0x601bbd282140] Summary:

  Integrated loudness:
    I:         -41.1 LUFS
    Threshold: -51.1 LUFS

  Loudness range:
    LRA:         0.0 LU
    Threshold: -61.1 LUFS
    LRA low:   -41.1 LUFS
    LRA high:  -41.1 LUFS

  Sample peak:
    Peak:      -38.1 dBFS

  True peak:
    Peak:      -38.0 dBFS
"""


def _source(sample_rate: int, channels: int) -> SourceProbe:
    return SourceProbe(
        format_name="wav",
        duration_s=1.0,
        has_video=False,
        audio_stream=0,
        codec="pcm_s16le",
        sample_rate=sample_rate,
        channels=channels,
    )


def test_parse_probe_ignores_cover_art() -> None:
    source = parse_probe(MP3_WITH_COVER_ART)
    assert source.has_video is False
    assert (source.codec, source.sample_rate, source.channels) == ("mp3", 48000, 2)
    assert source.duration_s == pytest.approx(775.008)
    assert source.bit_rate == 128000


def test_parse_probe_detects_video() -> None:
    data = {
        "streams": [
            {"codec_type": "video", "codec_name": "h264", "disposition": {"attached_pic": 0}},
            {"codec_type": "audio", "codec_name": "aac", "sample_rate": "44100", "channels": 1},
        ],
        "format": {"format_name": "mov,mp4", "duration": "3.0"},
    }
    assert parse_probe(data).has_video is True


def test_parse_probe_requires_audio() -> None:
    with pytest.raises(S2SError, match="no audio stream"):
        parse_probe({"streams": [{"codec_type": "video"}], "format": {}})


def test_extract_args_resample_and_downmix_only_when_needed() -> None:
    args = extract_args(Path("in.mp3"), Path("out.wav"), _source(48000, 2), 44100)
    assert "aresample=44100:resampler=soxr:precision=28" in args
    assert "-ac" not in args
    assert args[args.index("-c:a") + 1] == "pcm_f32le"
    assert args[-1] == "out.wav"
    args = extract_args(Path("in.wav"), Path("out.wav"), _source(44100, 6), 44100)
    assert "-af" not in args
    assert args[args.index("-ac") + 1] == "2"


def test_parse_ebur128_summary() -> None:
    assert parse_ebur128_summary("noise before\n" + SUMMARY) == {
        "integrated_lufs": -41.1,
        "loudness_range_lu": 0.0,
        "sample_peak_dbfs": -38.1,
        "true_peak_dbtp": -38.0,
    }


def test_parse_ebur128_summary_of_silence() -> None:
    silent = SUMMARY.replace("-41.1 LUFS\n    Threshold: -51.1", "-70.0 LUFS\n    Threshold: 0.0")
    silent = silent.replace("-38.1 dBFS", "-inf dBFS").replace("-38.0 dBFS", "-inf dBFS")
    values = parse_ebur128_summary(silent)
    assert values == dict.fromkeys(values, None)


def test_parse_ebur128_summary_missing() -> None:
    with pytest.raises(S2SError, match="no loudness summary"):
        parse_ebur128_summary("ffmpeg said nothing useful")


@pytest.mark.parametrize(("rate", "channels"), [(48000, 2), (44100, 1), (22050, 1)])
def test_extract_converts_to_44k_float(tmp_path: Path, rate: int, channels: int) -> None:
    mono = sweep(100, 8000, 1.5, sr=rate)
    data = np.stack([mono, 0.5 * mono], axis=1) if channels == 2 else mono
    src = write_wav(tmp_path / "in.wav", data, sr=rate, subtype="PCM_16")
    dst = tmp_path / "out.wav"
    extract_audio(src, dst, probe(src), SR)
    info = sf.info(str(dst))
    assert (info.samplerate, info.channels, info.subtype) == (SR, channels, "FLOAT")
    assert abs(info.frames - round(1.5 * SR)) <= 1
    assert sorted(p.name for p in tmp_path.iterdir()) == ["in.wav", "out.wav"]


def test_extract_downmixes_surround_to_stereo(tmp_path: Path) -> None:
    data = np.stack([sine(220 * (k + 1), 0.5) for k in range(6)], axis=1)
    src = write_wav(tmp_path / "surround.wav", data)
    dst = tmp_path / "out.wav"
    extract_audio(src, dst, probe(src), SR)
    assert sf.info(str(dst)).channels == 2


def test_extract_passes_44k_float_through_bit_identically(tmp_path: Path) -> None:
    data = np.stack([noise_burst(1.0, seed=1), noise_burst(1.0, seed=2)], axis=1) * 0.5
    src = write_wav(tmp_path / "in.wav", data)
    dst = tmp_path / "out.wav"
    extract_audio(src, dst, probe(src), SR)
    out, rate = sf.read(str(dst), dtype="float32")
    assert rate == SR
    assert np.array_equal(out, data)


def test_extract_is_reproducible(tmp_path: Path) -> None:
    src = write_wav(tmp_path / "in.wav", sweep(50, 5000, 1.0, sr=48000), sr=48000)
    first, second = tmp_path / "a.wav", tmp_path / "b.wav"
    extract_audio(src, first, probe(src), SR)
    extract_audio(src, second, probe(src), SR)
    assert sha256_file(first) == sha256_file(second)


def test_extract_from_video(tmp_path: Path) -> None:
    src = make_video(tmp_path / "clip.mp4", seconds=1.0)
    source = probe(src)
    assert source.has_video
    assert (source.sample_rate, source.channels) == (48000, 2)
    dst = tmp_path / "out.wav"
    extract_audio(src, dst, source, SR)
    info = sf.info(str(dst))
    assert (info.samplerate, info.channels) == (SR, 2)
    assert info.frames / SR == pytest.approx(1.0, abs=0.05)


def test_video_without_audio_is_rejected(tmp_path: Path) -> None:
    src = tmp_path / "silent.mp4"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i",
         "testsrc=size=64x64:rate=10:duration=0.5", "-c:v", "mpeg4", str(src)],
        check=True,
    )  # fmt: skip
    with pytest.raises(S2SError, match="no audio stream"):
        probe(src)


def test_corrupt_input_is_a_clean_error(tmp_path: Path) -> None:
    src = tmp_path / "broken.mp3"
    src.write_bytes(b"this is not audio" * 100)
    with pytest.raises(S2SError, match="ffprobe failed"):
        probe(src)


def test_loudness_of_a_known_sine(tmp_path: Path) -> None:
    path = write_wav(tmp_path / "tone.wav", sine(997, 3.0, amp=0.1))  # -20 dBFS peak, mono
    loudness = measure_loudness(path)
    assert loudness["integrated_lufs"] == pytest.approx(-23.0, abs=0.3)
    assert loudness["sample_peak_dbfs"] == pytest.approx(-20.0, abs=0.1)
    assert loudness["true_peak_dbtp"] == pytest.approx(-20.0, abs=0.3)


def test_loudness_of_silence(tmp_path: Path) -> None:
    path = write_wav(tmp_path / "silence.wav", np.zeros(SR, dtype=np.float32))
    assert set(measure_loudness(path).values()) == {None}


def test_describe_wav(tmp_path: Path) -> None:
    path = write_wav(tmp_path / "tone.wav", np.stack([sine(440, 2.0)] * 2, axis=1))
    info = describe_wav(path, "tone.wav")
    assert (info.path, info.sample_rate, info.channels, info.frames) == ("tone.wav", SR, 2, 2 * SR)
    assert info.duration_s == pytest.approx(2.0)
    assert info.subtype == "FLOAT"
    assert info.integrated_lufs is not None


def test_decode_mono_for_asr(tmp_path: Path) -> None:
    path = write_wav(tmp_path / "tone.wav", np.stack([sine(440, 2.0)] * 2, axis=1))
    samples = decode_mono(path, 16000)
    assert samples.dtype == np.float32
    assert abs(len(samples) - 32000) <= 16
