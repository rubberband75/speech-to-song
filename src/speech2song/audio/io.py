"""ffprobe/ffmpeg wrappers and WAV inspection.

Pure helpers (`parse_probe`, `extract_args`, `parse_ebur128_summary`) are kept apart from
the subprocess calls so they can be tested without media files.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from speech2song.errors import S2SError
from speech2song.manifest import temp_path_for
from speech2song.models import AudioInfo, SourceProbe

# Integrated loudness at or below this is ffmpeg's "nothing passed the gate" value.
SILENCE_LUFS = -70.0


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise S2SError(f"`{name}` was not found on PATH. Install ffmpeg (it provides {name}).")
    return path


def run_tool(args: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(args, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        tail = "\n".join(exc.stderr.strip().splitlines()[-5:])
        raise S2SError(f"{Path(args[0]).name} failed:\n{tail}") from exc


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_probe(data: dict[str, Any]) -> SourceProbe:
    """Summarize ffprobe JSON. The first audio stream is the one we use."""
    streams = data.get("streams", [])
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    if not audio:
        raise S2SError("The input has no audio stream.")
    first = audio[0]
    has_video = any(
        s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")
        for s in streams
    )  # cover art in MP3/M4A files shows up as an "attached picture" video stream
    fmt = data.get("format", {})
    bit_rate = _float(first.get("bit_rate")) or _float(fmt.get("bit_rate"))
    return SourceProbe(
        format_name=fmt.get("format_name", "unknown"),
        duration_s=_float(fmt.get("duration")) or _float(first.get("duration")),
        has_video=has_video,
        audio_stream=0,
        codec=first.get("codec_name"),
        sample_rate=int(first["sample_rate"]),
        channels=int(first["channels"]),
        channel_layout=first.get("channel_layout"),
        bit_rate=int(bit_rate) if bit_rate else None,
    )


def probe(path: Path) -> SourceProbe:
    proc = run_tool(
        [
            require_tool("ffprobe"),
            "-v", "error",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(path),
        ]
    )  # fmt: skip
    return parse_probe(json.loads(proc.stdout))


def extract_args(src: Path, dst: Path, source: SourceProbe, sample_rate: int) -> list[str]:
    """ffmpeg arguments for a float32 WAV at `sample_rate`, keeping mono/stereo.

    Resampling (soxr, very high quality) only happens when the rate differs, so a
    float WAV at the target rate passes through bit-identically. Bitexact flags keep
    encoder metadata out of the file, so its hash is reproducible.
    """
    args = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src),
        "-map", f"0:a:{source.audio_stream}",
        "-vn", "-sn", "-dn", "-map_metadata", "-1",
    ]  # fmt: skip
    if source.sample_rate != sample_rate:
        args += ["-af", f"aresample={sample_rate}:resampler=soxr:precision=28"]
    if source.channels > 2:
        args += ["-ac", "2"]
    args += [
        "-c:a", "pcm_f32le",
        "-rf64", "auto",
        "-fflags", "+bitexact", "-flags:a", "+bitexact",
        "-f", "wav", str(dst),
    ]  # fmt: skip
    return args


def extract_audio(src: Path, dst: Path, source: SourceProbe, sample_rate: int) -> None:
    tmp = temp_path_for(dst)
    args = extract_args(src, tmp, source, sample_rate)
    args[0] = require_tool("ffmpeg")
    try:
        run_tool(args)
        tmp.replace(dst)
    finally:
        tmp.unlink(missing_ok=True)


_SUMMARY_FIELDS = {
    "integrated_lufs": r"Integrated loudness:\s+I:\s+(\S+) LUFS",
    "loudness_range_lu": r"Loudness range:\s+LRA:\s+(\S+) LU",
    "sample_peak_dbfs": r"Sample peak:\s+Peak:\s+(\S+) dBFS",
    "true_peak_dbtp": r"True peak:\s+Peak:\s+(\S+) dBFS",
}


def parse_ebur128_summary(stderr: str) -> dict[str, float | None]:
    """Read the Summary block that ffmpeg's ebur128 filter logs at the end."""
    start = stderr.rfind("Summary:")
    if start < 0:
        raise S2SError("ffmpeg ebur128 printed no loudness summary.")
    summary = stderr[start:]
    values: dict[str, float | None] = {}
    for key, pattern in _SUMMARY_FIELDS.items():
        match = re.search(pattern, summary)
        value = _float(match.group(1)) if match else None
        values[key] = value if value is not None and value != float("-inf") else None
    lufs = values["integrated_lufs"]
    if lufs is not None and lufs <= SILENCE_LUFS:
        values["integrated_lufs"] = None
        values["loudness_range_lu"] = None
    return values


def measure_loudness(path: Path) -> dict[str, float | None]:
    """EBU R128 integrated loudness, loudness range, sample and true peak (streaming)."""
    proc = run_tool(
        [
            require_tool("ffmpeg"), "-nostdin", "-hide_banner", "-nostats",
            "-i", str(path),
            "-af", "ebur128=peak=sample+true:framelog=quiet",
            "-f", "null", "-",
        ]
    )  # fmt: skip
    return parse_ebur128_summary(proc.stderr)


def decode_mono(path: Path, sample_rate: int) -> np.ndarray:
    """Decode any input to mono float32 at `sample_rate` (soxr), e.g. 16 kHz for ASR."""
    args = [
        require_tool("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(path), "-map", "0:a:0", "-ac", "1",
        "-af", f"aresample={sample_rate}:resampler=soxr",
        "-f", "f32le", "-c:a", "pcm_f32le", "-",
    ]  # fmt: skip
    try:
        proc = subprocess.run(args, capture_output=True, check=True)
    except subprocess.CalledProcessError as exc:
        tail = "\n".join(exc.stderr.decode(errors="replace").strip().splitlines()[-5:])
        raise S2SError(f"ffmpeg failed to decode {path.name}:\n{tail}") from exc
    return np.frombuffer(proc.stdout, dtype=np.float32)


def describe_wav(path: Path, shown_path: str) -> AudioInfo:
    info = sf.info(str(path))
    return AudioInfo(
        path=shown_path,
        sample_rate=info.samplerate,
        channels=info.channels,
        frames=info.frames,
        duration_s=info.frames / info.samplerate,
        subtype=info.subtype,
        **measure_loudness(path),
    )
