"""A stand-in for the ElevenLabs client: records calls, returns real MP3 bytes."""

import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from .synth import SR, write_wav


def mp3_bytes(seconds: float, freq: float = 220.0, rate: int = 48000) -> bytes:
    """A stereo chord-ish tone of `seconds`, encoded like the API's default output."""
    t = np.arange(round(seconds * SR)) / SR
    tone = sum(np.sin(2 * np.pi * f * t) for f in (freq, freq * 1.2, freq * 1.5)) * 0.2
    pulse = 0.5 + 0.5 * (np.sin(2 * np.pi * 2 * t) > 0)  # some rhythm for the analysis
    audio = np.stack([tone * pulse, tone * pulse], axis=1).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        wav = write_wav(Path(tmp) / "x.wav", audio)
        mp3 = Path(tmp) / "x.mp3"
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(wav),
                        "-ar", str(rate), "-c:a", "libmp3lame", "-b:a", "192k", str(mp3)],
                       check=True)  # fmt: skip
        return mp3.read_bytes()


class FakeElevenLabs:
    """`music.compose_detailed`, `music.upload` and `user.subscription.get`.

    `responses` scripts compose_detailed: an Exception is raised, anything else is
    ignored and a song of the plan's length is returned.
    """

    def __init__(self, *responses: Any, credits_per_minute: int = 1000) -> None:
        self.responses = list(responses)
        self.compose_calls: list[dict[str, Any]] = []
        self.upload_calls: list[dict[str, Any]] = []
        self.credits = 50_000
        self.credits_per_minute = credits_per_minute
        self.music = SimpleNamespace(compose_detailed=self._compose, upload=self._upload)
        self.user = SimpleNamespace(
            subscription=SimpleNamespace(get=lambda: SimpleNamespace(character_count=self.credits))
        )

    def _compose(self, **kwargs: Any) -> SimpleNamespace:
        self.compose_calls.append(kwargs)
        if self.responses:
            item = self.responses.pop(0)
            if isinstance(item, Exception):
                raise item
        plan = kwargs["composition_plan"]
        ms = sum(c["range"]["end_ms"] - c["range"]["start_ms"] if "song_id" in c
                 else c["duration_ms"] for c in plan["chunks"])  # fmt: skip
        self.credits += round(ms / 60_000 * self.credits_per_minute)
        n = len(self.compose_calls)
        return SimpleNamespace(
            json={"composition_plan": plan, "song_metadata": {"title": f"Song {n}"}},
            audio=mp3_bytes(ms / 1000, freq=220.0 + 10 * n),
            filename=f"song_{n}.mp3",
            song_id=f"song_{n}",
        )

    def _upload(self, *, file: Any, request_options: Any = None) -> SimpleNamespace:
        data = file.read()
        self.upload_calls.append({"bytes": len(data), "request_options": request_options})
        self.credits += self.credits_per_minute // 2
        return SimpleNamespace(song_id=f"ref_{len(self.upload_calls)}")
