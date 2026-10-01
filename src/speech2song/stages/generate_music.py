"""Stage 6: the backing track. `generate` asks the music backend for takes (paid with
ElevenLabs, free with the stub); `take` picks one and conforms it for the mixer (free).

`generate` keeps takes exactly as the backend returned them. Its cache key is the
backend's request (built from the arrangement), not the arrangement file's bytes, so
edits that don't change what would be generated (clip offsets, melody-layer choices)
never repeat it.
"""

from pathlib import Path
from typing import ClassVar

import numpy as np
import soundfile as sf

from speech2song.arrangement import SONG_TAIL_S
from speech2song.audio.synth import write_wav
from speech2song.backends.music_base import backend_name, make_backend
from speech2song.config import SAMPLE_RATE, load_preset
from speech2song.costs import SpendEstimate
from speech2song.errors import StageError
from speech2song.llm.claude import request_digest
from speech2song.models import Arrangement, TakeMeta
from speech2song.pipeline import Context, Stage, StagePlan, StageResult
from speech2song.stages.arrange import ARRANGEMENT
from speech2song.stages.melody import REFERENCE

MUSIC_DIR = "06_music"
SELECTED = "06_music/selected.wav"
END_FADE_S = 0.05  # when a take is longer than the song and gets cut


def take_meta_path(number: int) -> str:
    return f"{MUSIC_DIR}/take_{number:03d}.meta.json"


def _arrangement(ctx: Context) -> Arrangement | None:
    path = ctx.run.path(ARRANGEMENT)
    if not path.exists():
        return None
    return Arrangement.model_validate_json(path.read_text(encoding="utf-8"))


class GenerateStage(Stage):
    name: ClassVar[str] = "generate"
    version: ClassVar[int] = 1

    def is_paid(self, ctx: Context) -> bool:
        return backend_name(ctx.config, ctx.run.manifest.options) != "stub"

    def _request(self, ctx: Context) -> dict | None:
        arrangement = _arrangement(ctx)
        if arrangement is None or self.is_paid(ctx):  # paid backends arrive in M5
            return None
        preset = load_preset(ctx.run.manifest.preset, ctx.config.presets_dir)
        backend = make_backend(ctx.config, ctx.run.manifest.options)
        reference = ctx.run.path(REFERENCE)
        return backend.request(arrangement, preset, reference if reference.exists() else None)

    def plan(self, ctx: Context) -> StagePlan:
        takes = ctx.config.music_takes
        request = self._request(ctx)
        params = {
            "backend": backend_name(ctx.config, ctx.run.manifest.options),
            "takes": takes,
            "request": request_digest(request) if request is not None else None,
        }
        outputs = [take_meta_path(n) for n in range(1, takes + 1)]
        return StagePlan({}, params, outputs, waits_for={"arrangement": ctx.run.path(ARRANGEMENT)})

    def estimate(self, ctx: Context, plan: StagePlan) -> list[SpendEstimate]:
        request = self._request(ctx)
        if request is None:
            return []
        backend = make_backend(ctx.config, ctx.run.manifest.options)
        return backend.estimate(request, ctx.config.music_takes)

    def run(self, ctx: Context, plan: StagePlan) -> StageResult:
        backend = make_backend(ctx.config, ctx.run.manifest.options)
        request = self._request(ctx)
        if request is None:
            raise StageError("generate: the arrangement must exist first")
        ctx.say(f"  generating {plan.params['takes']} take(s) with the {backend.name} backend")
        takes = backend.generate(request, ctx.run.path(MUSIC_DIR), plan.params["takes"],
                                 ctx.run.root)  # fmt: skip
        outputs = []
        for take in takes:
            outputs += [take.meta.file, take_meta_path(take.meta.take)]
            ctx.say(f"  take {take.meta.take}: {take.meta.seconds:.1f} s -> {take.meta.file}")
        return StageResult(
            outputs=outputs,
            summary={"backend": backend.name, "takes": len(takes),
                     "usd": sum(t.meta.usd for t in takes)},
        )  # fmt: skip


def conform(audio: np.ndarray, frames: int, sample_rate: int) -> np.ndarray:
    """Stereo, exactly `frames` long: padded with silence, or cut with a short fade."""
    audio = audio if audio.shape[1] == 2 else np.repeat(audio[:, :1], 2, axis=1)
    if len(audio) >= frames:
        out = np.array(audio[:frames], dtype=np.float32)
        fade = min(frames, round(END_FADE_S * sample_rate))
        if fade:
            out[frames - fade :] *= np.linspace(1, 0, fade, dtype=np.float32)[:, None]
        return out
    pad = np.zeros((frames - len(audio), 2), dtype=np.float32)
    return np.concatenate([audio.astype(np.float32), pad])


def song_frames(arrangement: Arrangement, sample_rate: int = SAMPLE_RATE) -> int:
    return round((arrangement.total_seconds + SONG_TAIL_S) * sample_rate)


class TakeStage(Stage):
    """Picks the take to mix (`--take N`, default 1) and conforms it to the song."""

    name: ClassVar[str] = "take"
    version: ClassVar[int] = 1

    def plan(self, ctx: Context) -> StagePlan:
        number = ctx.run.manifest.options.take or 1
        meta_path = ctx.run.path(take_meta_path(number))
        inputs: dict[str, Path] = {"meta": meta_path}
        if meta_path.exists():
            meta = TakeMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
            inputs["audio"] = ctx.run.path(meta.file)
        arrangement = _arrangement(ctx)  # only the song's length matters here
        frames = song_frames(arrangement) if arrangement is not None else None
        params = {"take": number, "sample_rate": SAMPLE_RATE, "frames": frames}
        return StagePlan(inputs, params, [SELECTED],
                         waits_for={"arrangement": ctx.run.path(ARRANGEMENT)})  # fmt: skip

    def run(self, ctx: Context, plan: StagePlan) -> StageResult:
        number = plan.params["take"]
        if number > ctx.config.music_takes:
            raise StageError(f"take {number} does not exist (music_takes is "
                             f"{ctx.config.music_takes})")  # fmt: skip
        audio, rate = sf.read(str(plan.inputs["audio"]), dtype="float32", always_2d=True)
        if rate != SAMPLE_RATE:
            import librosa

            audio = librosa.resample(audio.T, orig_sr=rate, target_sr=SAMPLE_RATE,
                                     res_type="soxr_vhq").T  # fmt: skip
        frames = plan.params["frames"]
        if abs(len(audio) - frames) > SAMPLE_RATE * 0.5:
            ctx.say(f"[yellow]  take {number} is {len(audio) / SAMPLE_RATE:.1f} s; the song is "
                    f"{frames / SAMPLE_RATE:.1f} s (padded or cut to fit)[/]")  # fmt: skip
        out = conform(audio, frames, SAMPLE_RATE)
        write_wav(ctx.run.path(SELECTED), out, SAMPLE_RATE)
        ctx.say(f"  using take {number} ({plan.inputs['audio'].name})")
        return StageResult(summary={"take": number, "seconds": round(frames / SAMPLE_RATE, 3)})
