"""Stage 6: the backing track. `generate` asks the music backend for takes (paid with
ElevenLabs, free with the stub); `take` picks one and conforms it for the mixer (free).

`generate` keeps takes exactly as the backend returned them. Its cache key is the
backend's request (built from the arrangement), not the arrangement file's bytes, so
edits that don't change what would be generated (clip offsets, melody-layer choices)
never repeat it.
"""

import tempfile
from pathlib import Path
from typing import ClassVar

import numpy as np
import soundfile as sf

from speech2song.arrangement import SONG_TAIL_S, bar_seconds
from speech2song.audio.analysis import analyze_take
from speech2song.audio.io import extract_audio, probe
from speech2song.audio.synth import write_wav
from speech2song.backends.music_base import MusicBackend, backend_name, make_backend
from speech2song.config import SAMPLE_RATE, load_preset
from speech2song.costs import CostLog, SpendEstimate, unit_usd
from speech2song.errors import S2SError, StageError
from speech2song.llm.claude import request_digest
from speech2song.manifest import write_json
from speech2song.models import Arrangement, TakeAnalysis, TakeChoice, TakeMeta
from speech2song.music_plan import plan_minutes
from speech2song.pipeline import Context, Stage, StagePlan, StageResult
from speech2song.stages.arrange import ARRANGEMENT
from speech2song.stages.melody import REFERENCE

MUSIC_DIR = "06_music"
SELECTED = "06_music/selected.wav"
ANALYSIS = "06_music/analysis.json"
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

    def _backend(self, ctx: Context) -> MusicBackend:
        options = ctx.run.manifest.options
        return make_backend(ctx.config, options, CostLog(ctx.run.costs_path), ctx.run.id)

    def _request(self, ctx: Context) -> dict | None:
        arrangement = _arrangement(ctx)
        if arrangement is None:
            return None
        preset = load_preset(ctx.run.manifest.preset, ctx.config.presets_dir)
        reference = ctx.run.path(REFERENCE)
        return self._backend(ctx).request(arrangement, preset,
                                          reference if reference.exists() else None)  # fmt: skip

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
        return self._backend(ctx).estimate(request, ctx.config.music_takes)

    def run(self, ctx: Context, plan: StagePlan) -> StageResult:
        backend = self._backend(ctx)
        request = self._request(ctx)
        if request is None:
            raise StageError("generate: the arrangement must exist first")
        ctx.say(f"  generating {plan.params['takes']} take(s) with the {backend.name} backend")
        if ctx.force and self.is_paid(ctx):
            ctx.say("  --force: making new takes (the current ones move to 06_music/archive/)")
        takes = backend.generate(request, ctx.run.path(MUSIC_DIR), plan.params["takes"],
                                 ctx.run.root, say=ctx.say, fresh=ctx.force)  # fmt: skip
        outputs = []
        for take in takes:
            outputs += [take.meta.file, take_meta_path(take.meta.take)]
            ctx.say(f"  take {take.meta.take}: {take.meta.seconds:.1f} s -> {take.meta.file}")
        return StageResult(
            outputs=outputs,
            summary={"backend": backend.name, "takes": len(takes),
                     "usd": round(sum(t.meta.usd for t in takes), 4)},
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


def read_take(path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """A take as float32 frames x channels at `sample_rate`. Compressed files (MP3 from
    ElevenLabs) are decoded and resampled with ffmpeg (soxr)."""
    if path.suffix.lower() == ".wav":
        audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
        if rate == sample_rate:
            return audio
    with tempfile.TemporaryDirectory() as tmpdir:
        decoded = Path(tmpdir) / "take.wav"
        extract_audio(path, decoded, probe(path), sample_rate)
        audio, _ = sf.read(str(decoded), dtype="float32", always_2d=True)
    return audio


def analysis_targets(arrangement: Arrangement) -> dict:
    """What a take is measured against (also the take stage's cache key)."""
    bar = bar_seconds(arrangement.bpm)
    return {
        "bpm": arrangement.bpm,
        "key": arrangement.key,
        "spans_s": [[round(s.start_bar * bar, 4), round((s.start_bar + s.bars) * bar, 4)]
                    for s in arrangement.sections],
        "energies": [s.energy for s in arrangement.sections],
        "expected_s": arrangement.total_seconds,
    }  # fmt: skip


def choose_take(analyses: list[TakeAnalysis], requested: int | None) -> tuple[int, str]:
    if requested is not None:
        return requested, "requested"
    if len(analyses) == 1:
        return analyses[0].take, "only take"
    best = max(analyses, key=lambda a: (a.score, -a.take))
    return best.take, "best score"


class TakeStage(Stage):
    """Analyses every take against the arrangement (tempo, key, energy contour, loudness),
    picks one (`--take N`, else the best score) and conforms it to the song. Free."""

    name: ClassVar[str] = "take"
    version: ClassVar[int] = 2  # 2: analysis and automatic choice
    analysis_version: ClassVar[int] = 3  # 3: tempogram guess refined by autocorrelation

    def plan(self, ctx: Context) -> StagePlan:
        inputs: dict[str, Path] = {}
        for number in range(1, ctx.config.music_takes + 1):
            meta_path = ctx.run.path(take_meta_path(number))
            inputs[f"take {number} meta"] = meta_path
            if meta_path.exists():
                meta = TakeMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
                inputs[f"take {number}"] = ctx.run.path(meta.file)
        arrangement = _arrangement(ctx)
        params = {
            "take": ctx.run.manifest.options.take,
            "sample_rate": SAMPLE_RATE,
            "frames": song_frames(arrangement) if arrangement is not None else None,
            "targets": analysis_targets(arrangement) if arrangement is not None else None,
            "tolerance_bpm": load_preset(ctx.run.manifest.preset,
                                         ctx.config.presets_dir).tempo.tolerance_bpm,
            "analysis": self.analysis_version,
        }  # fmt: skip
        return StagePlan(inputs, params, [SELECTED, ANALYSIS],
                         waits_for={"arrangement": ctx.run.path(ARRANGEMENT)})  # fmt: skip

    def run(self, ctx: Context, plan: StagePlan) -> StageResult:
        requested = plan.params["take"]
        takes = ctx.config.music_takes
        if requested is not None and requested > takes:
            raise StageError(f"take {requested} does not exist (music_takes is {takes})")
        targets = plan.params["targets"]
        frames = plan.params["frames"]
        audio_by_take: dict[int, np.ndarray] = {}
        analyses = []
        for number in range(1, takes + 1):
            path = plan.inputs[f"take {number}"]
            audio = read_take(path)
            audio_by_take[number] = audio
            analysis = analyze_take(
                audio, SAMPLE_RATE, take=number, bpm=targets["bpm"], key=targets["key"],
                spans_s=[tuple(span) for span in targets["spans_s"]],
                energies=targets["energies"], tolerance_bpm=plan.params["tolerance_bpm"],
                expected_s=targets["expected_s"],
            )  # fmt: skip
            analyses.append(analysis)
            tempo = (f"{analysis.tempo_bpm:.1f} BPM (x{analysis.tempo_ratio:g})"
                     if analysis.tempo_bpm else "tempo n/a")  # fmt: skip
            corr = analysis.energy_correlation
            ctx.say(f"  take {number}: {tempo}, key {analysis.key} ({analysis.key_relation}), "
                    f"energy r={'n/a' if corr is None else f'{corr:.2f}'}, "
                    f"{analysis.lufs} LUFS, score {analysis.score:.2f}")  # fmt: skip
            for flag in analysis.flags:
                ctx.say(f"[yellow]    {flag}[/]")
        number, reason = choose_take(analyses, requested)
        write_json(ctx.run.path(ANALYSIS), TakeChoice(chosen=number, reason=reason,
                                                      takes=analyses))  # fmt: skip
        audio = audio_by_take[number]
        expected = round(targets["expected_s"] * SAMPLE_RATE)  # the mix adds its own tail
        if abs(len(audio) - expected) > SAMPLE_RATE * 0.5:
            ctx.say(
                f"[yellow]  take {number} is {len(audio) / SAMPLE_RATE:.1f} s; the "
                f"arrangement is {expected / SAMPLE_RATE:.1f} s (padded or cut)[/]"
            )
        write_wav(ctx.run.path(SELECTED), conform(audio, frames, SAMPLE_RATE), SAMPLE_RATE)
        ctx.say(f"  using take {number} ({reason})")
        return StageResult(summary={"take": number, "reason": reason,
                                    "seconds": round(frames / SAMPLE_RATE, 3)})  # fmt: skip


def current_take(ctx: Context) -> TakeMeta:
    """The take the mix uses: `--take`, else the one the take stage chose."""
    number = ctx.run.manifest.options.take
    if number is None:
        path = ctx.run.path(ANALYSIS)
        if not path.exists():
            raise S2SError("No take has been chosen yet; run `speech2song generate` first.")
        number = TakeChoice.model_validate_json(path.read_text(encoding="utf-8")).chosen
    meta_path = ctx.run.path(take_meta_path(number))
    if not meta_path.exists():
        raise S2SError(f"Take {number} does not exist; run `speech2song generate` first.")
    return TakeMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))


def regenerate_section(ctx: Context, section_id: str, note: str | None) -> TakeMeta | None:
    """Inpaint one section of the current take (paid): estimate, confirm, call, and
    make the result the take's new version. Returns None on a dry run."""
    from speech2song.backends.music_elevenlabs import ElevenLabsBackend
    from speech2song.costs import confirm_spend
    from speech2song.music_plan import inpaint_plan

    meta = current_take(ctx)
    if meta.backend != "elevenlabs" or not meta.song_id:
        raise S2SError(f"Take {meta.take} was not generated with ElevenLabs (stored for "
                       "inpainting); regenerate needs `--music-backend elevenlabs`.")  # fmt: skip
    arrangement = _arrangement(ctx)
    if arrangement is None:
        raise S2SError("No arrangement; run `speech2song arrange` first.")
    preset = load_preset(ctx.run.manifest.preset, ctx.config.presets_dir)
    backend = ElevenLabsBackend(ctx.config, CostLog(ctx.run.costs_path), ctx.run.id)
    reference = ctx.run.path(REFERENCE)
    request = backend.request(arrangement, preset, reference if reference.exists() else None)
    if request_digest(request) != meta.request_sha256:
        raise S2SError(f"The arrangement or music settings changed since take {meta.take} was "
                       "generated, so its sections no longer line up. Run `speech2song "
                       "generate` first.")  # fmt: skip
    try:
        plan, span = inpaint_plan(
            arrangement,
            preset,
            meta.song_id,
            section_id,
            note,
            context_adherence=ctx.config.elevenlabs.context_adherence,
        )
    except ValueError as exc:
        raise S2SError(str(exc)) from exc  # fmt: skip
    minutes = plan_minutes(plan)
    estimate = SpendEstimate(
        service="elevenlabs", model=meta.model or ctx.config.music_model,
        units={"minutes": round(minutes, 3)},
        usd=unit_usd(ctx.config.pricing, "music", minutes),
        description=f"regenerate {section_id} of take {meta.take} "
                    f"({(span[1] - span[0]) / 1000:.1f} s new; priced as the whole song)",
    )  # fmt: skip
    ctx.say(f"Regenerating {section_id} ({span[0] / 1000:.1f}-{span[1] / 1000:.1f} s) of take "
            f"{meta.take}; the rest of the take is kept as it is.")  # fmt: skip
    if not confirm_spend([estimate], console=ctx.console, yes=ctx.yes, dry_run=ctx.dry_run):
        return None
    return backend.inpaint(plan, meta, ctx.run.path(MUSIC_DIR), ctx.run.root,
                           section=section_id, note=note, span_ms=span, say=ctx.say)  # fmt: skip
