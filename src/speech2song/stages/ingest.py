"""Stage 1: ingest the source (00_source.wav) and optionally isolate the voice (01_clean.wav)."""

from pathlib import Path
from typing import ClassVar

from rich.progress import BarColumn, Progress, TimeElapsedColumn, TimeRemainingColumn

from speech2song.audio.io import describe_wav, extract_audio, probe
from speech2song.backends.isolate_base import isolate_file
from speech2song.config import SAMPLE_RATE
from speech2song.manifest import temp_path_for
from speech2song.models import AudioInfo
from speech2song.pipeline import Context, Stage, StagePlan, StageResult

SOURCE = "00_source.wav"
CLEAN = "01_clean.wav"


def _describe(info: AudioInfo) -> str:
    lufs = f"{info.integrated_lufs:.1f} LUFS" if info.integrated_lufs is not None else "silent"
    peak = f"{info.true_peak_dbtp:.1f} dBTP" if info.true_peak_dbtp is not None else "-inf dBTP"
    return f"{info.duration_s:.1f} s, {info.channels} ch, {info.sample_rate} Hz, {lufs}, {peak}"


class IngestStage(Stage):
    name: ClassVar[str] = "ingest"
    version: ClassVar[int] = 1

    def plan(self, ctx: Context) -> StagePlan:
        return StagePlan(
            inputs={"input": Path(ctx.run.manifest.input.path)},
            params={"sample_rate": SAMPLE_RATE, "codec": "pcm_f32le", "resampler": "soxr-28"},
            outputs=[SOURCE],
        )

    def run(self, ctx: Context, plan: StagePlan) -> StageResult:
        source = probe(plan.inputs["input"])
        video = ", with video" if source.has_video else ""
        ctx.say(
            f"  source: {source.codec}, {source.sample_rate} Hz, {source.channels} ch, "
            f"{source.duration_s or 0:.1f} s{video}"
        )
        dst = ctx.run.path(SOURCE)
        extract_audio(plan.inputs["input"], dst, source, SAMPLE_RATE)
        info = describe_wav(dst, SOURCE)
        ctx.run.manifest.source_probe = source
        ctx.run.manifest.source = info
        ctx.say(f"  {SOURCE}: {_describe(info)}")
        return StageResult(
            summary={
                "duration_s": round(info.duration_s, 3),
                "channels": info.channels,
                "integrated_lufs": info.integrated_lufs,
            }
        )


class IsolateStage(Stage):
    """01_clean.wav: demucs vocals when --isolate-voice is on, else a symlink to the source."""

    name: ClassVar[str] = "isolate"
    version: ClassVar[int] = 1

    def plan(self, ctx: Context) -> StagePlan:
        enabled = ctx.run.manifest.options.isolate_voice
        params: dict[str, object] = {"enabled": enabled}
        if enabled:  # device and job count don't change the result
            params["demucs"] = ctx.config.demucs.model_dump(exclude={"device", "jobs"})
        return StagePlan({"source": ctx.run.path(SOURCE)}, params, [CLEAN])

    def run(self, ctx: Context, plan: StagePlan) -> StageResult:
        dst = ctx.run.path(CLEAN)
        if not plan.params["enabled"]:
            _link_or_copy(SOURCE, dst)
            ctx.run.manifest.clean = None
            ctx.say(f"  {CLEAN} -> {SOURCE} (voice isolation off)")
            return StageResult(summary={"isolated": False})

        from speech2song.backends.isolate_demucs import DemucsSeparator

        cfg = ctx.config.demucs
        separate = DemucsSeparator(cfg)
        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=ctx.console,
            transient=True,
        ) as bar:
            task = bar.add_task(f"demucs {cfg.model}", total=1.0)
            isolate_file(
                plan.inputs["source"],
                dst,
                separate,
                window_s=cfg.window_s,
                context_s=cfg.context_s,
                crossfade_s=cfg.crossfade_s,
                progress=lambda done, total: bar.update(task, completed=done / max(total, 1)),
            )
        info = describe_wav(dst, CLEAN)
        ctx.run.manifest.clean = info
        ctx.say(f"  {CLEAN} (demucs {cfg.model} vocals): {_describe(info)}")
        return StageResult(summary={"isolated": True, "model": cfg.model})


def _link_or_copy(target_name: str, link: Path) -> None:
    """Point `link` at a sibling file: relative symlink, or a copy where symlinks fail."""
    tmp = temp_path_for(link)
    tmp.unlink(missing_ok=True)
    try:
        tmp.symlink_to(target_name)
    except OSError:
        import shutil

        shutil.copyfile(link.parent / target_name, tmp)
    tmp.replace(link)
