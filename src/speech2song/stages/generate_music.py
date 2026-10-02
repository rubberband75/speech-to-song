"""Stage 6: the backing track. `generate` asks the music backend for takes (paid with
ElevenLabs, free with the stub); `take` picks one and conforms it for the mixer (free).

`generate` keeps takes exactly as the backend returned them. Its cache key is the
backend's request (built from the arrangement), not the arrangement file's bytes, so
edits that don't change what would be generated (clip offsets, melody-layer choices)
never repeat it. Takes of older requests stay available while their timing fits; `take`
picks among the newest by default, and `--take N` can pick any of them.
"""

import tempfile
from pathlib import Path
from typing import ClassVar

import numpy as np
import soundfile as sf

from speech2song.arrangement import (
    SONG_TAIL_S,
    bar_seconds,
    energy_bounds,
    music_bars,
    music_total_bars,
)
from speech2song.audio.analysis import analyze_take
from speech2song.audio.io import extract_audio, probe
from speech2song.audio.synth import write_wav
from speech2song.backends.music_base import (
    MelodyFiles,
    MusicBackend,
    backend_name,
    make_backend,
    take_fits,
    take_metas,
)
from speech2song.config import SAMPLE_RATE, load_preset
from speech2song.costs import CostLog, SpendEstimate, unit_usd
from speech2song.errors import S2SError, StageError
from speech2song.llm.claude import request_digest
from speech2song.manifest import write_json
from speech2song.models import Arrangement, QuoteMelodies, TakeAnalysis, TakeChoice, TakeMeta
from speech2song.music_plan import grid_ms, plan_minutes, tail_ms
from speech2song.pipeline import Context, Stage, StagePlan, StageResult
from speech2song.stages.arrange import ARRANGEMENT, usable_arrangement
from speech2song.stages.melody import QUOTE_INDEX, QUOTE_MELODIES, REFERENCE

MUSIC_DIR = "06_music"
SELECTED = "06_music/selected.wav"
ANALYSIS = "06_music/analysis.json"
END_FADE_S = 0.05  # when a take is longer than the song and gets cut
STOP_FADE_S = 0.005  # where the music stops for a passage played alone (the mix rings it)
RETURN_FADE_S = 0.003  # where it comes back, on the downbeat


def take_meta_path(number: int) -> str:
    return f"{MUSIC_DIR}/take_{number:03d}.meta.json"


def _arrangement(ctx: Context) -> Arrangement | None:
    path = ctx.run.path(ARRANGEMENT)
    if not path.exists():
        return None
    return Arrangement.model_validate_json(path.read_text(encoding="utf-8"))


def melody_files(ctx: Context) -> MelodyFiles:
    """The melody renders that exist in this run (runs from before M7 have no quotes)."""
    reference, quotes, index = (ctx.run.path(p) for p in (REFERENCE, QUOTE_MELODIES, QUOTE_INDEX))
    return MelodyFiles(
        reference=reference if reference.exists() else None,
        quotes=quotes if quotes.exists() and index.exists() else None,
        index=QuoteMelodies.model_validate_json(index.read_text()) if index.exists() else None,
    )


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
        return self._backend(ctx).request(arrangement, preset, melody_files(ctx))

    def plan(self, ctx: Context) -> StagePlan:
        takes = ctx.config.music_takes
        request = self._request(ctx)
        params = {
            "backend": backend_name(ctx.config, ctx.run.manifest.options),
            "takes": takes,
            "request": request_digest(request) if request is not None else None,
        }
        # The take numbers are known only after running (see StageResult.outputs).
        return StagePlan({}, params, [], waits_for={"arrangement": ctx.run.path(ARRANGEMENT)})

    def estimate(self, ctx: Context, plan: StagePlan) -> list[SpendEstimate]:
        request = self._request(ctx)
        if request is None:
            return []
        return self._backend(ctx).estimate(request, ctx.config.music_takes)

    def run(self, ctx: Context, plan: StagePlan) -> StageResult:
        usable_arrangement(ctx)  # never pay for music for a broken arrangement
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
        pinned = ctx.run.manifest.options.take
        if pinned is not None and pinned not in {take.meta.take for take in takes}:
            ctx.say(f"[yellow]  take {pinned} is still the one mixed (--take); use "
                    "`mix --take auto` for the best of these[/]")  # fmt: skip
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


def cut_at(music: np.ndarray, end: int, sr: int) -> np.ndarray:
    """The music up to sample `end`, with a 5 ms fade so the cut never clicks (the mix
    rings out what was sounding there)."""
    out = np.array(music[:end], dtype=np.float32)
    fade = min(len(out), round(STOP_FADE_S * sr))
    if fade and len(music) > end:
        out[len(out) - fade :] *= np.linspace(1, 0, fade, dtype=np.float32)[:, None]
    return out


def splice_rests(music: np.ndarray, arrangement: Arrangement, frames: int,
                 sr: int) -> np.ndarray:  # fmt: skip
    """The take, generated in music time, laid onto the song: each passage played alone
    opens a silence in it, so the music on either side stays exactly as generated (a
    build still runs into its drop as the model composed it). Music generated past the
    song's last bar (`tail_ms`) is cut away on that bar line. Stereo, `frames` long."""
    bar = bar_seconds(arrangement.bpm) * sr
    if tail_ms(arrangement):  # the music generated past the last bar is cut away
        music = cut_at(music, round(music_total_bars(arrangement) * bar), sr)
    if not any(section.rest for section in arrangement.sections):
        return conform(music, frames, sr)
    music = music if music.shape[1] == 2 else np.repeat(music[:, :1], 2, axis=1)
    starts = music_bars(arrangement)
    out = np.zeros((frames, 2), dtype=np.float32)
    runs: list[list[int]] = []  # neighbouring sections with music, by index
    for i, section in enumerate(arrangement.sections):
        if section.rest:
            continue
        if runs and runs[-1][-1] == i - 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    sections = arrangement.sections
    for run in runs:
        first, last = sections[run[0]], sections[run[-1]]
        song_a = round(first.start_bar * bar)
        rest_after = run[-1] + 1 < len(sections)
        song_b = round((last.start_bar + last.bars) * bar) if rest_after else frames
        source = round(starts[run[0]] * bar)
        block = np.array(music[source : source + song_b - song_a], dtype=np.float32)
        if run[0] > 0:  # back after a silence, on the downbeat
            fade = min(len(block), round(RETURN_FADE_S * sr))
            block[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)[:, None]
        cut = rest_after or len(music) - source > song_b - song_a
        fade = min(len(block), round((STOP_FADE_S if rest_after else END_FADE_S) * sr))
        if cut and fade:
            block[len(block) - fade :] *= np.linspace(1, 0, fade, dtype=np.float32)[:, None]
        out[song_a : song_a + len(block)] = block
    return out


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
    """What a take is measured against (also the take stage's cache key), in music time.
    Gaps are left out (the mix lifts them, whatever the music does there), and so are
    passages played alone (no music is generated for them)."""
    bar = bar_seconds(arrangement.bpm)
    measured = [(s, (a + b) / 2, first) for s, (a, b), first
                in zip(arrangement.sections, energy_bounds(arrangement.sections),
                       music_bars(arrangement), strict=True)
                if not s.silent and not s.rest]  # fmt: skip
    return {
        "bpm": arrangement.bpm,
        "key": arrangement.key,
        "sections": [s.id for s, _, _ in measured],
        "spans_s": [[round(first * bar, 4), round((first + s.bars) * bar, 4)]
                    for s, _, first in measured],
        "energies": [round(energy, 4) for _, energy, _ in measured],
        "expected_s": round(music_total_bars(arrangement) * bar + tail_ms(arrangement) / 1000, 4),
    }  # fmt: skip


def choose_take(analyses: list[TakeAnalysis], requested: int | None) -> tuple[int, str]:
    if requested is not None:
        return requested, "requested"
    if len(analyses) == 1:
        return analyses[0].take, "only take"
    best = max(analyses, key=lambda a: (a.score, -a.take))
    return best.take, "best score"


class TakeStage(Stage):
    """Analyses every take that fits the arrangement (tempo, key, energy contour,
    loudness), picks one (`--take N`, else the best score among the takes made for the
    current request) and conforms it to the song. Free."""

    name: ClassVar[str] = "take"
    version: ClassVar[int] = 4  # 3: takes of older requests that still fit; 4: rests spliced
    # 4: silent sections left out; 5: music time; 6: near-silent sections flagged
    analysis_version: ClassVar[int] = 6

    def plan(self, ctx: Context) -> StagePlan:
        inputs: dict[str, Path] = {}
        for meta in take_metas(ctx.run.path(MUSIC_DIR)):
            inputs[f"take {meta.take} meta"] = ctx.run.path(take_meta_path(meta.take))
            inputs[f"take {meta.take}"] = ctx.run.path(meta.file)
        arrangement = _arrangement(ctx)
        request = GenerateStage()._request(ctx)
        params = {
            "take": ctx.run.manifest.options.take,
            "backend": backend_name(ctx.config, ctx.run.manifest.options),
            "request": request_digest(request) if request is not None else None,
            "grid_ms": grid_ms(arrangement) if arrangement is not None else None,
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
        usable_arrangement(ctx)
        params = plan.params
        grid, digest = params["grid_ms"], params["request"]
        metas = [
            m
            for m in take_metas(ctx.run.path(MUSIC_DIR))
            if m.backend == params["backend"] and take_fits(m, ctx.run.root, grid, digest)
        ]
        if not metas:
            raise StageError("no take fits the arrangement; run `speech2song generate`")
        requested = params["take"]
        if requested is not None and requested not in {meta.take for meta in metas}:
            raise StageError(f"take {requested} is not available; takes that fit the "
                             f"arrangement: {', '.join(str(m.take) for m in metas)}")  # fmt: skip
        targets = params["targets"]
        frames = params["frames"]
        audio_by_take: dict[int, np.ndarray] = {}
        analyses = []
        for meta in metas:
            audio = read_take(ctx.run.path(meta.file))
            audio_by_take[meta.take] = audio
            analysis = analyze_take(
                audio, SAMPLE_RATE, take=meta.take, bpm=targets["bpm"], key=targets["key"],
                spans_s=[tuple(span) for span in targets["spans_s"]],
                energies=targets["energies"], tolerance_bpm=params["tolerance_bpm"],
                expected_s=targets["expected_s"], section_ids=targets["sections"],
            ).model_copy(update={"sections": targets["sections"],
                                 "current": meta.request_sha256 == digest})  # fmt: skip
            analyses.append(analysis)
            tempo = (f"{analysis.tempo_bpm:.1f} BPM (x{analysis.tempo_ratio:g})"
                     if analysis.tempo_bpm else "tempo n/a")  # fmt: skip
            corr = "n/a" if analysis.energy_correlation is None else \
                f"{analysis.energy_correlation:.2f}"  # fmt: skip
            older = "" if analysis.current else " (older request)"
            ctx.say(f"  take {meta.take}{older}: {tempo}, key {analysis.key} "
                    f"({analysis.key_relation}), energy r={corr}, {analysis.lufs} LUFS, "
                    f"score {analysis.score:.2f}")  # fmt: skip
            for flag in analysis.flags:
                ctx.say(f"[yellow]    {flag}[/]")
        newest = [a for a in analyses if a.current]
        number, reason = choose_take(newest or analyses, requested)
        write_json(ctx.run.path(ANALYSIS), TakeChoice(chosen=number, reason=reason,
                                                      takes=analyses))  # fmt: skip
        audio = audio_by_take[number]
        expected = round(targets["expected_s"] * SAMPLE_RATE)  # the mix adds its own tail
        if abs(len(audio) - expected) > SAMPLE_RATE * 0.5:
            ctx.say(
                f"[yellow]  take {number} is {len(audio) / SAMPLE_RATE:.1f} s; the "
                f"arrangement's music is {expected / SAMPLE_RATE:.1f} s (padded or cut)[/]"
            )
        arrangement = usable_arrangement(ctx)
        rests = [s.id for s in arrangement.sections if s.rest]
        if rests:
            ctx.say(f"  silence spliced in for the passages played alone ({', '.join(rests)})")
        write_wav(ctx.run.path(SELECTED), splice_rests(audio, arrangement, frames, SAMPLE_RATE),
                  SAMPLE_RATE)  # fmt: skip
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


def regenerate_sections(
    ctx: Context, specs: list[str], note: str | None, adherence: str | None = None
) -> TakeMeta | None:
    """Inpaint neighbouring sections of the current take (paid): estimate, confirm, call,
    and make the result the take's new version. `adherence` overrides the config's
    context_adherence for this call (lower lets the new part differ more from the old).
    Returns None on a dry run."""
    from speech2song.backends.music_base import take_grid
    from speech2song.backends.music_elevenlabs import ElevenLabsBackend
    from speech2song.costs import confirm_spend
    from speech2song.music_plan import expand_sections, inpaint_plan

    meta = current_take(ctx)
    if meta.backend != "elevenlabs" or not meta.song_id:
        raise S2SError(f"Take {meta.take} was not generated with ElevenLabs (stored for "
                       "inpainting); regenerate needs `--music-backend elevenlabs`.")  # fmt: skip
    arrangement = _arrangement(ctx)
    if arrangement is None:
        raise S2SError("No arrangement; run `speech2song arrange` first.")
    preset = load_preset(ctx.run.manifest.preset, ctx.config.presets_dir)
    backend = ElevenLabsBackend(ctx.config, CostLog(ctx.run.costs_path), ctx.run.id)
    if take_grid(meta, ctx.run.root) != grid_ms(arrangement):
        raise S2SError(f"The arrangement's timing changed since take {meta.take} was "
                       "generated, so its sections no longer line up. Run `speech2song "
                       "generate` first.")  # fmt: skip
    try:
        section_ids = expand_sections(arrangement, specs)
        adherence = adherence or ctx.config.elevenlabs.context_adherence
        plan, span = inpaint_plan(arrangement, preset, meta.song_id, section_ids, note,
                                  context_adherence=adherence)  # fmt: skip
    except ValueError as exc:
        raise S2SError(str(exc)) from exc
    label = section_ids[0] if len(section_ids) == 1 else f"{section_ids[0]}-{section_ids[-1]}"
    minutes = plan_minutes(plan)
    estimate = SpendEstimate(
        service="elevenlabs", model=meta.model or ctx.config.music_model,
        units={"minutes": round(minutes, 3)},
        usd=unit_usd(ctx.config.pricing, "music", minutes),
        description=f"regenerate {label} of take {meta.take} "
                    f"({(span[1] - span[0]) / 1000:.1f} s new; priced as the whole song)",
    )  # fmt: skip
    ctx.say(f"Regenerating {label} ({span[0] / 1000:.1f}-{span[1] / 1000:.1f} s) of take "
            f"{meta.take} (context adherence {adherence}); the rest of the take is kept as "
            "it is.")  # fmt: skip
    if not confirm_spend([estimate], console=ctx.console, yes=ctx.yes, dry_run=ctx.dry_run):
        return None
    return backend.inpaint(plan, meta, ctx.run.path(MUSIC_DIR), ctx.run.root,
                           section=label, note=note, span_ms=span, say=ctx.say)  # fmt: skip


def undo_regeneration(ctx: Context) -> TakeMeta:
    """Step the current take back to its previous version (free). The undone version's
    audio stays on disk and is listed under `undone` in the take's meta."""
    meta = current_take(ctx)
    history = list(meta.params.get("history") or [])
    if not history:
        raise S2SError(f"Take {meta.take} has no earlier version to go back to.")
    previous = history.pop()
    edits = list(meta.params.get("edits") or [])
    last_edit = edits.pop() if edits else {}
    undone = [*meta.params.get("undone", []),
              {"file": meta.file, "song_id": meta.song_id, **last_edit}]  # fmt: skip
    info = probe(ctx.run.path(previous["file"]))
    updated = meta.model_copy(update={
        "file": previous["file"], "song_id": previous["song_id"],
        "seconds": round(info.duration_s or meta.seconds, 3),
        "params": {**meta.params, "history": history, "edits": edits, "undone": undone},
    })  # fmt: skip
    write_json(ctx.run.path(take_meta_path(meta.take)), updated)
    ctx.say(f"Take {meta.take} is back to {previous['file']} (undid {meta.file}).")
    return updated
