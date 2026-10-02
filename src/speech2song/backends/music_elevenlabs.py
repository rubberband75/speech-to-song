"""The Eleven Music backend (official `elevenlabs` SDK, docs read 2026-10-01).

Each take is one `music.compose_detailed` call with the arrangement's composition plan,
`model_id` music_v2/v2.5 and `store_for_inpainting=True`, so the song can be edited
later by section. Calls are never retried automatically (a timeout may still have been
billed). Every call is logged to costs.json right after it returns. A take whose request
is unchanged is reused instead of generated again. Takes made for an older request stay
selectable while their timing still fits the arrangement, and otherwise move to
06_music/archive/; nothing paid is ever overwritten.
"""

import json
import logging
import shutil
import tempfile
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from speech2song.audio.io import probe, require_tool, run_tool
from speech2song.audio.synth import write_wav
from speech2song.backends.music_base import (
    MelodyFiles,
    Take,
    next_take_number,
    remove_stub_takes,
    take_fits,
    take_grid,
    take_metas,
)
from speech2song.config import AppConfig, Preset, load_secrets
from speech2song.costs import CostLog, SpendEstimate, unit_usd
from speech2song.errors import S2SError, StageError
from speech2song.llm.claude import request_digest
from speech2song.manifest import atomic_write_text, sha256_file, write_json
from speech2song.models import Arrangement, CostEntry, TakeMeta
from speech2song.music_plan import (
    MAX_REFERENCE_MS,
    MELODY_SONG,
    build_plan,
    check_inpaint_plan,
    check_plan,
    grid_ms,
    plan_minutes,
    plan_references,
)

log = logging.getLogger(__name__)

PRICE_ITEM = "music"
REFERENCE_FILE = "reference.json"  # the uploaded melody reference's song_id, per audio hash
UPLOADS_FILE = "uploads.json"  # uploaded quote melodies: song_id per upload digest
UPLOAD_GAP_MS = 500  # silence between the quotes in an upload
ARCHIVE_DIR = "archive"


def melody_upload(melody: MelodyFiles, clips: list[str]) -> dict[str, Any] | None:
    """What to upload so the plan can be conditioned on these clips' melodies: their
    ranges in the quote melodies file (`source`) and where each lands in the upload
    (`ranges`, back to back with a short silence between). Pure but for hashing the file."""
    if melody.quotes is None or melody.index is None:
        return None
    clips = [clip for clip in clips if clip in melody.index.clips]
    if not clips:
        return None
    source, ranges, cursor = {}, {}, 0
    for clip in clips:
        start, end = melody.index.clips[clip]
        length = min(end - start, MAX_REFERENCE_MS)
        source[clip] = [start, start + length]
        ranges[clip] = [cursor, cursor + length]
        cursor += length + UPLOAD_GAP_MS
    return {"path": str(melody.quotes), "sha256": sha256_file(melody.quotes),
            "source": source, "ranges": ranges, "ms": cursor - UPLOAD_GAP_MS}  # fmt: skip


def _default_client(api_key: str) -> Any:
    from elevenlabs.client import ElevenLabs

    return ElevenLabs(api_key=api_key)


# Tests replace this to inject a fake client; nothing in tests may reach the network.
client_factory: Callable[[str], Any] = _default_client


def audio_extension(output_format: str) -> str:
    """File extension for an output format (config allows auto, mp3_* and opus_*)."""
    return ".opus" if output_format.startswith("opus_") else ".mp3"


def API_ERRORS() -> tuple[type[Exception], ...]:  # noqa: N802 - reads like a constant
    """What the SDK raises for API and network failures (anything else is a bug here)."""
    import httpx
    from elevenlabs.core.api_error import ApiError

    return (ApiError, httpx.HTTPError)


def _error_detail(exc: Exception) -> tuple[int | None, dict]:
    status = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    detail = body.get("detail") if isinstance(body, dict) else None
    return status, detail if isinstance(detail, dict) else {"message": str(detail or body)}


class ElevenLabsBackend:
    name = "elevenlabs"
    paid = True

    def __init__(self, config: AppConfig, cost_log: CostLog | None = None,
                 run_id: str = "", stage: str = "generate") -> None:  # fmt: skip
        self.config = config
        self.cost_log = cost_log
        self.run_id = run_id
        self.stage = stage
        self._client: Any = None

    # --- request and estimate (pure, no network) ---------------------------------------

    def request(
        self, arrangement: Arrangement, preset: Preset, melody: MelodyFiles
    ) -> dict[str, Any]:
        settings = self.config.elevenlabs
        reference = upload = None
        if arrangement.plan_version < 2:  # M5: the main phrase conditions the first chunk
            if settings.melody_reference and melody.reference is not None:
                reference = {"path": str(melody.reference),
                             "sha256": sha256_file(melody.reference)}  # fmt: skip
        elif settings.melody_conditioning:  # M7: each quote conditions the music around it
            upload = melody_upload(melody, plan_references(arrangement))
        plan, layout = build_plan(
            arrangement, preset, context_adherence=settings.context_adherence,
            reference_song_id="<uploaded melody reference>" if reference else None,
            condition_strength=settings.condition_strength,
            melody_ranges={k: tuple(v) for k, v in upload["ranges"].items()} if upload else None,
        )  # fmt: skip
        request = {
            "backend": self.name,
            "model_id": self.config.music_model,
            "output_format": settings.output_format,
            "composition_plan": plan,
            "grid_ms": grid_ms(arrangement),
            "layout": [vars(chunk) for chunk in layout],
            "reference": reference,
        }
        if upload:  # only M7 requests carry the key, so M5 requests keep their digest
            request["melody_upload"] = upload
        return request

    def estimate(self, request: dict[str, Any], takes: int) -> list[SpendEstimate]:
        minutes = plan_minutes(request["composition_plan"])
        model = request["model_id"]
        pricing = self.config.pricing
        estimates = []
        if request.get("reference"):
            ref_minutes = MAX_REFERENCE_MS / 60_000
            estimates.append(SpendEstimate(
                service="elevenlabs", model="music.upload", units={"minutes": ref_minutes},
                usd=unit_usd(pricing, PRICE_ITEM, ref_minutes),
                description="upload the melody reference (once; billed like a generation)",
            ))  # fmt: skip
        upload = request.get("melody_upload")
        if upload:
            up_minutes = upload["ms"] / 60_000
            estimates.append(SpendEstimate(
                service="elevenlabs", model="music.upload",
                units={"minutes": round(up_minutes, 3)},
                usd=unit_usd(pricing, PRICE_ITEM, up_minutes),
                description=f"upload {len(upload['ranges'])} quote melodies, "
                            f"{upload['ms'] / 1000:.0f} s (once; billed like a generation)",
            ))  # fmt: skip
        for n in range(1, takes + 1):
            estimates.append(SpendEstimate(
                service="elevenlabs", model=model, units={"minutes": round(minutes, 3)},
                usd=unit_usd(pricing, PRICE_ITEM, minutes),
                description=f"new take {n} of {takes}: {minutes * 60:.0f} s of music",
            ))  # fmt: skip
        return estimates

    # --- paid calls -------------------------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is None:
            key = load_secrets().elevenlabs_api_key
            if key is None:
                raise S2SError("No ElevenLabs key: set ELEVENLABS_API_KEY in .env.")
            self._client = client_factory(key.get_secret_value())
        return self._client

    def _options(self) -> dict[str, Any]:
        return {"timeout_in_seconds": self.config.elevenlabs.timeout_s, "max_retries": 0}

    def _log_cost(self, operation: str, model: str, minutes: float, ref: str | None,
                  note: str | None = None) -> float | None:  # fmt: skip
        price = self.config.pricing.elevenlabs.get(PRICE_ITEM)
        usd = unit_usd(self.config.pricing, PRICE_ITEM, minutes)
        units: dict[str, float] = {"minutes": round(minutes, 4)}
        if self.cost_log is not None:
            self.cost_log.append(
                CostEntry(
                    ts=datetime.now().astimezone(), run_id=self.run_id, stage=self.stage,
                    service="elevenlabs", operation=operation, model=model, units=units,
                    usd=round(usd or 0.0, 6), estimated=True, request_id=ref, note=note,
                    price_ref=(f"pricing.elevenlabs.{PRICE_ITEM} (as of {price.as_of})"
                               if price else None),
                )
            )  # fmt: skip
        return usd

    def _api_error(self, exc: Exception, out_dir: Path) -> S2SError:
        status, detail = _error_detail(exc)
        kind = detail.get("status") or detail.get("code") or ""
        message = detail.get("message") or str(detail)
        if kind in ("bad_composition_plan", "bad_prompt"):
            suggestion = (detail.get("data") or {}).get("composition_plan_suggestion") or (
                detail.get("data") or {}).get("prompt_suggestion")  # fmt: skip
            path = out_dir / "plan_suggestion.json"
            write_json(path, {"status": kind, "message": message, "suggestion": suggestion})
            return S2SError(f"ElevenLabs rejected the plan ({kind}: {message}). Its suggested "
                            f"alternative is in {path}; no artist or song names may appear "
                            "in styles.")  # fmt: skip
        if status is None:  # network or timeout: the call may still have been billed
            return S2SError(f"ElevenLabs could not be reached or timed out ({exc}). If a "
                            "generation was cut off it may still be billed; check your "
                            "ElevenLabs history before trying again.")  # fmt: skip
        if status == 401:
            return S2SError("ElevenLabs rejected the API key (check ELEVENLABS_API_KEY in .env).")
        if status in (402, 429) or "quota" in kind or "limit" in kind:
            return S2SError(f"ElevenLabs refused for quota or rate reasons ({status} {kind}): "
                            f"{message}")  # fmt: skip
        return S2SError(f"ElevenLabs error {status} {kind}: {message}")

    def _reference_song(self, request: dict[str, Any], out_dir: Path) -> str | None:
        """Upload the melody reference once per audio hash (paid); return its song_id."""
        reference = request.get("reference")
        if not reference:
            return None
        cache = out_dir / REFERENCE_FILE
        if cache.exists():
            saved = json.loads(cache.read_text())
            if saved.get("sha256") == reference["sha256"]:
                return saved["song_id"]
        clip = out_dir / ".reference_upload.mp3"
        seconds = MAX_REFERENCE_MS / 1000
        run_tool([require_tool("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error",
                  "-y", "-i", reference["path"], "-t", f"{seconds:g}", "-c:a", "libmp3lame",
                  "-b:a", "192k", str(clip)])  # fmt: skip
        try:
            with clip.open("rb") as fh:
                response = self._get_client().music.upload(file=fh, request_options=self._options())
        except API_ERRORS() as exc:
            raise self._api_error(exc, out_dir) from exc
        finally:
            clip.unlink(missing_ok=True)
        self._log_cost("music.upload", "music.upload", seconds / 60, response.song_id,
                       "melody reference for conditioning")  # fmt: skip
        write_json(cache, {"sha256": reference["sha256"], "song_id": response.song_id})
        return response.song_id

    def _melody_song(self, request: dict[str, Any], out_dir: Path) -> str | None:
        """Upload the referenced quote melodies once (paid); return the song_id. The
        upload is cached by its content (file hash and ranges), so later takes, other
        requests and copies of the run reuse it."""
        upload = request.get("melody_upload")
        if not upload:
            return None
        digest = request_digest({k: upload[k] for k in ("sha256", "source")})
        cache = out_dir / UPLOADS_FILE
        saved = json.loads(cache.read_text()) if cache.exists() else {}
        if digest in saved:
            return saved[digest]["song_id"]
        audio, rate = sf.read(upload["path"], dtype="float32", always_2d=True)
        gap = np.zeros((round(UPLOAD_GAP_MS / 1000 * rate), audio.shape[1]), dtype=np.float32)
        pieces = []
        for start, end in upload["source"].values():
            if pieces:
                pieces.append(gap)
            pieces.append(audio[round(start / 1000 * rate) : round(end / 1000 * rate)])
        with tempfile.TemporaryDirectory() as tmpdir:
            wav, mp3 = Path(tmpdir) / "quotes.wav", Path(tmpdir) / "quotes.mp3"
            write_wav(wav, np.concatenate(pieces), rate)
            run_tool([require_tool("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error",
                      "-y", "-i", str(wav), "-c:a", "libmp3lame", "-b:a", "192k",
                      str(mp3)])  # fmt: skip
            try:
                with mp3.open("rb") as fh:
                    response = self._get_client().music.upload(
                        file=fh, request_options=self._options()
                    )
            except API_ERRORS() as exc:
                raise self._api_error(exc, out_dir) from exc
        self._log_cost(
            "music.upload",
            "music.upload",
            upload["ms"] / 60_000,
            response.song_id,
            f"quote melodies for conditioning: {', '.join(upload['ranges'])}",
        )
        saved[digest] = {"song_id": response.song_id, "clips": list(upload["ranges"])}
        write_json(cache, saved)
        return response.song_id

    def _archive_misfits(self, out_dir: Path, request: dict[str, Any], run_dir: Path) -> None:
        """Move paid takes whose timing no longer fits the arrangement out of the way."""
        digest = request_digest(request)
        for meta in take_metas(out_dir):
            if meta.backend == "stub" or take_fits(meta, run_dir, request["grid_ms"], digest):
                continue
            target = out_dir / ARCHIVE_DIR / meta.request_sha256[:12]
            target.mkdir(parents=True, exist_ok=True)
            audio = [run_dir / v["file"] for v in meta.params.get("history", [])]
            audio.append(run_dir / meta.file)
            files = [*audio, *(path.with_suffix(".response.json") for path in audio),
                     out_dir / f"take_{meta.take:03d}.meta.json"]  # fmt: skip
            for path in files:
                if path.exists():
                    shutil.move(str(path), str(target / path.name))
            log.info("archived take %d to %s", meta.take, target)

    def _compose(self, plan: dict, model: str, output_format: str, out_dir: Path,
                 path: Path, label: str, stage: str) -> tuple[Any, float | None]:  # fmt: skip
        """One paid compose_detailed call: audio to `path`, the JSON beside it, cost logged."""
        kwargs: dict[str, Any] = {}
        if output_format != "auto":
            kwargs["output_format"] = output_format
        try:
            response = self._get_client().music.compose_detailed(
                composition_plan=plan, model_id=model, store_for_inpainting=True,
                request_options=self._options(), **kwargs,
            )  # fmt: skip
        except API_ERRORS() as exc:
            raise self._api_error(exc, out_dir) from exc
        self.stage = stage
        usd = self._log_cost("music.compose_detailed", model, plan_minutes(plan),
                             response.song_id, label)  # fmt: skip
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_bytes(response.audio)
        tmp.replace(path)
        atomic_write_text(path.with_suffix(".response.json"),
                          json.dumps(response.json, indent=2) + "\n")  # fmt: skip
        return response, usd

    def generate(
        self, request: dict[str, Any], out_dir: Path, takes: int, run_dir: Path,
        say: Callable[[str], None] = print, fresh: bool = False,
    ) -> list[Take]:  # fmt: skip
        """Make sure `takes` takes exist for this request: reuse the newest ones made for
        it (unless `fresh`) and generate the rest under new take numbers."""
        problems = check_plan(request["composition_plan"])
        if problems:
            raise StageError("the composition plan breaks the API limits: " + "; ".join(problems))
        out_dir.mkdir(parents=True, exist_ok=True)
        digest = request_digest(request)
        remove_stub_takes(out_dir, run_dir)
        self._archive_misfits(out_dir, request, run_dir)
        matching = [m for m in take_metas(out_dir)
                    if m.request_sha256 == digest and (run_dir / m.file).exists()]  # fmt: skip
        reused = [] if fresh else matching[-takes:]
        results = []
        for meta in reused:
            say(f"  take {meta.take}: reusing the paid take for this request ({meta.file})")
            results.append(Take(run_dir / meta.file, meta))
        missing = takes - len(reused)
        if not missing:
            return results
        plan = request["composition_plan"]
        song_id = self._reference_song(request, out_dir)
        if song_id:
            plan = json.loads(json.dumps(plan).replace("<uploaded melody reference>", song_id))
        melody_song = self._melody_song(request, out_dir)
        if melody_song:
            plan = json.loads(json.dumps(plan).replace(MELODY_SONG, melody_song))
            song_id = melody_song
        minutes = plan_minutes(plan)
        model = request["model_id"]
        output_format = request["output_format"]
        first = next_take_number(out_dir)
        for number in range(first, first + missing):
            say(f"  take {number}: composing {minutes * 60:.0f} s with {model} "
                "(this can take a few minutes)")  # fmt: skip
            path = out_dir / f"take_{number:03d}{audio_extension(output_format)}"
            response, usd = self._compose(plan, model, output_format, out_dir, path,
                                          f"take {number}", "generate")  # fmt: skip
            info = probe(path)
            meta = TakeMeta(
                take=number, backend=self.name, model=model,
                file=path.relative_to(run_dir).as_posix(), sample_rate=info.sample_rate,
                channels=info.channels, seconds=round(info.duration_s or minutes * 60, 3),
                usd=round(usd or 0.0, 6), song_id=response.song_id, request_sha256=digest,
                grid_ms=request["grid_ms"],
                params={"output_format": output_format, "reference_song_id": song_id,
                        "filename": getattr(response, "filename", None)},
            )  # fmt: skip
            write_json(out_dir / f"take_{number:03d}.meta.json", meta)
            say(f"  take {number}: song {response.song_id}, ${usd or 0:.2f} (est.)")
            results.append(Take(path, meta))
        return results

    def inpaint(
        self, plan: dict, meta: TakeMeta, out_dir: Path, run_dir: Path, *, section: str,
        note: str | None, span_ms: tuple[int, int], say: Callable[[str], None] = print,
    ) -> TakeMeta:  # fmt: skip
        """Regenerate part of a stored take (`plan` keeps the rest as audio references).
        The result becomes the take's new version; earlier versions stay on disk."""
        problems = check_inpaint_plan(plan)
        if problems:
            raise StageError("the inpainting plan breaks the API limits: " + "; ".join(problems))
        output_format = meta.params.get("output_format", self.config.elevenlabs.output_format)
        version = len(meta.params.get("history", [])) + 2
        extension = audio_extension(output_format)
        while (out_dir / f"take_{meta.take:03d}_v{version}{extension}").exists():
            version += 1  # an undone version keeps its file
        path = out_dir / f"take_{meta.take:03d}_v{version}{extension}"
        say(f"  take {meta.take}: regenerating {section} "
            f"({span_ms[0] / 1000:.1f}-{span_ms[1] / 1000:.1f} s), keeping the rest")  # fmt: skip
        response, usd = self._compose(plan, meta.model or self.config.music_model,
                                      output_format, out_dir, path,
                                      f"take {meta.take}: regenerate {section}",
                                      "regenerate")  # fmt: skip
        info = probe(path)
        history = [*meta.params.get("history", []),
                   {"file": meta.file, "song_id": meta.song_id}]  # fmt: skip
        edits = [
            *meta.params.get("edits", []),
            {
                "version": version,
                "section": section,
                "note": note,
                "start_ms": span_ms[0],
                "end_ms": span_ms[1],
                "song_id": response.song_id,
            },
        ]
        updated = meta.model_copy(update={
            "grid_ms": take_grid(meta, run_dir),
            "file": path.relative_to(run_dir).as_posix(), "song_id": response.song_id,
            "seconds": round(info.duration_s or meta.seconds, 3),
            "usd": round(meta.usd + (usd or 0.0), 6),
            "params": {**meta.params, "history": history, "edits": edits},
        })  # fmt: skip
        write_json(out_dir / f"take_{meta.take:03d}.meta.json", updated)
        say(f"  take {meta.take} v{version}: song {response.song_id}, ${usd or 0:.2f} (est.)")
        return updated
