# speech2song

Personal CLI tool: spoken-word recording in, finished electronic song out. Claude picks the
best lines, the speech's pitch becomes a melody, ElevenLabs Music generates the backing track,
and everything is mixed locally. The speech itself is never regenerated: clips are cut
sample-exactly from the source.

`docs/SPEC.md` is the source of truth; accepted changes are logged in `docs/DECISIONS.md`.

**Status:** M1 — ingest, voice isolation, transcription and transcript alignment work.
Clip selection (M2) and later steps are stubs.

## Install

Needs Python 3.12+, [uv](https://docs.astral.sh/uv/), and `ffmpeg`/`ffprobe` on `PATH`.

```bash
uv sync                    # app + dev tools (pytest, ruff)
uv sync --extra isolate    # also demucs voice isolation (CPU PyTorch, ~1 GB)
```

Plain `uv sync` removes the `isolate` extra if it was installed, so keep passing
`--extra isolate` once you use it. With pip: `pip install -e .[dev]`, and for the extra,
`pip install -e .[isolate] --extra-index-url https://download.pytorch.org/whl/cpu`.

Models download on first use into the Hugging Face cache (`~/.cache/huggingface`):
Whisper `large-v3-turbo` (~1.6 GB), demucs `htdemucs` (~80 MB).

## Usage

```bash
speech2song ingest inputs/talk.mp3 --transcript inputs/talk.txt   # new run
speech2song transcribe              # Whisper + alignment (cached afterwards)
speech2song status                  # what is done, stale or pending
speech2song transcribe --no-transcript   # re-align without the transcript; ASR stays cached
speech2song run inputs/talk.mp3 --transcript inputs/talk.txt      # every implemented step
```

- Each run lives in `runs/<run_id>/` (`manifest.json`, `costs.json`, `log.txt`, numbered
  artifacts). Commands act on the latest run unless given `--run ID` (or a unique prefix).
- Stages are skipped when their inputs and settings are unchanged; `--force` re-runs them.
- `--isolate-voice` (on `ingest`/`run`) writes `01_clean.wav` from demucs-isolated vocals;
  later stages analyse and cut clips from it instead of the original. It takes roughly 1.5x
  the talk's length on CPU.
- Flags like `--transcript`, `--whisper-model`, `--language` and `--isolate-voice` stick to
  the run, so later commands don't silently re-run expensive stages.
- Settings: copy `config.example.yaml` to `config.yaml` (gitignored). API keys go in `.env`
  (see `.env.example`), never in config files.

### Official transcripts

Plain UTF-8 text, one paragraph per line (text copied from a PDF with hard line wraps is
detected and unwrapped). Lines that are not spoken, such as image captions and headings, are
detected during alignment and left out; spoken words missing from the transcript are kept
from the ASR. `transcribe` prints both lists so you can check them.

## Spending

No paid API is called yet. From M2 on, every paid step estimates its cost first, supports
`--dry-run`, asks before spending unless `--yes` is given, and logs to `costs.json`
(`speech2song costs`). Prices are not built in; set them under `pricing:` in `config.yaml`.

## Development

```bash
uv run pytest                     # unit tests: synthetic fixtures only, no network
uv run pytest -m integration      # real Whisper/demucs on synthetic audio (downloads models)
uv run ruff check . && uv run ruff format .
```

Never commit secrets, user audio, or copyrighted material: `inputs/`, `runs/`, `outputs/`
and `.env` are gitignored, and test fixtures are generated or invented.
