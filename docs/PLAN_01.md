# Plan: Speech-to-Song: M0 (scaffold) + M1 (ingest, transcribe, align)

## Context

The repo has `CLAUDE.md`, `docs/SPEC.md`, an empty `docs/DECISIONS.md`, one preset, and a sample talk in `inputs/`: a 12.9-minute 48 kHz stereo MP3 plus the official `.txt` (and a PDF). There's no code yet and no git repo. This plan covers M0 and M1 only. Once you approve it, I'll implement both, run the tests, commit at the end of each milestone, and stop with a summary before M2.

What I found on this machine:
- Python 3.12.3, uv 0.12.21, ffmpeg 6.1.1 with libsoxr, fluidsynth, and GM soundfonts (FluidR3_GM, GeneralUser-GS), which M3 needs.
- An RTX 2060 is present, but its NVIDIA driver isn't loaded, so everything runs on the CPU: an i7-8750H (6 cores, 12 threads) with 15 GB RAM, about 7 GB of it free.

Your answers to my questions: **large-v3-turbo** is the default Whisper model, **demucs isolation gets implemented in M1**, and I **`git init` and commit at the end of each milestone** after tests and ruff pass.

---

## 1. My understanding

`speech2song` is a personal, local-first CLI that turns a recorded talk into an electronic song:
1. Claude picks a handful of strong, self-contained lines.
2. Those lines are cut from the source sample-exactly. Only edge fades are applied, and no generative model ever touches them.
3. The pitch contour of those lines becomes a key-snapped, looped melody (the speech-to-song illusion).
4. ElevenLabs Music generates a backing track arranged around the speech: sparse under it, swelling between ideas.
5. Everything is mixed locally with ducking and loudness targeting.

Style comes from swappable YAML presets. Each stage writes its artifacts to a run directory and is skipped when its content hashes haven't changed. Every paid call (Claude, ElevenLabs) is estimated up front, supports `--dry-run`, needs confirmation unless `--yes` is passed, and is logged to `costs.json`. M0/M1 involve **no paid calls at all**: transcription is local Whisper.

---

## 2. Spec issues, risks and proposed resolutions

When you approve this plan, I'll record each of these in `docs/DECISIONS.md`.

1. **Your `.env` isn't protected.** The ignore file is named `gitignore` (no dot), so git would ignore nothing, including `.env`, which holds your real keys.
   - I'll rename it to `.gitignore` and add `outputs/`.
   - Before the first commit I'll check with `git status --ignored` that `.env`, `inputs/`, `runs/` and `outputs/` are excluded.
2. **Official transcripts contain text that wasn't spoken.** The sample `.txt` has 12 image-caption and heading lines ("Vargas family", "Conclusion", …).
   - The spec only handles the opposite case: ASR words missing from the official text.
   - If caption lines got interpolated timestamps, they would become fake "sentences". M2 could then select them, and they would pass its quote-match check.
   - **Resolution:** official lines that barely match the audio, and runs of 3+ official tokens with no audio, are flagged `unspoken`. They're excluded from `words` and `sentences` and listed in the alignment report.
   - ASR-only runs (ad-libs, missing quotes) become `source: "asr"` sentences.
3. **Re-aligning shouldn't re-run Whisper.** On this CPU, turbo takes roughly 10–20 minutes per talk.
   - The CLI `ingest` command runs two cached manifest stages: `ingest` and `isolate`.
   - The CLI `transcribe` command runs two more: `asr` and `align`.
   - That adds one artifact, **`02_asr.json`** (the raw ASR output).
4. **"Bit-exact" conflicts with the speech processing in the mix.**
   - Goal 3 says bit-exact except fades, §2 also allows level changes, and §7 applies HPF, compression, EQ and reverb to speech.
   - **Resolution:**
     - `clips/*.wav` and the dry `07_mix/stems/speech.wav` are sample-exact slices of the analysis source multiplied by the fade envelope.
     - Gain, EQ, compression and effect sends happen only on the mix bus. That's deterministic DSP, never a generative model.
     - With `--isolate-voice`, "source" means `01_clean.wav`. demucs is a separation model, which §2 permits.
     - The manifest records which file the clips were cut from.
5. **Source format.** The source is 48 kHz, and the spec requires 44.1 kHz.
   - I'll resample once with soxr VHQ and write `00_source.wav` as **32-bit float**. Float keeps the MP3's inter-sample overs instead of clipping them.
   - The WAV switches to RF64 automatically if it exceeds 4 GB, and ffmpeg's bitexact flags make the output hash reproducible.
   - Sample-exactness is defined relative to `00_source.wav`.
   - Loudness and true peak are measured with ffmpeg's EBU R128 filter, which streams in constant memory. That means pyloudnorm and scipy aren't needed until M7.
6. **Clip targets have nowhere to live.** Stage 3 needs a clip count, min/max seconds, a speech budget, fade length and the boundary search window, and neither the preset nor the config has these.
   - I'll add an optional `clips:` block to the preset schema: count 5, 3–15 s per clip, 60 s speech budget, 15 ms fades, 150 ms boundary search.
   - `--clips N` overrides the count, and the existing YAML stays valid unchanged.
7. **Pinning conflicts with "Python 3.11+".** Current numpy (2.5) and PyAV (19) need Python 3.12 or newer.
   - I'll set `requires-python >=3.12` and use exact `==` pins for direct dependencies, and the repo will include `uv.lock`.
   - Torch for the demucs extra comes from the **PyTorch CPU index**. PyPI's Linux torch would pull about 3 GB of CUDA 13 libraries.
   - Dev tools go in both `[dependency-groups]`, for `uv sync`, and `[project.optional-dependencies].dev`, for `pip install -e .[dev]`.
8. **`--dry-run` scope.** A dry run executes nothing, not even the free stages.
   - It prints which stages would run and which are cached, plus paid estimates from cached artifacts or heuristics, then exits.
   - Paid confirmation needs a TTY or `--yes`, and it aborts otherwise.
9. **`costs.json` is append-only.** It's a JSON array whose entries are only ever appended. The file is rewritten atomically, right after each paid call.
10. **Small additions** that don't conflict with the spec:
    - a `speech2song status` command;
    - `--run` accepts an ID, a unique prefix or `latest` (the default, printed when used);
    - new modules `pipeline.py`, `costs.py`, `errors.py`, `text/`, `backends/transcribe_base.py` and `backends/isolate_*.py`.
    - **The Scribe backend moves to M5**, alongside the ElevenLabs docs pass.
11. **Later risks, noted now:**
    - **Whisper word timings** are off by about 0.1–0.3 s. M2 snaps cut points to energy minima, with a forced aligner as the fallback.
    - **Preset tempo:** with `bpm: 114` and `feel: half-time`, I treat 114 as the grid tempo, and M3/M5 analysis accepts 57 or 228 as equivalent.
    - **ElevenLabs:** the model IDs and limits in the spec stay unverified until the M5 docs pass.
    - **demucs on CPU** runs at about 1.5× the track length, so 20+ minutes per talk. It's opt-in and cached.
    - **huggingface-hub 2.0** was released last week. If it breaks faster-whisper or demucs, I'll constrain it to `<2`.

---

## 3. Implementation

### Files (no empty placeholder modules for M2+; those commands exist only as CLI stubs)
```
.gitignore (renamed) pyproject.toml uv.lock README.md config.example.yaml
docs/DECISIONS.md                     # entries from section 2
src/speech2song/
  __init__.py errors.py config.py models.py manifest.py costs.py pipeline.py cli.py
  audio/io.py                         # ffprobe/ffmpeg wrappers, audio info, EBU R128
  stages/ingest.py                    # IngestStage + IsolateStage
  stages/transcribe.py                # AsrStage
  stages/align.py                     # AlignStage (I/O only; logic in text/)
  backends/transcribe_base.py transcribe_whisper.py isolate_base.py isolate_demucs.py
  text/normalize.py sentences.py align.py   # pure functions
tests/conftest.py tests/fixtures/{synth.py, mini_talk.txt} tests/test_*.py
```

### Dependencies (approval to install)
- **Core:** typer, rich, pydantic, pyyaml, python-dotenv, numpy, soundfile, rapidfuzz, faster-whisper (pulls in ctranslate2, onnxruntime, av and tokenizers).
- **`isolate` extra:** `demucs==4.1.0` and CPU `torch`, about 1 GB installed.
- **Dev:** pytest, ruff.
- **Model downloads** go to the Hugging Face cache outside the repo: large-v3-turbo is about 1.6 GB, and htdemucs about 80 MB.
- **Not yet:** librosa, scipy, pedalboard, pretty_midi, anthropic and elevenlabs wait for the milestone that uses them.

### M0: scaffold

**`config.py`**
- `Preset` model (`extra="forbid"`, so typos fail loudly):
  - fields: tempo{bpm, feel, tolerance_bpm}, key{mode, tonic: "auto" or a pitch class, fallback}, positive_styles, negative_styles, section_roles{role: {energy 0–1, styles}}, arc, speech_interaction, mix, melody, plus the optional `clips` block;
  - validation: every `arc` entry must name a defined role, and `speech_bed` must exist.
- `AppConfig`, read from an optional `config.yaml`:
  - the flat keys from spec §8 (`claude_model`, `music_model`, `transcribe_backend`, `whisper_model=large-v3-turbo`), plus `runs_dir`, `presets_dir`, `default_preset` and `music_backend=stub`;
  - nested `whisper`, `demucs`, `align` and `pricing` blocks.
- `Secrets` loads from env or `.env` as `SecretStr`. It's never serialized, and the settings snapshot excludes it.

**`models.py`** (the M0 models)
- `FileRef{path, sha256, size}`
- `HashMemo{size, mtime_ns, sha256}`
- `StageRecord{status: running|complete|failed, version, fingerprint, inputs: {name: FileRef}, params, outputs: {relpath: FileRef}, started_at, finished_at, elapsed_s, summary, error}`
- `Manifest{schema_version, run_id, created_at, tool_version, input: FileRef, transcript: FileRef|None, preset, options{isolate_voice, clips, music_backend}, source_probe, source: AudioInfo|None, clean: AudioInfo|None, stages: {name: StageRecord}, hash_memo: {abs_path: HashMemo}}`
- `CostEntry{ts, run_id, stage, service, operation, model, units: {name: float}, usd, estimated, price_ref, request_id, note}`

**`manifest.py`**
- Run IDs look like `YYYYMMDD-HHMMSS-<slug ≤32>`, with a numeric suffix on collision.
- `Run.create(...)` and `Run.open(ref)`, where `ref` is an ID, a prefix or `latest`.
- JSON writes are atomic: write a temp file, fsync, then `os.replace`.
- `file_sha256` reuses `hash_memo` while a file's size and mtime are unchanged, so the 270 MB WAV isn't rehashed on every command.
- `fingerprint` = sha256 of the canonical JSON `{stage, version, input hashes, params}`.
- Each run gets its own `log.txt`.

**Caching (`pipeline.py`, `execute(stage, ctx)`)**
1. `stage.plan(ctx)` returns the input paths, JSON params (only the settings that affect the output, e.g. not the thread count) and the expected outputs.
2. **Skip** when all of these hold: no `--force`, the record is `complete`, its fingerprint matches, and every recorded output exists.
   - A stage's own outputs aren't re-hashed when deciding this. If you hand-edit an output it's kept, and downstream stages see the change through their input hashes.
3. **Paid stages** (M2+) print an estimate first. `--dry-run` stops there; otherwise they ask for confirmation unless `--yes`.
4. **Run:** the record is marked `running`, outputs are written to temp files and then renamed, the outputs are hashed, and the record becomes `complete`.
   - If the stage raises, the record becomes `failed` with the error, and the next invocation retries.
   - Because outputs are deterministic, an upstream re-run that produces identical bytes leaves downstream stages cached.

**`costs.py`**
- `CostLog` can append, read and total entries.
- `PriceTable` reads `pricing` from config and never hardcodes prices. Claude defaults arrive in M2, sourced from the Claude API reference, and ElevenLabs rates in M5 after the docs pass.
- `confirm_spend(estimate, yes, dry_run)` handles the prompt.

**`cli.py`** (typer)

Every command prints what it read, what it wrote, and the spend. Global options: `--config`, `--runs-dir`, `-v`.

| Command | M0/M1 behavior |
|---|---|
| `run INPUT [--transcript] [--preset] [--isolate-voice] [--clips] [--music-backend] [--dry-run] [--yes] [--stop-after] [--force] [--run]` | Creates a run (or resumes one with `--run`), runs ingest → isolate → asr → align, then reports "select: planned for M2" |
| `ingest [INPUT] [--run] [--transcript] [--preset] [--isolate-voice/--no-isolate-voice] [--force]` | Creates a new run from INPUT, or re-ingests an existing run |
| `transcribe [--run] [--transcript FILE \| --no-transcript] [--whisper-model] [--language] [--force]` | Runs asr + align |
| `select`, `melody`, `arrange`, `generate`, `mix`, `regenerate` | Stubs: print "planned for Mx", exit 2 |
| `costs [--run]` | Prints entries and totals from `costs.json` |
| `presets list` | Lists name, description and validation status |
| `status [--run]` | Shows each stage as complete, stale, missing or failed, plus total spend |

### M1: ingest, isolate, transcribe, align

**Ingest** (`audio/io.py`, `stages/ingest.py`)
- ffprobe's JSON becomes a `SourceProbe`. The first audio stream is used, and a file with no audio stream fails with a clear error.
- ffmpeg writes `00_source.wav`: 44.1 kHz float32 via soxr. Mono and stereo are kept; more than 2 channels are downmixed to stereo. Bitexact flags and automatic RF64 are on.
- `AudioInfo{path, sample_rate, channels, frames, duration_s, subtype, integrated_lufs, lra, sample_peak_dbfs, true_peak_dbtp}` is written to the manifest.
  - The R128 numbers come from parsing the summary of ffmpeg's `ebur128=peak=sample+true` filter.

**Isolate** (`backends/isolate_*.py`)
- **When off,** `01_clean.wav` is a relative symlink to `00_source.wav`, with a copy as the fallback.
- **When on,** it uses `demucs.api.Separator(model="htdemucs", device="cpu", shifts=0)` and keeps the vocals stem.
  - The audio is processed in 120 s windows with 5 s of context on each side and 0.5 s crossfades, and streamed to disk. That keeps memory bounded for hour-long lectures.
  - Mono sources are duplicated to stereo for the model and averaged back afterwards.
  - The output's frame count, channel count and sample rate are asserted to match `00_source.wav`.
  - The window and crossfade math are pure functions, separate from the model call.
- I'll read the installed demucs source to confirm the API before coding against it.

**ASR** (`backends/transcribe_whisper.py`, `stages/transcribe.py`)
- Settings: faster-whisper 1.2.1, `large-v3-turbo`, device auto (CPU here), `int8`, `cpu_threads` = the number of physical cores, beam 5, `vad_filter`, `word_timestamps`. Language is auto-detected unless `--language` is given.
- Batched inference (`batch_size`) is optional. I'll pick its default after timing a 2-minute excerpt both ways.
- The import is lazy, and progress shows as a rich bar.
- A pure `segments_to_asr()` cleans the words: strips whitespace, drops empty words, clamps `end >= start`, and keeps times monotonic.
- I'll check the parameter names against the installed source.
- Output **`02_asr.json`**: `AsrResult{backend, model, params, language, language_probability, duration_s, audio, segments[{id, start, end, text, avg_logprob, no_speech_prob, words[{w, start, end, conf}]}]}`

**Align** (pure logic in `text/`, I/O in `stages/align.py`) writes **`02_transcript.json`**.
- **Normalize:**
  - Apply NFKC, turn curly quotes and dashes into plain ones, and fold accents.
  - Drop bracket characters but keep the words inside them, and drop footnote digits glued to punctuation.
  - Split on hyphens, dashes and slashes, lowercase everything, and remove apostrophes.
  - Every token remembers which source word it came from.
- **Sentences:**
  - Split on `.?!` (including when followed by a closing quote), but not after initials ("Russell M. Nelson") or abbreviations.
  - Line breaks are hard boundaries, unless the text looks hard-wrapped (e.g. pasted from a PDF).
- **Alignment:**
  1. A global LCS on token IDs (rapidfuzz `Indel.opcodes`) finds exact anchors.
  2. Each gap between anchors goes through a local dynamic-programming pass that allows 1:1, 1:2 and 2:1 pairings and skips. Pairs scoring rapidfuzz ratio ≥ 0.6 are marked `fuzzy`. This handles names, words split or merged by ASR, and `eight`/`8`.
  3. Official lines with less than 25% of their tokens matched are marked unspoken, and the alignment runs again without them.
  4. What remains in each gap is classified:
     - **Official tokens with no audio:** 3+ tokens become `unspoken`; fewer become `interpolated`.
     - **ASR tokens with no official text:** these become `asr_only`.
     - **Both sides present:** a substitution. The official words take the ASR time span, so "1984" covers "nineteen eighty-four"; they're flagged `fuzzy` and keep the original ASR text.
- **Words** come out in time order with the official spelling and ASR timing, and a `flag` of matched, fuzzy, interpolated, asr_only or asr.
  - An ASR word is emitted as `asr_only` only if none of its tokens were paired.
- **Sentences** in the output:
  - Official sentences are mapped onto the emitted word ranges, and any sentence that is entirely unspoken is dropped.
  - ASR-only runs of 4+ words become their own `asr` sentences; shorter runs attach to the neighboring sentence.
  - Each sentence's text is the join of its words, IDs run 1..N, and each sentence carries `avg_conf`.
- **Without an official transcript,** the ASR words are used directly. Sentences split on punctuation (with the same initials guard), on pauses of 1.5 s or more, and sentences over 30 s split at their longest gap.
- **Report:** `AlignmentReport{official_words, matched, fuzzy, interpolated, unspoken, asr_only, quality (spec: matched / official), quality_spoken, coverage (fraction of ASR words paired), unspoken_spans, asr_only_spans}`.
  - If coverage is below 0.2, it warns "wrong transcript?" and builds the ASR-only transcript with `official_transcript_used: false`.
- **Transcript model:** `Transcript{schema_version, source, duration_s, language, words[Word{w, start, end, conf, flag, asr?}], sentences[Sentence{id, text, start, end, word_start, word_end, source, avg_conf}], official_transcript_used, alignment, asr{backend, model}}`. The spec's fields are unchanged; the extras are additions.

### Testing (synthetic fixtures only)

**Test safety (`conftest.py`):**
- An autouse fixture blocks socket connections.
- API key variables are removed from the environment.
- Tests run with the working directory set to `tmp_path`, so the real `.env` is never read.

**Fixtures** (`tests/fixtures/synth.py`, generated at test time and seeded):
- Sine waves, sweeps and noise bursts.
- "Speech-like" harmonic tones with syllable envelopes, glides and gaps, returned together with their ground-truth word times.
- Mono, stereo and 5.1 WAVs at 44.1 and 48 kHz, plus a tiny MP4 generated with ffmpeg lavfi.
- `fake_asr(text, subs, drops, inserts)`, which produces timed ASR words with controlled errors.
- `mini_talk.txt`: invented text (no copyrighted material) containing captions, a heading, verse lines, curly quotes, brackets, initials, numerals and em-dashes.

**M0 tests**
- **Presets:** the real preset loads; presets with an unknown key, a bad arc role, energy > 1 or a bad tonic are rejected.
- **Config:** config and env precedence work; secrets never appear in the manifest or logs (checked by grepping for a sentinel key).
- **Runs:** run IDs, slugs and collisions; resolving runs by prefix and by `latest`.
- **Manifest:** an atomic write survives a simulated crash, the hash memo is reused, and fingerprints don't depend on dict order.
- **Executor:**
  - a stage runs once, then is cached;
  - it re-runs when an input, a param or the stage version changes, when an output is missing, or with `--force`;
  - a failure is recorded and retried;
  - downstream stages re-run only when upstream bytes change;
  - a hand-edited output is kept.
- **Costs:** appending, totals, price math, and the confirm cases (dry-run, `--yes`, no TTY aborts).
- **CLI:** `--help`, `presets list`, `costs`, `status` and the stubs.

**M1 tests**
- **Ingest:**
  - Parsing canned ffprobe JSON, and building the ffmpeg arguments.
  - Real ffmpeg on tiny files:
    - 48 kHz → 44.1 kHz keeps the duration;
    - mono and stereo stay as they are, and 5.1 becomes stereo;
    - audio is extracted from an MP4;
    - a file without audio gives a clear error;
    - a 44.1 kHz float input passes through bit-identically.
  - A 997 Hz sine at −20 dBFS reads about −23.0 LUFS (±0.3).
- **Isolate:**
  - The symlink is created when isolation is off.
  - The window plan covers every sample exactly once.
  - An identity fake separator reproduces the input (to within 1e-7), and a gain fake applies exactly that gain.
  - Mono round-trips, and the output length matches the input.
- **ASR:**
  - The adapter is tested with fake segment objects.
  - Through the CLI with a fake transcriber, the asr and align stages cache and invalidate correctly.
- **Text:** a normalization table, and edge cases for the sentence splitter.
- **Aligner:**
  - Scenarios: identical texts, misspelled names, split and merged words, numerals, captions becoming unspoken, an ASR-only quote becoming its own sentence, a dropped word becoming interpolated, and a wrong transcript triggering the fallback.
  - Invariants: words sorted by time, `start <= end`, sentences contiguous, and sentence text equal to its joined words.
- **Opt-in `pytest -m integration`** (deselected by default): real faster-whisper on a synthetic tone to check the plumbing, and real demucs on 6 s of synthetic audio to check output shape and finite values.

---

## 4. Defaults I'm assuming (tell me if any are wrong)
- Python 3.12 floor.
- `--dry-run` executes nothing.
- `--run` defaults to `latest`.
- `clips` block defaults as in section 2.
- `00_source.wav` is float32.
- Scribe moves to M5.
- `outputs/` is gitignored and unused until M4.
- If `git config user.name`/`user.email` are unset, I'll ask you rather than set them globally.

---

## Verification
1. `uv sync --extra isolate`, then `ruff check . && ruff format --check .`, then `pytest` (fast and offline).
2. After the models download, `pytest -m integration`.
3. **Acceptance on the real talk** (local only; outputs go to the gitignored `runs/`):
   ```
   speech2song ingest inputs/come-home-by-elder-clark-g-gilbert.mp3 --transcript inputs/come-home-by-elder-clark-g-gilbert.txt
   speech2song transcribe                    # turbo; time a 2-min excerpt first to pick batch_size
   speech2song status
   speech2song transcribe --no-transcript    # ASR cached, only align re-runs; compare outputs
   speech2song ingest inputs/come-home-by-elder-clark-g-gilbert.mp3 --isolate-voice   # new run, in background
   ```
   - The 12 caption and heading lines should come out as unspoken, and `quality_spoken` should be ≥ 0.95.
   - I'll spot-check about 5 sentences by slicing them into WAVs in the scratchpad, so you can listen.
   - I'll report the runtimes, alignment numbers and isolation result.
4. Commit M0, then commit M1, each after green tests and ruff. Finish with a summary: what was built, what was tested, and what's uncertain.
