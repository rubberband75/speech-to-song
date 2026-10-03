# Speech-to-Song: Project Specification

A command-line tool that takes a spoken-word recording (a lecture or conference talk), uses AI to choose the best lines, extracts a melody from the speech, generates a backing track around those lines, and mixes everything into a finished song. Reference inspiration: electronic tracks that weave philosophical lecture excerpts into cinematic future-bass music.

This document is the source of truth. If implementation reveals a problem with the spec, stop and propose a change rather than silently diverging.

---

## 1. Goals and non-goals

### Goals
1. Input: one audio or video file, plus an optional official transcript (plain text).
2. Claude selects the strongest self-contained lines and proposes an ordering.
3. Selected clips are kept **bit-exact apart from short edge fades**. They are cut from the source and mixed locally. They never pass through a generative model.
4. Extract a melody from the cadence and pitch of the spoken clips (the speech-to-song illusion), snap it to a key, and use it to inform the backing track.
5. Generate the backing track with ElevenLabs Music, arranged so the music recedes during speech and swells between ideas.
6. Mix locally with sidechain ducking, effects and loudness normalization.
7. Style is driven by swappable preset files (first preset: cinematic future bass; later: soft piano, lofi).
8. Every stage writes its outputs to disk, so any stage can be rerun alone.
9. Paid API calls are visible, estimated before they run, and logged.

### Non-goals (for now)
- No downloading from YouTube or any other site. Input is always a local file.
- No GUI or web app. CLI only.
- No real-time or streaming processing.
- No training or fine-tuning of models.
- No generated singing vocals or lyrics. Voice in the song is the speaker only.

---

## 2. Principles

- **Exact speech.** Clips are sample-accurate slices of the (optionally cleaned) source. Only short edge fades and level changes are allowed by default. Pitch correction is an explicit opt-in "stylize" mode, off by default.
- **Cache everything.** Each stage reads files from a run directory and writes files back. A stage is skipped if its outputs exist and its inputs are unchanged (use content hashes in a `manifest.json`), unless `--force` is passed.
- **Cheap before expensive.** Local and free stages come first. Paid calls happen only in the Claude steps and the ElevenLabs music step.
- **Cost safety.** Every paid stage supports `--dry-run` (prints an estimate, makes no calls) and asks for confirmation unless `--yes` is passed. All spending is appended to `costs.json` in the run directory.
- **Swappable backends.** Music generation sits behind an interface with at least two implementations: `elevenlabs` and `stub` (a local placeholder generator that produces simple tones/noise at the right key and tempo, so the whole pipeline can be developed and tested without spending credits).
- **No secrets in code.** Keys come from environment variables or a `.env` file that is gitignored.

---

## 3. Tech stack

- Python 3.11+, managed with `uv` (or `venv` + pip). Package layout under `src/`.
- CLI: `typer`.
- Config and presets: YAML, validated with `pydantic`.
- Audio I/O and processing: `ffmpeg` (system), `soundfile`, `numpy`, `scipy`, `librosa`, `pyloudnorm`.
- Transcription and alignment: `faster-whisper` or `whisperx` (local) as default. Optional ElevenLabs Scribe backend. Official-transcript alignment via fuzzy token alignment (`rapidfuzz` or `difflib`).
- Voice cleanup (optional): `demucs` locally. Optional ElevenLabs voice isolation backend.
- Pitch tracking: `librosa.pyin` first; `torchcrepe` as an optional upgrade.
- Symbolic music: `pretty_midi`. Rendering: `pyfluidsynth` or the `fluidsynth` CLI with a General MIDI soundfont.
- Mixing and effects: `pedalboard`.
- APIs: `anthropic` SDK, `elevenlabs` SDK.
- Tests: `pytest`. Lint/format: `ruff`.

Pin versions in `pyproject.toml`. Do not add heavy dependencies without noting why.

---

## 4. Repository layout

```
speech-to-song/
  CLAUDE.md                 # short rules for Claude Code
  README.md                 # user-facing quickstart
  pyproject.toml
  .env.example
  .gitignore
  docs/
    SPEC.md                 # this file
    DECISIONS.md            # log of design decisions and changes to the spec
  presets/
    cinematic_future_bass.yaml
    soft_piano.yaml         # later
    lofi.yaml               # later
  src/speech2song/
    __init__.py
    cli.py                  # typer app: one command per stage plus `run`
    config.py               # settings, env loading, preset schema (pydantic)
    manifest.py             # run dir, hashing, caching, costs.json
    models.py               # pydantic models for all JSON artifacts
    stages/
      ingest.py
      transcribe.py
      align.py
      select_clips.py
      melody.py
      arrange.py
      generate_music.py
      mix.py
    backends/
      music_base.py         # MusicBackend interface
      music_elevenlabs.py
      music_stub.py
      transcribe_whisper.py
      transcribe_scribe.py  # optional
    audio/
      io.py
      pitch.py
      theory.py             # key detection, scale snapping, chord fitting
      dsp.py                # ducking, fades, loudness
    llm/
      claude.py             # thin wrapper: JSON output, retries, token/cost logging
      prompts/
        select_clips.md
        arrange.md
  tests/
    fixtures/               # tiny synthetic audio only, no copyrighted material
  inputs/                   # gitignored: user's source audio/video/transcripts
  runs/                     # gitignored: one subdirectory per run
```

---

## 5. Run directory and artifacts

Each run lives in `runs/<run_id>/` (`run_id` = timestamp plus short source slug). A run
holds one talk and songs of several lengths (M8), each in its own folder; paths inside a
length's files are relative to its folder. Contents:

```
runs/<run_id>/
  manifest.json          # inputs, stage status (the talk's and each length's), hashes, settings
  costs.json             # append-only log of paid calls (service, model, units, est. USD, length)
  00_source.wav          # normalized mono/stereo 44.1 kHz source audio
  01_clean.wav           # optional voice-isolated version (else symlink/copy of 00)
  02_transcript.json
  <length>/              # summary, highlights, short
    03_selection.json
    03_clips.json
    clips/clip_001.wav ...
    04_melody.json
    04_melody.mid
    04_melody_reference.wav
    05_arrangement.json
    06_music/take_001.mp3 + take_001.meta.json ...
    06_music/selected.wav
    07_mix/stems/{speech.wav,music.wav,melody_layer.wav}
    07_mix/master.wav
```

Runs made before M8 (their song at the run root) become their `summary` length the first
time they are opened: the files and stage records move unchanged, so nothing re-runs.

---

## 6. Stages in detail

### Stage 1: Ingest
- Accept audio or video. Use `ffmpeg` to extract audio and write `00_source.wav` (44.1 kHz, 24-bit or float, keep stereo if present, also make a mono analysis copy in memory).
- Optional `--isolate-voice`: run `demucs` (or ElevenLabs isolation backend) to produce `01_clean.wav`. Pitch tracking and clip cutting use the clean version; keep the original for reference.
- Record duration, sample rate, channels, loudness in the manifest.

### Stage 2: Transcribe and align
- Run local Whisper (default) to get word-level timestamps and confidence. Output `02_transcript.json`:
  ```json
  {
    "source": "inputs/talk.mp3",
    "duration_s": 1234.5,
    "language": "en",
    "words": [{"w": "Hello", "start": 0.12, "end": 0.40, "conf": 0.98}],
    "sentences": [{"id": 1, "text": "...", "start": 0.12, "end": 4.30}],
    "official_transcript_used": false
  }
  ```
- If an official transcript is supplied: normalize both texts (case, punctuation, numerals, hyphens), align ASR tokens to official tokens with sequence alignment, and transfer timestamps to the official words. Where alignment fails, keep the ASR word and flag it. The `sentences` come from the official text when available.
- Report an alignment quality score (fraction of official words matched).

### Stage 3: Clip selection (Claude)
- Input to Claude: sentence-level transcript with start/end times (not word-level, to save tokens), the preset's description, and target parameters (number of clips, min/max seconds, total speech seconds budget).
- Claude must return **strict JSON** matching `models.ClipSelection`:
  ```json
  {
    "clips": [
      {"id": "c1", "start_sentence": 12, "end_sentence": 13,
       "text": "...", "score": 0.93,
       "role": "hook|build|payoff|breakdown|outro",
       "reason": "why this works as a standalone line"}
    ],
    "suggested_order": ["c3", "c1", "c4"],
    "notes": "optional overall guidance"
  }
  ```
- Selection criteria in the prompt: self-contained, punchy, emotionally or philosophically resonant, clean start and end, avoids references that need earlier context, avoids applause/laughter/hymn segments, varied in pace and idea, total duration within budget.
- Convert sentence ranges to time ranges, then **refine boundaries in code**: search within about +/-150 ms of each boundary for the lowest-energy frame, never cut inside a word, and apply short fades (default 15 ms, configurable, 0 allowed). Write `03_clips.json` and the WAV files in `clips/`.
- Validate: every clip duration within bounds, no overlaps, quote text actually matches the transcript for that range (reject hallucinated ranges and retry once with feedback).
- Support `--interactive-review`: print clips with text and timestamps; let the user drop, reorder or audition (writes preview WAVs) before continuing.
- Lengths (M8): one call chooses the quotes of every length a command asks for that still needs some ("versions": every quote once, and per length the quotes it plays, each length checked against its own targets). The first length's `03_selection.json` holds the call; the others copy their part. A length chosen later is shown the quotes the run's other lengths already play and keeps the strongest of them (a kept quote keeps its ID and sentence range). The summary chosen on its own keeps its pre-M8 prompt.

### Stage 4: Melody extraction (speech-to-song)
For each clip:
1. Track F0 with pYIN across the clip (voiced/unvoiced), smooth with median filtering, and remove octave jumps.
2. Segment into note events using word and syllable boundaries (word timestamps from stage 2, split further at energy/voicing dips). Each segment's pitch = duration-weighted median of voiced frames, converted to MIDI note numbers.
3. Estimate the key across all clips (Krumhansl-Schmuckler profiles on a duration-weighted pitch-class histogram). Prefer the preset's mode (minor/major) as a prior; allow `--key` override.
4. Snap notes to the scale using `scale_snap_strength` (0 = raw contour, 1 = fully snapped), keep octave in a sensible vocal-melody range.
5. Quantize rhythm to the preset grid at the chosen tempo. Tempo selection: search BPM within the preset's tolerance to minimize total misalignment of clip start/end times to the bar grid (or beats), since speech clips are not time-stretched.
6. Optionally repeat each phrase `loop_phrase_count` times in the melody layer (repetition is key to the speech-to-song illusion).
7. Derive a simple chord progression by choosing, per bar, the diatonic triad that best covers the melody notes (with a bias toward common progressions).
8. Outputs: `04_melody.json` (key, bpm, notes, chords per clip), `04_melody.mid`, and `04_melody_reference.wav` rendered via FluidSynth (soft piano by default). The reference WAV is intentionally short (a loop or two of the main phrase), since it will be used as an Audio Reference for generation.

Optional "stylize" mode (off by default): gently pitch-correct the spoken clips toward the scale with PSOLA (`parselmouth`) or `pyrubberband`, at a configurable strength. Output to separate files, never overwrite the exact clips.

Lengths (M8): the first length of a run whose melody is made is its anchor. Its tempo, key, the speaker's tuning offset and the melody's octave are kept in the manifest and used by the run's other lengths instead of being found from their own clips (`--key` still wins), so a quote in two lengths gets the same notes and chords.

### Stage 5: Arrangement
- Place clips on a bar-aligned timeline using the order from stage 3 (or the user's edited order). Insert sections from the preset's `arc` and `section_roles`: intro, speech beds under clips, builds, gaps, drops, breakdowns, outro.
- Claude may be asked (optional, cheap call) to refine the arc given the clip texts, e.g. place the most climactic line just before the main drop. Output must be strict JSON.
- How the music meets each speech passage (M7): it leads in, settling into a quiet bed a bar before the first word, and stays naturally quiet under the words ("under"); or, for the one or two lines that matter most ("alone"), the quiet bed carries the line until its last phrase, which lands in silence, and the music returns a beat or two after the last word (M7.1; in M7 the music stopped for the whole passage). Sections develop across the song (a role's first and last occurrence differ; Claude may add a few style words per part), music sections take chord progressions that fit the nearby speech melody, and the song's ending (held chord, stop or fade) is chosen to suit it.
- Output `05_arrangement.json`:
  ```json
  {
    "bpm": 114, "key": "C minor", "time_signature": "4/4",
    "sections": [
      {"id": "s1", "role": "intro", "start_bar": 0, "bars": 8, "energy": 0.15,
       "styles": ["..."], "clip_id": null},
      {"id": "s2", "role": "speech_bed", "start_bar": 8, "bars": 4,
       "energy": 0.2, "styles": ["..."], "clip_id": "c3", "clip_offset_beats": 0}
    ],
    "total_bars": 96, "total_seconds": 202.1
  }
  ```
- Print an ASCII timeline of the arrangement for review. Support `--stop-after arrange` so the user can approve before spending on music generation.
- Lengths (M8): each length has its own arc, bars and clip targets in the preset (`lengths:`; `summary` is the preset itself), and an arc entry may give its part's bars ("drop 16"). A length with a time window (`short`: 61–75 s) has its music sections lengthened or shortened (0.5–2x their bars, whole bars) until the song plus its ending's ring fits, changing as little as possible. A length that isn't the anchor plays the anchor's chords in its music sections (a role's last occurrence takes the anchor's last, the others the anchor's in turn).

### Stage 6: Backing track generation
- Backend interface: `generate(arrangement, preset, melody_reference) -> list[Take]`, with a `Take` containing audio path, metadata, cost and (when available) a stored-song ID for inpainting.
- **ElevenLabs backend.** Use the official `elevenlabs` Python SDK. Models: `music_v2` and `music_v2_5` (configurable). Build a composition plan from the arrangement: one or more sections per arrangement section, with durations matching the bars, global positive/negative styles from the preset, and local styles per section role. Before writing this code, **read the current ElevenLabs Music docs** (composition plans, `compose`, `compose_detailed` with `store_for_inpainting`, `music.upload`, Audio Reference, inpainting) and use the real field names. Do not guess the schema. Note documented duration limits (single generation vs composition-plan generation) and chunk limits, and split long arrangements accordingly.
- Attach the melody reference as Audio Reference if the API supports it for this model; treat it as a soft guide, not a guarantee.
- Handle the copyrighted-material error: prompts must contain no artist or song names; if the API returns a suggested plan, log it and surface it.
- Generate `n_takes` (default 2). Save every take with its metadata. Use `store_for_inpainting` so sections can be regenerated later.
- **Analysis of each take** (librosa): estimated tempo (and half/double-time ambiguity), key, section energy contour, and loudness. Compare with targets. Flag takes whose key/tempo drift past tolerance. Optionally correct small drift locally (pitch-shift or time-stretch the music only, never the speech).
- **Inpainting loop**: `speech2song regenerate --section s5 --note "less busy"` regenerates just that section while keeping the rest, using audio reference chunks. Keep this behind a flag until the basic flow works.
- **Music time (M7)**: in M7 arrangements nothing is generated for a passage played alone; the take is spliced open there afterwards, so the music on either side is exactly as generated. Since M7.1 such a passage has a bed like any other and the mix silences its last phrase. The music model fades out at the end of any generation, so a song ending on a held chord or a stop is generated past its end and cut on its last bar line.
- **Lengths (M8)**: a length that isn't the anchor is conditioned on the anchor's chosen take, through the take's stored song (a stored song can condition a new one; nothing is uploaded): every chunk on the anchor's section of the same role (a role's last chunk on the anchor's last, the others in turn; the middle 30 s of a longer one; `elevenlabs.anchor_strength`, low). Its generation waits until the anchor's take is chosen, and the take it was made with is kept, so choosing another anchor take later doesn't ask for its music again.
- **Stub backend**: produces a placeholder track at the arrangement's tempo and key (chord pads, kick on beats, energy envelope per section) using numpy/pedalboard. Used for tests and for developing the mixer for free.

### Stage 7: Mix and master
- Build the speech track: clips placed on the timeline at sample-accurate positions (start of the matching speech_bed section plus offset), original audio untouched except for edge fades.
- Music under speech (M7): the arrangement's own quiet beds do the work. The mix never follows the voice word by word: under each passage the music is set once, at the passage's bar lines, only as far as needed to sit a margin under the speech (at most `sidechain_duck_db`). A per-word guard keeps every word clear. For a passage played alone the music stops just before its last phrase with a short reverb ring and swells back in a beat or two after the last word (at most two beats of silence after the words). At the end, a held chord rings out through a long reverb, a stop gets a short ring, and a fade is left as generated.
- Speech processing chain (`pedalboard`): high-pass, gentle compression, optional EQ, reverb and delay sends per preset. Reverb/delay tails must not clip the next clip's start.
- Optional melody layer: the rendered melody reference mixed low under or over the track in the matching key, per preset.
- Master: limiter, loudness normalization to preset target (default -14 LUFS), true-peak ceiling -1 dBTP.
- Outputs: `master.wav` (24-bit), `master.mp3`, and stems.

---

## 7. CLI

```
speech2song run INPUT [--transcript FILE] [--preset NAME] [--isolate-voice]
                      [--length short|highlights|summary[,...]] [--clips N]
                      [--music-backend elevenlabs|stub]
                      [--dry-run] [--yes] [--stop-after STAGE] [--force]
speech2song ingest | transcribe | select | melody | arrange | generate | mix  (each takes --run RUN_ID;
                      select to mix also --length)
speech2song regenerate --run RUN_ID --section SECTION_ID [--note TEXT]
speech2song costs --run RUN_ID
speech2song presets list
```

Every command prints what it read, what it wrote, and any estimated or actual spend.

---

## 8. Configuration

- `.env`: `ANTHROPIC_API_KEY`, `ELEVENLABS_API_KEY`. Never log keys.
- `config.yaml` (optional, gitignored): default models and paths, e.g.
  `claude_model: claude-sonnet-5-5` (override with `claude-opus-5-5` for harder selection tasks), `music_model: music_v2_5`, `transcribe_backend: whisper`, `whisper_model: small`.
- Presets in `presets/*.yaml` validated by a pydantic schema (see `presets/cinematic_future_bass.yaml` for the first one).

---

## 9. Cost tracking

- `claude.py` logs input/output tokens per call and computes USD from a configurable price table (do not hardcode prices without a config override; prices change).
- ElevenLabs cost is estimated from minutes generated times a configurable rate, and logged as an estimate. Credits-based plans differ from per-minute API rates, so both rate and unit are configurable.
- `--dry-run` prints: expected Claude tokens and USD, expected music minutes, takes and USD, and the total, then exits.
- Confirmation prompt before any paid call unless `--yes`.
- Expected scale: roughly 40-50k Claude input tokens and 8-10k output tokens per song, plus a few minutes of music generation including retries.

---

## 10. Testing strategy

- Unit tests with synthetic fixtures only (generated sine sweeps, noise bursts, synthetic "speech-like" pitched tones). Do not commit copyrighted audio.
- Tests for: boundary snapping, transcript alignment, key detection on known scales, scale snapping, bar-grid tempo search, arrangement timeline math, ducking envelope, LUFS targeting.
- Mock the Anthropic and ElevenLabs clients in tests; no network in CI.
- One end-to-end test that runs the whole pipeline with the stub backend and a mocked Claude response, asserting that the clip audio inside `speech.wav` is sample-identical to the cut clips (aside from fades).

---

## 11. Milestones

- **M0 Scaffold:** project layout, config, manifest/caching, cost logging, CLI skeleton, preset loading, `.env` handling, tests run.
- **M1 Ingest, transcribe, align:** working on a real talk with and without an official transcript.
- **M2 Clip selection:** Claude JSON selection with validation, boundary snapping, review mode, `--dry-run`.
- **M3 Melody:** pitch tracking, key detection, MIDI and reference WAV output; a small listening-test notebook or script.
- **M4 Offline end-to-end:** arrangement, stub music backend, mixer. A full song from a talk with zero music-API spend.
- **M5 ElevenLabs:** composition-plan generation, analysis, take selection, inpainting regeneration.
- **M6 Mix polish (free, local):** no muted sections (the pre-drop `gap` becomes a short lift: the build's tail and a swell, never silence); ducking per quote, not per word (the music eases down about 0.6 s before a quote's first word, stays down through its pauses and comes back over about 1.5 s after its last word); a gentler speech guard (slower slopes, nearby words handled together); a ring-out of the last chord so a song never just stops. Checked on the existing "Timing" take at no cost.
- **M7 Song form and melody (paid probe first):** a beginning, middle and end (opening acts, a developing middle, a climax that differs from the first drop, a resolution); per-section style words and richer chord progressions instead of one loop repeated; each quote's speech melody, rendered on a soft instrument and uploaded once, as a `conditioning_ref` (at most 30 s, `low`/`medium` strength) for the music around that quote. Quotes are woven in rather than ducked: the music leads into each passage and is composed quiet under it (set once per passage in the mix, never per word), and a key line may play with no music at all. The ending comes from the song (a held chord rung out, a crisp stop, or the music's own fade; the closing line may be heard alone), not always a ring-out. Proven with one ~60 s probe before full takes, then compared with the current "Timing" song (its quotes reused). Revised after your first own run, "Songs Sung and Unsung" (M7.1): a line played alone keeps its quiet bed until its last phrase, the music returns at most two beats after the words, and melody conditioning is off by default (the music didn't follow the melodies; the rendered piano's sound came through).
- **M8 Lengths `short`, `highlights`, `summary`:** `--length` on `run`/`select` (one or several; the other song commands take one). `short`: 2–4 quotes, a one-drop form fitted to 61–75 s. `highlights`: 3–7 key quotes, 2–3.5 minutes. `summary`: today's behaviour (about 6–10 quotes, 4–6 minutes). Each length lives in its own folder inside the run; ingest, transcribe and align are shared. Lengths of one run share tempo, key and chords, taken from the run's anchor (the first length whose melody is made); lengths asked for together are chosen in one Claude call, and a length chosen later sees the quotes of lengths already made and keeps the strongest; its music is conditioned on the anchor's chosen take, section by section (each reference at most 30 s, from the take's stored song), so the versions sound like one song. Existing runs' songs become their `summary` length.
- **M9 Length `full`:** the whole talk in order, no selection. Claude splits the transcript into points and phrases and marks each point's weight; phrases stay sample-exact cuts of the source, with music between them. A minor point gets a short gap, a key point a swell, a major point a build and drop. A 28-minute talk makes a ~35–40 minute song, over the 10-minute limit of one composition plan, so the music is several generations joined on the bar grid, each conditioned on the end of the one before. One take by default (about $5–6 of music for a 28-minute talk).
- **M10 Styles and polish:** soft piano and lofi presets, stylize mode, README, better review UX.

Work one milestone at a time. After each, summarize what was built, what was tested, and what is uncertain.

---

## 12. Risks and open questions

- Whether Audio Reference meaningfully follows a supplied melody (likely a soft guide only). Keep the local melody layer as a fallback so the melody is guaranteed.
- Whether the music model follows chords given as words (undocumented: partly, measured in M7) and per-section melody conditioning from a rendered speech melody (measured on the first full run: the conditioned sections' chroma matched their quotes no better than other quotes', so it is off by default since M7.1).
- Joining several generations into one long `full` song without audible seams (M9).
- Whether preserved audio chunks in inpainting can sit under speech inside a generation. Not assumed; speech is mixed locally.
- Pitch tracking on noisy or reverberant recordings. Voice isolation helps; expose a confidence threshold.
- Tempo ambiguity (half-time vs double-time) in generated music. Analysis must handle both.
- Generated tracks may not match the requested key exactly. Plan for local pitch/time correction on the music only.
- Rights: source recordings are copyrighted. The tool is for personal experimentation; releasing a track needs permission from rights holders and compliance with ElevenLabs' music terms for the user's plan.
- ElevenLabs API surface (model IDs, plan schema, limits, pricing) changes. Read current docs before coding stage 6 and again if calls fail.
