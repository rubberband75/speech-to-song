# speech2song

Personal CLI tool: spoken-word recording in, finished electronic song out. Claude picks the
best lines, the speech's pitch becomes a melody, ElevenLabs Music generates the backing track,
and everything is mixed locally. The speech itself is never regenerated: clips are cut
sample-exactly from the source.

`docs/SPEC.md` is the source of truth; accepted changes are logged in `docs/DECISIONS.md`.

**Status:** M5 — the whole pipeline runs: ingest, voice isolation, transcription,
alignment, clip selection, the speech melody, arrangement, the backing track (Eleven Music,
or the free `stub` placeholder), take analysis, section regeneration and the mix.

## Install

Needs Python 3.12+, [uv](https://docs.astral.sh/uv/), `ffmpeg`/`ffprobe`, and `fluidsynth`
with a General MIDI soundfont (Ubuntu: `fluidsynth fluid-soundfont-gm`).

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
speech2song select --dry-run         # estimated Claude cost, no calls
speech2song select --interactive-review   # Claude picks lines (asks first); review them
speech2song select --quotes my_quotes.yaml   # lines the song must include
speech2song melody                  # speech pitch -> melody, MIDI, audio reference
uv run python scripts/melody_listen.py   # listening tests in runs/<run>/04_listen/
speech2song arrange                 # sections on the bar grid; prints the timeline
speech2song generate                # backing-track takes (stub: free)
speech2song generate --music-backend elevenlabs --dry-run   # Eleven Music: estimate first
speech2song regenerate --section s3-s5 --note "less busy"   # redo sections (paid)
speech2song regenerate --undo       # back to the take's previous version (free)
speech2song mix                     # pick a take, then stems, master.wav, master.mp3
speech2song mix --take 2            # mix another take (free); --take auto: best again
speech2song run inputs/talk.mp3 --transcript inputs/talk.txt      # every step
speech2song run --run latest --stop-after arrange   # resume; stop to check the arrangement
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

### Clips

`select` sends the sentence list (not the audio) to Claude, which picks lines by sentence
ID: about 8 (6 to 10), chosen so that together they retell the whole talk, usually in
its own order. Lines run 3-15 s, or up to 40 s for a passage that is essential to the
talk. The answer is checked against the transcript (ranges exist, quotes match, durations
fit, no overlaps, total within budget); one retry with feedback is allowed. The `clips`
stage then cuts each line from `01_clean.wav` at the quietest point near the sentence
edges, adds 15 ms fades, and writes `clips/clip_NNN.wav` and `03_clips.json`. A line
with more than 15 s of speech is cut into parts at its natural pauses (sentence ends,
then commas, then the longest gaps), as clips `c3-1`, `c3-2`, ...; the arrangement keeps
them together with a bar of music between them. Apart from the fades, clip audio is
sample-identical to the source. Review choices (drop, reorder) are saved to
`03_review.json` and survive re-runs. Targets (count, lengths, speech budget, part
length, fades) live in the preset's `clips:` block; `--clips N` asks for exactly N.

`--quotes FILE` (on `select` and `run`, sticky; `--no-quotes` drops it) names quotes the
song must include. A `.txt` file has one quote per line, or one per paragraph when it
has blank lines. A `.yaml` file is a list of quotes, each a string or a mapping:

```yaml
- text: The Savior loves all of us and is tenderly calling for you and for me to come home.
  role: outro          # optional: hook, build, payoff, breakdown or outro
- text: Come home.
  occurrence: last     # optional, for a line the talk says more than once (1, 2, ... or last)
```

Each quote is matched to the transcript (small wording differences are fine) and widened
to whole sentences before any paid call; a quote that can't be found stops the step.
Claude is told to include them, and any it leaves out are added anyway.

### Melody

`melody` tracks each clip's pitch (pYIN), turns syllables into notes, removes the
speaker's tuning offset, finds the key (`--key "D minor"` overrides it), snaps notes
toward the scale (`melody.scale_snap_strength`), moves them up into a melody register,
picks the tempo within the preset's tolerance that best fits the clips, quantizes to the
preset grid, and chooses one chord per bar. It writes `04_melody.json`, `04_melody.mid`
(each phrase looped `loop_phrase_count` times) and `04_melody_reference.wav` (two loops
of the main phrase, rendered with fluidsynth and a General MIDI soundfont).
`scripts/melody_listen.py` writes, per clip, the speech, the melody, both together, and
an "illusion" take where the speech repeats while the melody fades in.

### Arrangement

`arrange` fits the clips into the preset's `arc`. The arc's `speech_bed` entries are
slots: the clips, in play order, are shared out over them in consecutive groups that
balance speech time (a `hook` clip leans to the first slot, an `outro` clip to the last),
and each clip gets its own speech bed, starting on a bar and long enough for the clip
plus `speech_interaction.tail_beats`. Other sections take their `bars` from the preset.
Beds use their clip's chords from the melody, other sections loop the main phrase's
chords, and breakdowns and drops note which line was heard last, for the melody layer.
Silent roles (`silent: true`, the gap before a drop) are muted in the mix.
`05_arrangement.json` can be edited by hand: later steps pick up the edit, and the mixer
checks it first.

`arrange --refine-arc` asks Claude to propose the arc around the clip texts instead (a
small paid call, about $0.03 on Sonnet; it asks first). Its answer is kept in
`05_arc.json`; if it breaks the rules (unknown roles, clips out of order, odd lengths)
it is retried once with feedback, and the preset's arc is used if it still can't be.
`--no-refine-arc` goes back to the preset's arc. Both flags stick to the run.

### Music and mix

`generate` makes `music_takes` takes (default 2) with the run's music backend
(`--music-backend`, sticky; default `stub` from config):

- **elevenlabs**: the arrangement becomes an Eleven Music composition plan (one chunk per
  section or merged run of sections, each stating the tempo, key and "instrumental only",
  with the section's styles). Silent sections and anything under 3 s ride along with the
  chunk before them; nothing is generated for a gap. Each take is one paid call ($0.15
  per generated minute at API rates, about $0.51 for a 3.4-minute song); it estimates and
  asks first. Takes are stored for inpainting and never overwritten: a re-run reuses
  takes made for the same request, `--force` adds new ones, and takes of an older request
  stay available (new takes get the next numbers) as long as their timing still fits the
  arrangement. Takes that no longer fit move to `06_music/archive/`.
- **stub**: a free placeholder (chord pads, bass, half-time drums, risers).

`generate` re-runs only when what it would send changes, not on every arrangement edit.

`regenerate --section sN [--note TEXT]` (ElevenLabs takes only, paid) regenerates a
section, or a range of neighbouring ones (`--section s3-s5`: build, gap and drop
together), of the current take and keeps the rest unchanged, then re-mixes. It works on
any take whose timing fits, even one made for an older request, and never runs
`generate`. Each result is a new version of the take (`take_NNN_v2.mp3`, ...); earlier
versions stay on disk, and `regenerate --undo` (free) goes back one version. If a
regeneration comes back too much like the original, `--adherence medium` (or `low`)
lets it differ more from the music around it.

`mix` first checks every take that fits the arrangement (tempo, allowing half time; key;
whether section levels follow the planned energy; loudness), writes
`06_music/analysis.json`, and picks the best of the takes made for the current request,
or `--take N` (any take that fits; sticky, `--take auto` to undo). Then it writes
`07_mix/`:

- `stems/speech.wav`: the clips, sample-exact, on silence (float WAV)
- `stems/music.wav` and `stems/melody_layer.wav`: as heard in the mix (ducked)
- `master.wav` (24-bit) and `master.mp3` (320 kbps), at the preset's `target_lufs` with
  true peaks at or below -1 dBTP
- `mix.json`: clip positions, levels, the music's shaping and final loudness

The music is shaped toward the arrangement first: a section whose loudness strays more
than `mix.energy_tolerance_db` from a line through the sections' median
(`energy_range_db` from energy 0 to 1) is pulled back by the excess, at most
`energy_max_db`. Silent sections are cut, leaving a short reverb tail of the music before
them (`gap_reverb`). Generated drops often open with a silent bar and a riser; when the
section after a gap opens near-silent and reaches full level within
`late_entry_max_bars` (4), its music is taken from that many bars later, so it starts
on the downbeat (and its spill into the next section goes back with it). The speech bus
sits `mix.speech_level_lu` above the music's
loudness, with a high-pass, gentle compression, and reverb/delay sends whose tails fade
out before the next clip. The music ducks by `sidechain_duck_db` while speech plays, and
the speech guard makes sure every word (from the transcript's timings) stays at least
`speech_margin_db` (10 dB) above the music in the speech band (200 Hz-5 kHz): under
words that are softer than that, the music dips further, smoothly and only as far as
needed. `mix.json` lists those words.
`--melody-layer` (on `mix` and `run`, sticky) adds the MIDI melody: `off` (the preset's
default), `replay` (breakdowns and drops replay the line just heard) or `all` (also
quietly under the speech).

### Official transcripts

Plain UTF-8 text, one paragraph per line (text copied from a PDF with hard line wraps is
detected and unwrapped). Lines that are not spoken, such as image captions and headings, are
detected during alignment and left out; spoken words missing from the transcript are kept
from the ASR. `transcribe` prints both lists so you can check them.

## Spending

Paid calls: `select` (Claude, about $0.05 per 13-minute talk), the optional
`arrange --refine-arc` (Claude), and with `--music-backend elevenlabs`, `generate` and
`regenerate` (Eleven Music). Everything else is local and free.
Every paid step estimates its cost first, supports `--dry-run` (no calls), asks before
spending unless `--yes` is given (and refuses without a terminal), and logs each call to
`costs.json` (`speech2song costs`). Claude list prices are built in with an `as_of` date;
override them under `pricing:` in `config.yaml`. If Claude declines a request, the API
retries it on a recommended fallback model (`claude_fallbacks: false` turns that off);
fallback attempts are priced at the serving model's rates.

## Development

```bash
uv run pytest                     # unit tests: synthetic fixtures only, no network
uv run pytest -m integration      # real Whisper/demucs on synthetic audio (downloads models)
uv run ruff check . && uv run ruff format .
```

Never commit secrets, user audio, or copyrighted material: `inputs/`, `runs/`, `outputs/`
and `.env` are gitignored, and test fixtures are generated or invented.
