# Speech-to-Song

Personal CLI tool: spoken-word recording in, finished electronic song out. Claude picks the best lines, speech pitch is turned into a melody, ElevenLabs Music generates the backing track, and everything is mixed locally.

**Read `docs/SPEC.md` before doing anything.** It is the source of truth. Style presets live in `presets/`.

## Working rules

- Work on one milestone at a time (see SPEC section 11). Start each milestone by stating a short plan, and end it by summarizing what was built, what was tested and what is uncertain.
- If the spec seems wrong or incomplete, stop and propose a change. Record accepted changes in `docs/DECISIONS.md`.
- Never spend money implicitly. Anything calling the Anthropic or ElevenLabs APIs must support `--dry-run`, ask for confirmation unless `--yes`, and log to `costs.json`.
- Do not run paid API calls yourself to test code. Use mocks and the `stub` music backend. If a live call is needed, ask me first.
- Speech clips must stay sample-exact (edge fades only). Never route speech through a generative model.
- Before writing ElevenLabs code, read the current official docs (composition plans, compose_detailed, inpainting, Audio Reference). Do not guess field names or model IDs.
- Never commit secrets, user audio, or copyrighted material. `inputs/`, `runs/`, `.env` are gitignored. Test fixtures are synthetic only.
- Python 3.12+ (see DECISIONS), type hints, `ruff` for lint/format, `pytest` for tests. Keep dependencies lean and explain any heavy additions.
- Prefer small, composable functions with pure logic separated from I/O, so audio math is easy to test.

## How I like to work

- After each milestone, stop with a summary (what was built, what was tested, what is uncertain) and wait for my go-ahead before starting the next one.
- Commit at the end of each milestone, once `pytest` and `ruff` pass. Never commit `.env`, `inputs/`, `runs/` or `outputs/`.
- Ask before every live paid call, even when the code is ready. Say which model and the estimated cost.
- Check each milestone on the real talk in `inputs/` with the free, local stages, and say where the files are that I should listen to.
- When a stage's logic changes, bump its `version` so caches update. But never in a way that silently repeats a paid call: keep paid stages' outputs raw and put clean-up rules in free stages.
- Current status and open questions are at the top of `docs/DECISIONS.md`.

## Commands

- Install: `uv sync --extra isolate` (plain `uv sync` removes demucs/torch; `pip install -e .[dev]` also works)
- Test: `uv run pytest` (fast, offline); `uv run pytest -m integration` runs the real Whisper and demucs models
- Lint: `uv run ruff check . && uv run ruff format .`
- CLI: `uv run speech2song --help`
- Melody listening test: `uv run python scripts/melody_listen.py [RUN]`
