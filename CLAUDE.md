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
- Python 3.11+, type hints, `ruff` for lint/format, `pytest` for tests. Keep dependencies lean and explain any heavy additions.
- Prefer small, composable functions with pure logic separated from I/O, so audio math is easy to test.

## Commands

- Install: `uv sync` (or `pip install -e .[dev]`)
- Test: `pytest`
- Lint: `ruff check . && ruff format .`
- CLI: `speech2song --help`
