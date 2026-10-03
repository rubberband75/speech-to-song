"""Write listening-test WAVs for a run's melody.

Usage: uv run python scripts/melody_listen.py [RUN] [LENGTH]
(RUN: ID, prefix, or latest; LENGTH: short, highlights or summary, default the run's)

For each clip in 04_melody.json: <clip>_1_speech, _2_melody, _3_overlay (melody under
the speech) and _4_illusion (the speech repeated while the melody fades in), in
runs/<run>/<length>/04_listen/.
"""

import sys

from speech2song.audio.synth import find_soundfont
from speech2song.config import load_config
from speech2song.listen import write_listening_set
from speech2song.manifest import Run


def main(argv: list[str]) -> int:
    config = load_config()
    run = Run.open(config.runs_dir, argv[1] if len(argv) > 1 else None)
    song = run.song(argv[2] if len(argv) > 2 else None)
    for path in write_listening_set(song, find_soundfont(config.soundfont)):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
