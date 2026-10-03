#!/usr/bin/env bash
# Set up speech2song on Linux Mint / Ubuntu / Debian after a fresh git clone.
#   ./scripts/install.sh              # everything, including demucs voice isolation (~1 GB)
#   ./scripts/install.sh --no-isolate # skip demucs/torch
# Safe to re-run.
set -euo pipefail

ISOLATE=1
for arg in "$@"; do
  case "$arg" in
    --no-isolate) ISOLATE=0 ;;
    -h|--help) sed -n '2,5p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "==> System packages (sudo needed)"
sudo apt-get update
sudo apt-get install -y curl ffmpeg fluidsynth fluid-soundfont-gm

echo "==> uv"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

echo "==> Python dependencies (uv fetches Python 3.12 itself if the system one is older)"
if [ "$ISOLATE" -eq 1 ]; then
  uv sync --extra isolate
else
  uv sync
fi

echo "==> .env"
if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env: add your ANTHROPIC_API_KEY and ELEVENLABS_API_KEY."
else
  echo ".env already exists, left alone."
fi

echo "==> Check"
uv run speech2song --help >/dev/null && echo "speech2song runs."
uv run pytest -q

cat <<'EOF'

Done. Next:
  - Put your API keys in .env.
  - If `uv` is not found in new shells, open a new terminal (the installer edits your shell profile).
  - Whisper large-v3-turbo (~1.6 GB) and demucs (~80 MB) download on first use.
  - Run with: uv run speech2song --help
EOF
