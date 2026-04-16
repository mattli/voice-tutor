#!/bin/bash
set -e

cd "$(dirname "$0")"

# Clear any prior bot still bound to port 7860
lsof -ti :7860 | xargs kill -9 2>/dev/null || true

# Source .env for API keys
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Check required keys
for key in ANTHROPIC_API_KEY DEEPGRAM_API_KEY CARTESIA_API_KEY; do
    if [ -z "${!key}" ]; then
        echo "Error: $key is not set. Add it to .env"
        exit 1
    fi
done

exec uv run python bot.py --host 0.0.0.0 "$@"
