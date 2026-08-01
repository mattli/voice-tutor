#!/bin/bash
set -e
set -o pipefail

cd "$(dirname "$0")"

# ── Guard: never let a by-hand run kill the live server ─────────────────────
# The production launchd agent runs THIS script and sets XPC_SERVICE_NAME to its
# job label; we let that instance through. Any other (interactive) run, while the
# agent is loaded, is refused — otherwise the `kill -9` below would take down the
# live server on :7860 and then fight KeepAlive for the port.
if [ "$XPC_SERVICE_NAME" != "com.voice-tutor.server" ] \
   && launchctl print "gui/$(id -u)/com.voice-tutor.server" >/dev/null 2>&1; then
    echo "Refusing: launchd agent 'com.voice-tutor.server' is loaded and already serving :7860."
    echo "Running start.sh by hand would kill the live server. To restart production, run:"
    echo "  launchctl kickstart -k gui/$(id -u)/com.voice-tutor.server"
    exit 1
fi

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

exec uv run uvicorn app:app --host 0.0.0.0 --port 7860 --proxy-headers "$@" 2>&1 | tee -a voice-tutor.log
