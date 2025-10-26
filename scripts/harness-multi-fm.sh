#!/usr/bin/env bash
set -euo pipefail

# Multi-device FM (KEXP) harness via SoapySDR. Creates one capture per device and streams audio.
# Env:
#   HOST=127.0.0.1 PORT=8097 DURATION=10 AUDIO_RATE=48000
#   FILTER_DRIVERS="rtlsdr,sdrplay"   # optional: restrict to specific drivers

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKEND_DIR="$REPO_ROOT/backend"
VENV_DIR="$BACKEND_DIR/.venv"

cd "$BACKEND_DIR"

if [ ! -x "$VENV_DIR/bin/python" ]; then
  python3 -m venv --system-site-packages "$VENV_DIR"
  "$VENV_DIR/bin/python" -m pip install --upgrade pip >/dev/null
  "$VENV_DIR/bin/python" -m pip install fastapi uvicorn httpx websockets pyyaml numpy >/dev/null || true
  "$VENV_DIR/bin/python" -m pip install SoapySDR >/dev/null || true
fi

: "${HOST:=127.0.0.1}"
: "${PORT:=8097}"
: "${DURATION:=10}"
: "${AUDIO_RATE:=48000}"

# Build device-args list by querying /devices when available; otherwise fall back to drivers hint
FILTER="${FILTER_DRIVERS:-}"

echo "Starting multi-device FM harness on $HOST:$PORT (duration ${DURATION}s)" >&2

set +e
PYTHONPATH=. "$VENV_DIR/bin/python" - <<PY
import os, sys, json, time
import httpx

HOST=os.environ.get('HOST','127.0.0.1')
PORT=int(os.environ.get('PORT','8097'))
DURATION=float(os.environ.get('DURATION','10'))
AUDIO_RATE=int(os.environ.get('AUDIO_RATE','48000'))
FILTER=set([s for s in os.environ.get('FILTER_DRIVERS','').split(',') if s])

base=f"http://{HOST}:{PORT}/api/v1"

async def main():
    async with httpx.AsyncClient(base_url=base, timeout=10.0) as client:
        # Start in-process server by relying on harness? Instead, assume external start via --start-server below.
        pass

print('Note: this script assumes server is started by the harness call below.', file=sys.stderr)
PY
STATUS=$?
set -e

# Use the Python harness to start the server and run on all devices at once
# Add extra headroom to accommodate auto-gain probing across multiple devices.
TIMEOUT_SECONDS=$((DURATION + 80))
"$REPO_ROOT/scripts/run-with-timeout.sh" --seconds ${TIMEOUT_SECONDS} -- \
  env PYTHONPATH=. "$VENV_DIR/bin/python" -m wavecapsdr.harness \
    --start-server \
    --host "$HOST" \
    --port "$PORT" \
    --driver soapy \
    --preset kexp \
    --duration "$DURATION" \
    --auto-gain \
    --audio-rate "$AUDIO_RATE" \
    --all-devices

echo "WAV output (if any): $BACKEND_DIR/harness_out" >&2
