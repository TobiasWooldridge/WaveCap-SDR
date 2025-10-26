#!/usr/bin/env bash
set -euo pipefail

# Convenience harness for agents: KEXP (90.3 MHz) WBFM check via SoapySDR
# Optional env:
#   DEVICE_ARGS="driver=rtlsdr" | "driver=sdrplay,serial=..."
#   DURATION=10   AUDIO_RATE=48000
#   HOST=127.0.0.1 PORT=8097   # override bind/port to avoid conflicts

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKEND_DIR="$REPO_ROOT/backend"
VENV_DIR="$BACKEND_DIR/.venv"

cd "$BACKEND_DIR"

if [ ! -x "$VENV_DIR/bin/python" ]; then
  python3 -m venv --system-site-packages "$VENV_DIR"
  "$VENV_DIR/bin/python" -m pip install --upgrade pip >/dev/null
  "$VENV_DIR/bin/python" -m pip install fastapi uvicorn httpx websockets pyyaml numpy >/dev/null || true
  # Attempt to install SoapySDR Python bindings (may already be provided by system)
  "$VENV_DIR/bin/python" -m pip install SoapySDR >/dev/null || true
fi

: "${DURATION:=10}"
: "${AUDIO_RATE:=48000}"

set +e
PYTHONPATH=. "$VENV_DIR/bin/python" -m wavecapsdr.harness \
  --start-server \
  --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-8087}" \
  --driver soapy \
  --preset kexp \
  --duration "$DURATION" \
  --audio-rate "$AUDIO_RATE" \
  ${DEVICE_ARGS:+--device-args "$DEVICE_ARGS"} \
  ${GAIN:+--gain "$GAIN"}
STATUS=$?
set -e

echo "Harness exit code: $STATUS"
echo "WAV output (if any): $BACKEND_DIR/harness_out"
exit "$STATUS"
