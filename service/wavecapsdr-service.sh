#!/usr/bin/env bash
# WaveCap-SDR Service Wrapper
# This script is designed to be run by launchd as a background service.
# It sets up the environment and starts the WaveCap-SDR server.

set -euo pipefail

# Determine script location and project paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$PROJECT_DIR/backend"
VENV_DIR="$BACKEND_DIR/.venv"
LOG_DIR="$PROJECT_DIR/logs"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Redirect output to log file (launchd also captures stdout/stderr)
exec >> "$LOG_DIR/wavecapsdr.log" 2>&1

echo "=========================================="
echo "WaveCap-SDR Service Starting"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# Verify backend directory exists
if [ ! -d "$BACKEND_DIR" ]; then
    echo "ERROR: Backend directory not found: $BACKEND_DIR"
    exit 1
fi

# Verify virtual environment exists
if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "ERROR: Python virtual environment not found: $VENV_DIR"
    echo "Please run ./start-app.sh once to set up the environment"
    exit 1
fi

cd "$BACKEND_DIR"

# Default configuration (can be overridden via environment variables)
: "${HOST:=0.0.0.0}"
: "${PORT:=8087}"
: "${DRIVER:=soapy}"

# Export WaveCap-SDR configuration environment variables
export WAVECAPSDR__SERVER__BIND_ADDRESS="$HOST"
export WAVECAPSDR__SERVER__PORT="$PORT"
export WAVECAPSDR__DEVICE__DRIVER="$DRIVER"
[ -n "${DEVICE_ARGS:-}" ] && export WAVECAPSDR__DEVICE__DEVICE_ARGS="$DEVICE_ARGS"

echo "Configuration:"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Driver: $DRIVER"
[ -n "${DEVICE_ARGS:-}" ] && echo "  Device: $DEVICE_ARGS"
[ -n "${CONFIG:-}" ] && echo "  Config: $CONFIG"
echo ""

# Ensure PATH includes Homebrew for SoapySDR libraries
if [[ "$OSTYPE" == "darwin"* ]]; then
    export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
    # Ensure dynamic libraries can be found
    export DYLD_LIBRARY_PATH="/opt/homebrew/lib:${DYLD_LIBRARY_PATH:-}"
fi

# Build command arguments
CMD_ARGS=(-m wavecapsdr)
[ -n "${CONFIG:-}" ] && CMD_ARGS+=(--config "$CONFIG")

echo "Starting WaveCap-SDR server..."
echo "Web UI: http://$HOST:$PORT/"
echo ""

# Start the server
PYTHONPATH=. exec "$VENV_DIR/bin/python" "${CMD_ARGS[@]}"
