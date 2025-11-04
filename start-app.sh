#!/usr/bin/env bash
set -euo pipefail

# WaveCap-SDR startup script
# Starts the server with sensible defaults
#
# Optional environment variables:
#   HOST=0.0.0.0              # Bind address (default: 0.0.0.0)
#   PORT=8087                 # Port number (default: 8087)
#   DRIVER=soapy              # SDR driver (default: soapy)
#   DEVICE_ARGS="driver=..."  # Specific device arguments (optional)
#   CONFIG=path/to/config.yaml # Config file path (optional)
#
# Examples:
#   ./start-app.sh                                    # Start with defaults
#   HOST=127.0.0.1 PORT=8088 ./start-app.sh          # Custom host/port
#   DEVICE_ARGS="driver=rtlsdr" ./start-app.sh       # Specific device

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
VENV_DIR="$BACKEND_DIR/.venv"

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}WaveCap-SDR Startup${NC}"
echo "================================"

# Check if we're in the right directory
if [ ! -d "$BACKEND_DIR" ]; then
  echo "Error: backend/ directory not found"
  echo "Please run this script from the WaveCap-SDR root directory"
  exit 1
fi

cd "$BACKEND_DIR"

# Set up virtual environment if needed
if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo -e "${YELLOW}Setting up Python virtual environment...${NC}"
  python3 -m venv --system-site-packages "$VENV_DIR"
  "$VENV_DIR/bin/python" -m pip install --upgrade pip --quiet
  echo -e "${GREEN}Installing dependencies...${NC}"
  "$VENV_DIR/bin/python" -m pip install fastapi uvicorn httpx websockets pyyaml numpy scipy --quiet
fi

# Default values
: "${HOST:=0.0.0.0}"
: "${PORT:=8087}"
: "${DRIVER:=soapy}"

# Set environment variables for WaveCap-SDR config
# These override config file settings
export WAVECAPSDR__SERVER__BIND_ADDRESS="$HOST"
export WAVECAPSDR__SERVER__PORT="$PORT"
export WAVECAPSDR__DEVICE__DRIVER="$DRIVER"
[ -n "${DEVICE_ARGS:-}" ] && export WAVECAPSDR__DEVICE__DEVICE_ARGS="$DEVICE_ARGS"

echo ""
echo "Configuration:"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Driver: $DRIVER"
[ -n "${DEVICE_ARGS:-}" ] && echo "  Device: $DEVICE_ARGS"
[ -n "${CONFIG:-}" ] && echo "  Config: $CONFIG"
echo ""
echo -e "${GREEN}Starting WaveCap-SDR server...${NC}"
echo "Web UI will be available at: http://$HOST:$PORT/"
echo "Press Ctrl+C to stop"
echo ""

# Build command arguments
CMD_ARGS=(
  -m wavecapsdr
)

# Add optional config file if provided
[ -n "${CONFIG:-}" ] && CMD_ARGS+=(--config "$CONFIG")

# Start the server (config comes from environment variables)
PYTHONPATH=. "$VENV_DIR/bin/python" "${CMD_ARGS[@]}"
