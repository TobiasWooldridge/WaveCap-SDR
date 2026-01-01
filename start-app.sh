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
#   SDRPLAY_FIX=1             # Run scripts/fix-sdrplay.sh preflight before start (optional)
#   WAVECAP_LOG_LEVEL=INFO    # Root log level (DEBUG, INFO, WARNING, ERROR)
#   WAVECAP_LOG_CONSOLE_LEVEL=INFO # Console log level
#   WAVECAP_LOG_FILE_LEVEL=DEBUG   # File log level
#   WAVECAP_LOG_STREAM_LEVEL=INFO  # Log stream (WebSocket) level
#   WAVECAP_LOG_SAMPLING_LEVEL=INFO # Sampling filter max level
#   WAVECAP_UVICORN_LOG_LEVEL=info  # Uvicorn log level (lowercase)
#
# Examples:
#   ./start-app.sh                                    # Start with defaults
#   HOST=127.0.0.1 PORT=8088 ./start-app.sh          # Custom host/port
#   DEVICE_ARGS="driver=rtlsdr" ./start-app.sh       # Specific device

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
VENV_DIR="$BACKEND_DIR/.venv"

# Add ~/.local/bin to PATH for DSD-FME and other locally installed tools
export PATH="$HOME/.local/bin:$PATH"

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

  # On macOS, prefer Homebrew's Python for SoapySDR compatibility
  PYTHON_CMD="python3"
  if [[ "$OSTYPE" == "darwin"* ]]; then
    # Try Homebrew Python versions (newest first)
    for py in /opt/homebrew/bin/python3.14 /opt/homebrew/bin/python3.13 /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3; do
      if [ -x "$py" ]; then
        PYTHON_CMD="$py"
        echo "Using Homebrew Python: $PYTHON_CMD"
        break
      fi
    done
  fi

  "$PYTHON_CMD" -m venv --system-site-packages "$VENV_DIR"
  "$VENV_DIR/bin/python" -m pip install --upgrade pip --quiet
  echo -e "${GREEN}Installing dependencies...${NC}"
  "$VENV_DIR/bin/python" -m pip install fastapi uvicorn httpx websockets pyyaml numpy scipy slowapi --quiet
fi

# Build frontend (requires Node.js)
FRONTEND_DIR="$SCRIPT_DIR/frontend"
STATIC_DIR="$BACKEND_DIR/wavecapsdr/static"
if [ -d "$FRONTEND_DIR" ]; then
  if ! command -v npm &> /dev/null; then
    echo "Error: Node.js/npm is required to build the frontend"
    echo "Install Node.js: https://nodejs.org/ or 'brew install node' on macOS"
    exit 1
  fi

  echo -e "${YELLOW}Building frontend...${NC}"
  cd "$FRONTEND_DIR"

  # Install dependencies if node_modules doesn't exist
  if [ ! -d "node_modules" ]; then
    echo "Installing npm dependencies..."
    npm ci --silent
  fi

  # Build the frontend
  npm run build --silent

  # Copy built files to backend static directory
  if [ -d "dist" ]; then
    mkdir -p "$STATIC_DIR"
    cp -r dist/* "$STATIC_DIR/"
    echo -e "${GREEN}Frontend built successfully${NC}"
  fi

  cd "$BACKEND_DIR"
fi

# Default values
: "${HOST:=0.0.0.0}"
: "${PORT:=8087}"
: "${DRIVER:=soapy}"
: "${SDRPLAY_FIX:=0}"

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
[ "$SDRPLAY_FIX" != "0" ] && echo "  SDRplay Fix: enabled"
echo ""

did_fix=0
if [ "$SDRPLAY_FIX" != "0" ] && [ -x "$SCRIPT_DIR/scripts/fix-sdrplay.sh" ]; then
  echo "Running SDRplay fix preflight..."
  if [ -x "$SCRIPT_DIR/scripts/run-with-timeout.sh" ]; then
    if "$SCRIPT_DIR/scripts/run-with-timeout.sh" --seconds 90 -- "$SCRIPT_DIR/scripts/fix-sdrplay.sh" --non-interactive --no-kill --no-start; then
      did_fix=1
      echo "SDRplay fix complete."
    else
      echo "Warning: SDRplay fix failed; continuing startup."
    fi
  else
    if "$SCRIPT_DIR/scripts/fix-sdrplay.sh" --non-interactive --no-kill --no-start; then
      did_fix=1
      echo "SDRplay fix complete."
    else
      echo "Warning: SDRplay fix failed; continuing startup."
    fi
  fi
fi

# Attempt to refresh SDRplay service so devices enumerate cleanly.
if [ "$did_fix" -eq 0 ] && [ -x "$SCRIPT_DIR/restart-sdrplay.sh" ]; then
  if "$SCRIPT_DIR/restart-sdrplay.sh" --non-interactive; then
    echo "Refreshed sdrplay service."
  else
    echo "Warning: Could not auto-restart sdrplay (sudo password likely required). Run ./restart-sdrplay.sh manually if needed."
  fi
fi

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
