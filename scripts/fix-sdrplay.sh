#!/bin/bash
# Fix stuck SDRplay device by power cycling USB and restarting services
# Usage: ./scripts/fix-sdrplay.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$PROJECT_ROOT/backend"

# SDRplay USB location (hub and port)
SDRPLAY_HUB="2-1.1.4"
SDRPLAY_PORT="3"
SDRPLAY_SERIAL="240309F070"

echo "=== SDRplay Fix Script ==="
echo "Target device: $SDRPLAY_SERIAL on hub $SDRPLAY_HUB port $SDRPLAY_PORT"
echo ""

# Step 1: Kill WaveCap-SDR
echo "[1/5] Stopping WaveCap-SDR..."
pkill -9 -f "python.*wavecapsdr" 2>/dev/null || true
pkill -9 -f "wavecapsdr" 2>/dev/null || true
sleep 2
if pgrep -f wavecapsdr > /dev/null 2>&1; then
    echo "  WARNING: Some processes still running, force killing..."
    pkill -9 -f wavecapsdr 2>/dev/null || true
    sleep 1
fi
echo "  Done."

# Step 2: Power off the USB port
echo "[2/5] Power cycling USB port (off)..."
if ! command -v uhubctl &> /dev/null; then
    echo "  ERROR: uhubctl not found. Install with: brew install uhubctl"
    exit 1
fi
sudo uhubctl -l "$SDRPLAY_HUB" -p "$SDRPLAY_PORT" -a off 2>/dev/null || {
    echo "  WARNING: Failed to power off USB port (may need sudo)"
}
sleep 2
echo "  Done."

# Step 3: Power on the USB port
echo "[3/5] Power cycling USB port (on)..."
sudo uhubctl -l "$SDRPLAY_HUB" -p "$SDRPLAY_PORT" -a on 2>/dev/null || {
    echo "  WARNING: Failed to power on USB port"
}
sleep 3
echo "  Done."

# Step 4: Restart SDRplay service
echo "[4/5] Restarting SDRplay API service..."
sudo /bin/launchctl kickstart -kp system/com.sdrplay.service 2>/dev/null || {
    echo "  WARNING: Failed to restart service via launchctl, trying killall..."
    sudo killall sdrplay_apiService 2>/dev/null || true
    sleep 2
}
sleep 3

# Verify service is running
if pgrep -x sdrplay_apiService > /dev/null; then
    NEW_PID=$(pgrep -x sdrplay_apiService)
    echo "  SDRplay service running with PID: $NEW_PID"
else
    echo "  WARNING: SDRplay service not running!"
fi
echo "  Done."

# Step 5: Verify device detection
echo "[5/5] Verifying device detection..."
DETECT_TIMEOUT=10
if command -v gtimeout &> /dev/null; then
    TIMEOUT_CMD="gtimeout"
else
    TIMEOUT_CMD="timeout"
fi

# Try to detect devices with timeout
DEVICES=$($TIMEOUT_CMD $DETECT_TIMEOUT SoapySDRUtil --find 2>&1) || {
    echo "  ERROR: Device enumeration timed out or failed!"
    echo "  You may need to physically unplug and replug the SDRplay device."
    exit 1
}

if echo "$DEVICES" | grep -q "$SDRPLAY_SERIAL"; then
    echo "  SUCCESS: Device $SDRPLAY_SERIAL detected!"
else
    echo "  WARNING: Device $SDRPLAY_SERIAL not found in enumeration."
    echo "  Available devices:"
    echo "$DEVICES" | grep -E "driver|serial|label" | head -20
fi
echo ""

# Step 6: Start WaveCap-SDR
echo "[6/6] Starting WaveCap-SDR..."
cd "$BACKEND_DIR"
source .venv/bin/activate
PYTHONPATH=. python -m wavecapsdr --bind 0.0.0.0 --port 8087 &
WAVECAP_PID=$!
sleep 5

if ps -p $WAVECAP_PID > /dev/null 2>&1; then
    echo "  WaveCap-SDR started with PID: $WAVECAP_PID"
    echo ""
    echo "=== Fix Complete ==="
    echo "Server running at: http://0.0.0.0:8087"
else
    echo "  ERROR: WaveCap-SDR failed to start!"
    exit 1
fi
