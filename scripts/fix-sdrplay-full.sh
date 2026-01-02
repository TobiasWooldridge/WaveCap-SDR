#!/bin/bash
# Full SDRplay service reset script
# Run with: sudo /Users/thw/Projects/WaveCap-SDR/scripts/fix-sdrplay-full.sh
# Add to sudoers for passwordless: see bottom of script

set -e

echo "=== SDRplay Full Reset ==="

# 1. Kill all SDRplay service instances
echo "[1/5] Killing SDRplay service..."
killall -9 sdrplay_apiService 2>/dev/null || true
sleep 1

# 2. Power cycle SDRplay USB ports
echo "[2/5] Power cycling SDRplay USB ports..."
if command -v /opt/homebrew/bin/uhubctl &> /dev/null; then
    /opt/homebrew/bin/uhubctl -l 2-1.1.4 -p 3,4 -a off 2>/dev/null || true
    sleep 3
    /opt/homebrew/bin/uhubctl -l 2-1.1.4 -p 3,4 -a on 2>/dev/null || true
    sleep 3
else
    echo "    uhubctl not found, skipping USB power cycle"
fi

# 3. Start SDRplay service (single instance)
echo "[3/5] Starting SDRplay service..."
/Library/SDRplayAPI/3.15.1/bin/sdrplay_apiService &
disown
sleep 2

# 4. Verify service running
echo "[4/5] Verifying service..."
if pgrep -x sdrplay_apiService > /dev/null; then
    echo "    Service running: PID $(pgrep -x sdrplay_apiService | head -1)"
else
    echo "    ERROR: Service not running!"
    exit 1
fi

# 5. Test enumeration (5s timeout)
echo "[5/5] Testing enumeration..."
/opt/homebrew/bin/SoapySDRUtil --find=sdrplay 2>&1 &
SOAPY_PID=$!
sleep 5
if kill -0 $SOAPY_PID 2>/dev/null; then
    echo "    WARNING: Enumeration timed out"
    kill -9 $SOAPY_PID 2>/dev/null
    exit 1
fi
wait $SOAPY_PID 2>/dev/null || true
echo "    Enumeration completed"

echo ""
echo "=== SDRplay Reset Complete ==="

# To add passwordless sudo, run:
# echo 'thw ALL=(ALL) NOPASSWD: /Users/thw/Projects/WaveCap-SDR/scripts/fix-sdrplay-full.sh' | sudo tee /etc/sudoers.d/fix-sdrplay
