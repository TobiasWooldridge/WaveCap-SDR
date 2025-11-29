#!/bin/bash
# WaveCap-SDR E2E Health Check Script
# Usage: ./check_health.sh [--fix]

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

API_BASE="${API_BASE:-http://localhost:8087}"
FIX_MODE=false

if [ "$1" = "--fix" ]; then
    FIX_MODE=true
fi

echo "=== WaveCap-SDR E2E Health Check ==="
echo "API: $API_BASE"
echo

# 1. Check server is responding
echo -e "${GREEN}1. Server Status${NC}"
if curl -s "$API_BASE/api/v1/health" > /dev/null 2>&1; then
    echo "   ✓ Server responding"
else
    echo -e "   ${RED}✗ Server not responding${NC}"
    exit 1
fi

# 2. Check devices
echo
echo -e "${GREEN}2. SDR Devices${NC}"
curl -s "$API_BASE/api/v1/devices" | python3 -c "
import sys, json
devs = json.load(sys.stdin)
for d in devs:
    print(f\"   ✓ {d['driver']}: {d['label'][:50]}\")
if not devs:
    print('   ✗ No devices found')
"

# 3. Check captures
echo
echo -e "${GREEN}3. Captures${NC}"
STUCK_CAPTURES=$(curl -s "$API_BASE/api/v1/captures" | python3 -c "
import sys, json
stuck = []
for c in json.load(sys.stdin):
    if c['state'] == 'running':
        status = '✓'
    elif c['state'] == 'starting':
        status = '⧗'
        stuck.append(c['id'])
    else:
        status = '○'
    antenna = c.get('antenna') or 'N/A'
    print(f\"   {status} {c['id']}: {c['state']} (antenna: {antenna})\")
# Output stuck capture IDs for parsing
import sys
print('STUCK:' + ','.join(stuck), file=sys.stderr)
" 2>&1)

echo "$STUCK_CAPTURES" | grep -v "^STUCK:"
STUCK_IDS=$(echo "$STUCK_CAPTURES" | grep "^STUCK:" | sed 's/STUCK://')

if [ -n "$STUCK_IDS" ] && [ "$STUCK_IDS" != "" ]; then
    echo
    echo -e "   ${YELLOW}WARNING: Captures stuck in 'starting': $STUCK_IDS${NC}"

    if [ "$FIX_MODE" = true ]; then
        echo "   Attempting to fix..."
        # Try restarting SDRplay service first
        if sudo -n /bin/launchctl kickstart -kp system/com.sdrplay.service 2>/dev/null; then
            echo "   ✓ SDRplay service restarted"
            sleep 2

            # Try restarting stuck captures
            for cid in $(echo "$STUCK_IDS" | tr ',' ' '); do
                echo "   Restarting $cid..."
                curl -s -X POST "$API_BASE/api/v1/captures/$cid/stop" > /dev/null
                sleep 1
                curl -s -X POST "$API_BASE/api/v1/captures/$cid/start" > /dev/null
            done

            sleep 2
            echo "   Rechecking..."
            curl -s "$API_BASE/api/v1/captures" | python3 -c "
import sys, json
for c in json.load(sys.stdin):
    if c['state'] == 'running':
        status = '✓'
    else:
        status = '✗'
    print(f\"   {status} {c['id']}: {c['state']}\")
"
        else
            echo -e "   ${RED}Could not restart SDRplay service (no sudo access)${NC}"
        fi
    fi
fi

# 4. Check channels (fetch from each capture)
echo
echo -e "${GREEN}4. Channels${NC}"
curl -s "$API_BASE/api/v1/captures" | python3 -c "
import sys, json, urllib.request

captures = json.load(sys.stdin)
api_base = '$API_BASE'

for cap in captures:
    cid = cap['id']
    try:
        with urllib.request.urlopen(f'{api_base}/api/v1/captures/{cid}/channels') as resp:
            channels = json.load(resp)
            for ch in channels:
                rssi = ch.get('rssiDb')
                audio = ch.get('audioRmsDb')
                if ch['state'] != 'running':
                    status = '○'
                    note = 'not running'
                elif cap['state'] != 'running':
                    status = '?'
                    note = 'capture not running'
                elif rssi is None:
                    status = '?'
                    note = 'no RSSI'
                elif audio is None or audio < -80:
                    status = '~'
                    note = f'weak audio (RSSI={rssi:.0f}dB)'
                else:
                    status = '✓'
                    note = f'RSSI={rssi:.0f}dB, audio={audio:.0f}dB'
                print(f\"   {status} {ch['id']} ({cid}): {note}\")
    except Exception as e:
        print(f\"   ? {cid}: error fetching channels\")
"

# 5. Check spectrum data (sample first running capture)
echo
echo -e "${GREEN}5. Spectrum Data${NC}"
FIRST_RUNNING=$(curl -s "$API_BASE/api/v1/captures" | python3 -c "
import sys, json
for c in json.load(sys.stdin):
    if c['state'] == 'running':
        print(c['id'])
        break
")

if [ -n "$FIRST_RUNNING" ]; then
    curl -s "$API_BASE/api/v1/captures/$FIRST_RUNNING/spectrum/snapshot" | python3 -c "
import sys, json
cid = '$FIRST_RUNNING'
try:
    data = json.load(sys.stdin)
    power = data.get('power', [])
    if power and len(power) > 0:
        print(f'   ✓ Capture {cid}: {len(power)} bins, {min(power):.0f} to {max(power):.0f} dB')
    else:
        print(f'   ~ Capture {cid}: No spectrum data (normal if UI not viewing)')
except Exception as e:
    print(f'   ✗ Capture {cid}: Error - {e}')
" 2>/dev/null || echo "   ? Could not fetch spectrum"
else
    echo "   - No running captures to check"
fi

echo
echo "=== Check Complete ==="

# Exit with error if any captures stuck
if [ -n "$STUCK_IDS" ] && [ "$STUCK_IDS" != "" ]; then
    exit 1
fi
