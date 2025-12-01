#!/bin/bash
# Monitor radios every minute for 10 minutes

echo "=== MONITORING START: $(date) ==="

for i in 1 2 3 4 5 6 7 8 9 10; do
    echo ""
    echo "=== Check $i/10 at $(date) ==="

    # Check c2 (RSPdx-R2 240305E670 on Antenna C)
    c2_data=$(curl -s http://localhost:8087/api/v1/captures/c2 2>/dev/null)
    if [ -n "$c2_data" ]; then
        c2_state=$(echo "$c2_data" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['state'])" 2>/dev/null)
        c2_antenna=$(echo "$c2_data" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('antenna', 'N/A'))" 2>/dev/null)
        c2_overflow=$(echo "$c2_data" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['iqOverflowCount'])" 2>/dev/null)
        c2_error=$(echo "$c2_data" | python3 -c "import json,sys; d=json.load(sys.stdin); e=d.get('errorMessage'); print(e if e else 'none')" 2>/dev/null)
        echo "c2 (240305E670): state=$c2_state, antenna=$c2_antenna, iqOverflow=$c2_overflow, error=$c2_error"
    else
        echo "c2: Failed to query"
    fi

    # Check channels
    ch_data=$(curl -s http://localhost:8087/api/v1/captures/c2/channels 2>/dev/null)
    if [ -n "$ch_data" ]; then
        echo "$ch_data" | python3 -c "
import json,sys
try:
    chs = json.load(sys.stdin)
    for c in chs:
        rssi = c.get('rssiDb') or 0
        audio = c.get('audioRmsDb') or 0
        snr = c.get('snrDb') or 0
        print(f\"  ch {c['id']}: rssi={rssi:.1f}dB, audio={audio:.1f}dB, snr={snr:.1f}dB, state={c['state']}\")
except: pass
" 2>/dev/null
    fi

    # Wait 60 seconds before next check (except on last iteration)
    if [ $i -lt 10 ]; then
        sleep 60
    fi
done

echo ""
echo "=== MONITORING END: $(date) ==="
