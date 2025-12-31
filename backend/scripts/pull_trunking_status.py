#!/usr/bin/env python3
import json
import urllib.request
import sys

BASE = 'http://localhost:8087/api/v1/trunking/systems/sa_grn_2'

def fetch(path: str):
    with urllib.request.urlopen(f"{BASE}/{path}") as resp:
        return json.load(resp)

if __name__ == '__main__':
    status = fetch('')
    calls = fetch('calls/active')
    streams = fetch('voice-streams')
    print(json.dumps({
        'state': status.get('state'),
        'control': status.get('controlChannelFreqHz'),
        'crc_pass': status.get('control_monitor', {}).get('tsbk_crc_pass'),
        'crc_rate': status.get('control_monitor', {}).get('tsbk_crc_pass_rate'),
        'frames': status.get('control_monitor', {}).get('frames_decoded'),
        'calls': calls,
        'voice_streams': streams,
    }, indent=2))
