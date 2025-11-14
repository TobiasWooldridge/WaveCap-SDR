---
name: stream-validator
description: Validate WebSocket and HTTP stream health for WaveCap-SDR channels. Use when debugging streaming issues, measuring latency or throughput, detecting packet loss, or verifying audio/spectrum delivery.
---

# Stream Validator for WaveCap-SDR

This skill helps validate WebSocket and HTTP streaming performance, measure metrics, and diagnose streaming issues.

## When to Use This Skill

Use this skill when:
- Audio playback is stuttering or cutting out
- Spectrum display not updating smoothly
- WebSocket connections frequently disconnecting
- Need to measure streaming latency or throughput
- Debugging "no data" or buffering issues
- Performance testing under load
- Verifying stream quality after infrastructure changes

## Stream Types in WaveCap-SDR

**Audio Streams:**
- HTTP: `GET /api/v1/stream/channels/{chan_id}.pcm?format=pcm16`
- WebSocket: `ws://server/api/v1/stream/channels/{chan_id}`
- Format: PCM16 (16-bit signed) or F32 (32-bit float)
- Rate: 48 kHz default (configurable)

**Spectrum Streams:**
- WebSocket: `ws://server/api/v1/stream/spectrum/{capture_id}`
- Format: JSON messages with FFT bins
- Rate: Configurable FPS (typically 10-15)

**IQ Streams:**
- WebSocket: `ws://server/api/v1/stream/captures/{capture_id}.iq`
- Format: Complex IQ samples
- Rate: Full SDR sample rate (2+ MHz)

## Usage Instructions

### Step 1: Identify Stream to Validate

Determine the stream endpoint:

```bash
# List captures
curl http://127.0.0.1:8087/api/v1/captures | jq

# List channels for a capture
curl http://127.0.0.1:8087/api/v1/captures/{capture_id} | jq '.channels'
```

### Step 2: Run Stream Validator

Validate an audio stream:

```bash
PYTHONPATH=backend backend/.venv/bin/python .claude/skills/stream-validator/validate_stream.py \
  --type audio \
  --channel ch1 \
  --duration 10 \
  --port 8087
```

Validate a spectrum stream:

```bash
PYTHONPATH=backend backend/.venv/bin/python .claude/skills/stream-validator/validate_stream.py \
  --type spectrum \
  --capture cap_abc123 \
  --duration 10 \
  --port 8087
```

Parameters:
- `--type`: Stream type (audio, spectrum, iq)
- `--channel`: Channel ID for audio streams
- `--capture`: Capture ID for spectrum/IQ streams
- `--duration`: Seconds to monitor (default: 10)
- `--host`: Server host (default: 127.0.0.1)
- `--port`: Server port (default: 8087)
- `--report`: Generate detailed report

### Step 3: Interpret Results

The validator outputs:

**Connection Metrics:**
- Connection time (time to establish WebSocket/HTTP)
- Connection success rate
- Reconnection attempts

**Data Metrics:**
- Throughput (bytes/sec, samples/sec, messages/sec)
- Latency (time from server to client)
- Packet loss or gaps
- Buffer underruns

**Quality Metrics:**
- Audio: RMS level, silence detection, clipping
- Spectrum: Update rate, FFT bin count
- IQ: Sample continuity, overflow detection

**Health Status:**
- HEALTHY: Stream working correctly
- DEGRADED: Issues detected but stream usable
- UNHEALTHY: Critical problems, stream unusable

## Common Issues and Solutions

### Issue: Low Throughput

**Expected throughput:**
- Audio PCM16: 96 KB/s (48 kHz × 2 bytes)
- Spectrum: 1-10 KB/s (depends on FFT size and FPS)
- IQ: 4-16 MB/s (depends on sample rate)

**Solutions:**
- Check network bandwidth
- Reduce stream quality (lower sample rate, smaller FFT)
- Check server CPU usage
- Verify no other bandwidth-intensive processes

### Issue: High Latency

**Expected latency:**
- Local network: <10 ms
- WiFi: 10-50 ms
- Remote: Varies by distance

**Solutions:**
- Use wired connection instead of WiFi
- Reduce buffering (if configurable)
- Check server processing time
- Verify network not congested

### Issue: Packet Loss

**Symptoms:**
- Audio glitches or pops
- Spectrum gaps or freezes
- Counters show dropped packets

**Solutions:**
- Check WiFi signal strength
- Reduce stream bandwidth
- Verify no network congestion
- Check for server overload

## Files in This Skill

- `SKILL.md`: This file - instructions
- `validate_stream.py`: Stream validation and metrics script
