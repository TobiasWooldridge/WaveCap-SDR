# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WaveCap-SDR is a standalone SDR (Software Defined Radio) server that provides device control, RF capture, and demodulation via a REST/WebSocket API. It supports RTL-SDR, SDRplay, and other SoapySDR-compatible devices.

## Build & Run Commands

```bash
# Run the server (from project root)
./start-app.sh
# Or with custom settings:
HOST=0.0.0.0 PORT=8087 ./start-app.sh

# Manual start (from backend/)
cd backend
source .venv/bin/activate
PYTHONPATH=. python -m wavecapsdr --bind 0.0.0.0 --port 8087

# Run tests
cd backend && source .venv/bin/activate
PYTHONPATH=. pytest tests/

# Run a single test
PYTHONPATH=. pytest tests/test_captures_channels.py::test_channel_audio_stream -v

# Type checking (frontend)
cd frontend && npm run type-check

# Build frontend
cd frontend && npm run build
cp -r dist/* ../backend/wavecapsdr/static/

# Verify SDR device detection
scripts/soapy-find.sh
# Or directly:
SoapySDRUtil --find
```

## Architecture

### Backend (Python/FastAPI)

```
backend/wavecapsdr/
├── app.py          # FastAPI app factory, startup, static file serving
├── api.py          # REST/WebSocket endpoints (/api/v1/*)
├── capture.py      # Core: Capture, Channel, CaptureManager classes
├── config.py       # YAML config loading (wavecapsdr.yaml)
├── state.py        # AppState - runtime state container
├── models.py       # Pydantic models for API serialization
├── devices/        # SDR driver abstractions
│   ├── soapy.py    # SoapySDR driver (primary)
│   └── fake.py     # Test/mock driver
└── dsp/            # Signal processing
    ├── fm.py       # WBFM/NBFM demodulation
    ├── am.py       # AM/SSB demodulation
    ├── filters.py  # FIR/IIR filter design
    ├── agc.py      # Automatic gain control
    └── rds.py      # RDS decoder (FM broadcast)
```

**Key Data Flow:**
1. `Capture` owns an SDR device and runs IQ sampling in a dedicated thread
2. IQ samples are broadcast to `Channel` objects for demodulation
3. `Channel` performs DSP (freq shift → demod → filter → AGC) and broadcasts audio
4. Audio/IQ/FFT data is pushed to WebSocket subscribers via asyncio queues

**Threading Model:**
- Each `Capture` has a dedicated thread for device I/O (`_run_thread`)
- DSP processing runs in the capture thread to avoid blocking the event loop
- Only queue operations are scheduled on the asyncio event loop
- `_health_monitor` thread watches for device failures and IQ watchdog

### Frontend (React/TypeScript)

```
frontend/src/
├── components/     # React components (CaptureCard, SpectrumAnalyzer, etc.)
├── hooks/          # Custom hooks (useCaptures, useChannels, useSpectrumData)
├── services/       # API client utilities
└── types/          # TypeScript interfaces matching backend models
```

### Configuration

Configuration lives in `backend/config/wavecapsdr.yaml`:
- `presets`: Named RF configurations (center_hz, sample_rate, gain, etc.)
- `recipes`: Templates for common setups (Marine VHF, FM Broadcast, etc.)
- `captures`: Auto-start captures on server launch
- `device_names`: Human-readable names for device IDs

## API Structure

REST endpoints under `/api/v1/`:
- `GET/POST /captures` - List/create captures
- `POST /captures/{id}/start|stop` - Control capture lifecycle
- `GET/POST /captures/{id}/channels` - List/create channels
- `PATCH /channels/{id}` - Update channel settings (offset, squelch, filters)

WebSocket streams:
- `/api/v1/stream/captures/{id}/iq` - Raw IQ samples (int16 interleaved)
- `/api/v1/stream/captures/{id}/fft` - Spectrum data (JSON)
- `/api/v1/stream/channels/{id}` - Demodulated audio (PCM16/F32/MP3/Opus)

## Testing

Tests use the `fake` driver to simulate SDR hardware:
```python
cfg = AppConfig()
cfg.device = DeviceConfig(driver="fake")
app = create_app(cfg)
client = TestClient(app)
```

## SDRplay Service Recovery

The SDRplay API service can become stuck, causing captures to hang in "starting" state. Symptoms:
- `SoapySDRUtil --find` hangs indefinitely
- Spectrum analyzer shows "starting" but never updates
- SDRplay device not detected even though USB is connected

### Automatic Recovery (Proactive)

WaveCap-SDR has proactive SDRplay health monitoring that automatically:
1. Detects stuck state during device enumeration (2 consecutive timeouts trigger recovery)
2. Attempts service restart before retrying enumeration
3. Runs pre-flight health checks before opening SDRplay devices

Check SDRplay health via API:
```bash
curl http://localhost:8087/api/v1/devices/sdrplay/health
```

### Manual Recovery (API)

Restart the service via API:
```bash
curl -X POST http://localhost:8087/api/v1/devices/sdrplay/restart-service
```

### Manual Recovery (CLI)

To restart the service (macOS, configured in sudoers for passwordless operation):
```bash
sudo /bin/launchctl kickstart -kp system/com.sdrplay.service
```

The fix script with full diagnostics: `.claude/skills/sdrplay-service-fix/fix_sdrplay.sh`

### Architecture Notes

- Recovery is enabled during enumeration (safe - no active streams)
- Recovery is **disabled** during streaming (avoids thrashing)
- Rate limited: max 5 restarts per hour with 60-second cooldown
- Health counters reset on successful restart

## Claude Code Skills

Located in `.claude/skills/`:
- `capture-health-check`: **E2E verification** - check captures, channels, spectrum, audio flow
- `sdrplay-service-fix`: Diagnose and fix stuck SDRplay API service
- `radio-tuner`: Adjust SDR settings (frequency, gain, squelch, filters)
- `audio-quality-checker`: Analyze audio stream quality
- `spectrum-analyzer-debug`: Debug spectrum/waterfall display issues
- `device-prober`: Test SDR hardware capabilities
- `log-viewer`: View and analyze server logs

### Quick Health Check

```bash
.claude/skills/capture-health-check/check_health.sh
```

This verifies:
- Server responding
- SDR devices detected
- Captures running (not stuck in "starting")
- Channels receiving audio (RSSI and audio levels)
- Spectrum data flowing
