# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WaveCap-SDR is a standalone SDR (Software Defined Radio) server providing device control, RF capture, and demodulation via REST/WebSocket API. Supports RTL-SDR, SDRplay, and SoapySDR-compatible devices.

## Build & Run Commands

```bash
# Run the server (from project root)
./start-app.sh
HOST=0.0.0.0 PORT=8087 ./start-app.sh

# Manual start
cd backend && source .venv/bin/activate
PYTHONPATH=. python -m wavecapsdr --bind 0.0.0.0 --port 8087

# Run all tests
cd backend && source .venv/bin/activate
PYTHONPATH=. pytest tests/

# Run a single test
PYTHONPATH=. pytest tests/test_captures_channels.py::test_channel_audio_stream -v

# Type checking
cd frontend && npm run type-check
cd backend && source .venv/bin/activate && mypy wavecapsdr

# Build frontend
cd frontend && npm run build
cp -r dist/* ../backend/wavecapsdr/static/

# Verify SDR device detection
scripts/soapy-find.sh
```

## Architecture

### Backend (Python/FastAPI)

```
backend/wavecapsdr/
├── app.py           # FastAPI app factory, startup, static file serving
├── api.py           # REST/WebSocket endpoints (/api/v1/*)
├── capture.py       # Core: Capture, Channel, CaptureManager classes
├── config.py        # YAML config loading (wavecapsdr.yaml)
├── state.py         # AppState - runtime state container
├── models.py        # Pydantic models for API serialization
├── scanner.py       # Frequency scanner (sequential/priority/activity modes)
├── sdrplay_recovery.py  # SDRplay service health monitoring and recovery
├── devices/         # SDR driver abstractions
│   ├── soapy.py     # SoapySDR driver (primary)
│   ├── sdrplay_proxy.py  # Subprocess isolation for SDRplay multi-device
│   └── fake.py      # Test/mock driver
└── dsp/             # Signal processing
    ├── fm.py        # WBFM/NBFM demodulation
    ├── am.py        # AM/SSB demodulation
    ├── filters.py   # FIR/IIR filter design
    ├── agc.py       # Automatic gain control
    └── rds.py       # RDS decoder (FM broadcast)
```

### Key Data Flow

1. `Capture` owns an SDR device and runs IQ sampling in a dedicated thread
2. IQ samples are broadcast to `Channel` objects for demodulation
3. `Channel` performs DSP (freq shift → demod → filter → AGC) and broadcasts audio
4. Audio/IQ/FFT data is pushed to WebSocket subscribers via asyncio queues

### Threading Model

- Each `Capture` has a dedicated thread for device I/O (`_run_thread`)
- DSP processing runs in the capture thread to avoid blocking the event loop
- Only queue operations are scheduled on the asyncio event loop
- `_health_monitor` thread watches for device failures and IQ watchdog

### Frontend (React/TypeScript)

```
frontend/src/
├── App.tsx          # Main application component
├── components/      # React components (CaptureCard, SpectrumAnalyzer, etc.)
├── hooks/           # Custom hooks (useCaptures, useChannels, useSpectrumData)
├── services/        # API client utilities
└── types/           # TypeScript interfaces matching backend models
```

### Configuration

Located in `backend/config/wavecapsdr.yaml`:
- `presets`: Named RF configurations (center_hz, sample_rate, gain, etc.)
- `recipes`: Templates for common setups (Marine VHF, FM Broadcast, etc.)
- `captures`: Auto-start captures on server launch
- `device_names`: Human-readable names for device IDs

## API Structure

REST endpoints under `/api/v1/`:
- `GET/POST /captures` - List/create captures
- `POST /captures/{id}/start|stop` - Control capture lifecycle
- `GET/POST /captures/{id}/channels` - List/create channels
- `PATCH /channels/{id}` - Update channel settings

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

Hardware tests are marked with `@pytest.mark.hardware` and skipped by default.

## SDRplay Service Recovery

The SDRplay API service can become stuck, causing captures to hang in "starting" state.

**Symptoms:**
- `SoapySDRUtil --find` hangs indefinitely
- Spectrum analyzer shows "starting" but never updates
- SDRplay device not detected

**Automatic Recovery:**
WaveCap-SDR has proactive health monitoring that detects stuck states and attempts service restart.

**Manual Recovery:**
```bash
# Via API
curl -X POST http://localhost:8087/api/v1/devices/sdrplay/restart-service

# Via CLI (macOS)
sudo /bin/launchctl kickstart -kp system/com.sdrplay.service
```

## Claude Code Skills

Located in `.claude/skills/`:
- `capture-health-check`: E2E verification - check captures, channels, spectrum, audio flow
- `sdrplay-service-fix`: Diagnose and fix stuck SDRplay API service
- `radio-tuner`: Adjust SDR settings (frequency, gain, squelch, filters)
- `audio-quality-checker`: Analyze audio stream quality
- `spectrum-analyzer-debug`: Debug spectrum/waterfall display issues
- `device-prober`: Test SDR hardware capabilities
- `log-viewer`: View and analyze server logs

Quick health check: `.claude/skills/capture-health-check/check_health.sh`
