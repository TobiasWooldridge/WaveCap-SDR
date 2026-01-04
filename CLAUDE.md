# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WaveCap-SDR is a standalone SDR (Software Defined Radio) server providing device control, RF capture, demodulation, and streaming via REST/WebSocket APIs, plus a bundled web UI. Supports RTL-SDR, SDRplay, and other SoapySDR-compatible devices, with a direct RTL driver and a fake driver for tests.

## Build & Run Commands

```bash
# Run the server (from project root; builds frontend if present)
./start-app.sh
HOST=0.0.0.0 PORT=8087 ./start-app.sh

# Manual start
cd backend && source .venv/bin/activate
PYTHONPATH=. python -m wavecapsdr --bind 0.0.0.0 --port 8087

# Run all tests (wrap with timeout to avoid hangs)
scripts/run-with-timeout.sh --seconds 120 -- bash -lc 'cd backend && source .venv/bin/activate && PYTHONPATH=. pytest tests/'

# Run a single test
cd backend && source .venv/bin/activate
PYTHONPATH=. pytest tests/test_captures_channels.py::test_channel_audio_stream -v

# Frontend quality checks (no unit tests, only type-check and lint)
cd frontend && npm run type-check
cd frontend && npm run lint

# Backend type checking
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
├── config.py        # YAML config loading (wavecapsdr.yaml + local overrides)
├── encoders.py      # Audio encoders (mp3/opus/aac via ffmpeg)
├── state.py         # AppState - runtime state container
├── models.py        # Pydantic models for API serialization
├── scanner.py       # Frequency scanner (sequential/priority/activity modes)
├── sdrplay_recovery.py  # SDRplay service health monitoring and recovery
├── state_broadcaster.py # Pushes state updates to websocket clients
├── error_tracker.py  # Error collection for UI/API visibility
├── devices/         # SDR driver abstractions
│   ├── soapy.py     # SoapySDR driver (primary)
│   ├── rtl.py       # Direct RTL-SDR driver
│   ├── sdrplay_proxy.py  # Subprocess isolation for SDRplay multi-device
│   └── fake.py      # Test/mock driver
├── decoders/        # Digital decoders (e.g., POCSAG)
├── trunking/        # Trunking system configuration and control
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
5. Encoders provide MP3/Opus/AAC streams when requested

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

Located in `backend/config/wavecapsdr.yaml` with optional overrides in `backend/config/wavecapsdr.local.yaml` (gitignored):
- `presets`: Named RF configurations (center_hz, sample_rate, gain, etc.)
- `recipes`: Templates for common setups (Marine VHF, FM Broadcast, etc.)
- `captures`: Auto-start captures on server launch
- `device_names`: Human-readable names for device IDs
- `recovery`: SDRplay and IQ watchdog recovery settings
- `trunking`: Trunking system configuration (see `backend/wavecapsdr/trunking/`)

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

### Backend (pytest)

459 tests in `backend/tests/` covering DSP, trunking, API, and integration:

```bash
cd backend && source .venv/bin/activate

# Run all tests (with timeout to avoid hangs)
scripts/run-with-timeout.sh --seconds 120 -- bash -lc 'cd backend && source .venv/bin/activate && PYTHONPATH=. pytest tests/'

# Run all tests (direct)
PYTHONPATH=. pytest tests/

# Run unit tests only
PYTHONPATH=. pytest tests/unit/

# Skip hardware tests
PYTHONPATH=. pytest tests/ -m "not hardware"

# Run a single test
PYTHONPATH=. pytest tests/test_captures_channels.py::test_channel_audio_stream -v
```

Tests use the `fake` driver to simulate SDR hardware:
```python
cfg = AppConfig()
cfg.device = DeviceConfig(driver="fake")
app = create_app(cfg)
client = TestClient(app)
```

Hardware tests are marked with `@pytest.mark.hardware` and skipped by default.

### Frontend

No unit test framework is installed. Quality checks:

```bash
cd frontend
npm run type-check  # TypeScript validation
npm run lint        # ESLint
npm run build       # Compile check
```

## SDRplay Dependencies

**Important:** This project requires a custom SoapySDRPlay3 driver with multi-device serialization fixes:
- Repository: https://github.com/TobiasWooldridge/SoapySDRPlay3
- This fork includes API-level locking to prevent crashes on rapid config changes

**SoapySDRPlay3 Enhancements:**
- **Native subprocess proxy** (`SOAPY_SDRPLAY_MULTIDEV=1` or `proxy=true` in device args)
  - Transparent multi-device support at driver level
  - Each device runs in isolated subprocess via fork/exec
  - Shared memory ring buffer for zero-copy IQ transfer
  - Automatic worker health monitoring and restart
- Timeout protection on all blocking API calls (prevents indefinite hangs)
- Health monitoring with `DeviceHealthStatus` enum (Healthy, Warning, Stale, Recovering, ServiceUnresponsive, DeviceRemoved, Failed)
- Watchdog system with configurable `WatchdogConfig` (callback timeouts, auto-recovery)
- Settings cache for recovery (restores frequency, gain, antenna after stream recovery)
- Service health tracking (`isServiceResponsive()`, `getConsecutiveTimeouts()`)
- Recovery controls (`triggerRecovery()`, `restartService()`, `resetUSBDevice()`)
- Health callbacks via `registerHealthCallback()`

**Fork enhancements (TobiasWooldridge/SoapySDR):**
- `Device.make(args, timeoutUs)` - timeout parameter for device open
- `Device.cancelMake()` - cancel pending device open from another thread

Build and install:
```bash
cd ../SoapySDRPlay3
mkdir -p build && cd build
cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DENABLE_SUBPROCESS_MULTIDEV=ON ..
make -j4
sudo make install
```

**Build options:**
- `-DENABLE_SUBPROCESS_MULTIDEV=ON` - Enable native subprocess proxy for multi-device support
- `-DCMAKE_INSTALL_PREFIX=/Users/thw/.local` - Install to user-local prefix

Install recovery scripts (optional, for passwordless service restart):
```bash
cd ../SoapySDRPlay3
sudo scripts/install-recovery-scripts.sh
```

## SDRplay Multi-Device Architecture

WaveCap-SDR supports two proxy modes for multi-device SDRplay operation:

**1. Native Driver Proxy (Preferred)**
- Enabled when SoapySDRPlay3 is built with `ENABLE_SUBPROCESS_MULTIDEV=ON`
- WaveCap-SDR automatically detects and uses native proxy
- Driver spawns worker subprocess transparently
- Uses C++ shared memory ring buffer for IQ transfer
- More robust health monitoring and recovery

**2. Python Subprocess Proxy (Fallback)**
- Used when native proxy is unavailable
- Implemented in `sdrplay_proxy.py` and `sdrplay_worker.py`
- Uses Python multiprocessing + shared memory
- Works with any SoapySDRPlay3 build

The SDRplay API v3.15 only allows ONE device per process, so subprocess isolation is required for multi-device operation. WaveCap-SDR automatically selects the best available proxy mode.

## SDRplay Service Recovery

The SDRplay API service can become stuck, causing captures to hang in "starting" state.

**Symptoms:**
- `SoapySDRUtil --find` hangs indefinitely
- Spectrum analyzer shows "starting" but never updates
- SDRplay device not detected
- Log messages: "SDRplay API lock timed out" or "sdrplay_api_Open() timed out"

**Automatic Recovery:**
WaveCap-SDR integrates with SoapySDRPlay3's health monitoring system:
- Driver-level watchdog detects stale streams (callbacks stop arriving)
- API-level timeout protection prevents indefinite hangs
- Proactive service health checking via `check_service_responsive()`
- Automatic service restart when health degrades (configurable)

**Manual Recovery:**
```bash
# Preferred: Use the sdrplay-service-restart script from SoapySDRPlay3
# (handles SIGHUP soft restart, plist detection, stale lock cleanup)
sudo sdrplay-service-restart --force

# Full reset script (kills service, power-cycles USB, restarts service, verifies enumeration)
# Requires uhubctl for USB power cycling
sudo scripts/fix-sdrplay-full.sh

# Via API
curl -X POST http://localhost:8087/api/v1/devices/sdrplay/restart-service

# Check health status
curl http://localhost:8087/api/v1/devices/sdrplay/health

# Via CLI (Linux/systemd)
sudo systemctl restart sdrplayService

# Via CLI (macOS)
sudo launchctl kickstart -k system/com.sdrplay.service
```

**Passwordless sudo for recovery scripts:**
```bash
# Install recovery scripts with sudoers configuration
cd ../SoapySDRPlay3 && sudo scripts/install-recovery-scripts.sh

# Or manually configure sudoers for the fix script
echo 'thw ALL=(ALL) NOPASSWD: /Users/thw/Projects/WaveCap-SDR/scripts/fix-sdrplay-full.sh' | sudo tee /etc/sudoers.d/fix-sdrplay
```

## Claude Code Skills

Located in `.claude/skills/`:
- `agc-tuner`: Tune AGC response for audio stability
- `audio-quality-checker`: Analyze audio stream quality
- `capture-health-check`: E2E verification (captures, channels, spectrum, audio flow)
- `channel-optimizer`: Optimize per-channel settings (squelch, filters, gain)
- `config-validator`: Validate config file structure and settings
- `device-prober`: Test SDR hardware capabilities
- `dsp-filter-designer`: Design/validate DSP filter settings
- `frequency-lookup`: Resolve frequencies and labels
- `harness-runner`: Run harness scenarios with timeouts
- `log-viewer`: View and analyze server logs
- `radio-tuner`: Adjust SDR settings (frequency, gain, squelch, filters)
- `recipe-builder`: Create/update recipe templates
- `sdrplay-service-fix`: Diagnose and fix stuck SDRplay API service
- `signal-monitor`: Monitor signal levels and drift
- `spectrum-analyzer-debug`: Debug spectrum/waterfall display issues
- `stream-validator`: Validate audio/IQ stream outputs

Quick health check: `.claude/skills/capture-health-check/check_health.sh`
