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
├── config.py        # YAML config loading (wavecapsdr.yaml)
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
├── decoders/        # Digital decoders (P25, DMR, POCSAG, IMBE/AMBE voice)
│   ├── p25.py       # P25 Phase 1/2 decoder with trunking
│   ├── p25_tsbk.py  # P25 TSBK (control channel) message parsing
│   ├── p25_frames.py # P25 frame structures and parsing
│   ├── dmr.py       # DMR decoder
│   ├── imbe.py      # IMBE voice codec
│   ├── ambe.py      # AMBE voice codec
│   ├── lrrp.py      # LRRP (GPS location) decoder
│   ├── nac_tracker.py # Network Access Code tracking
│   ├── trunking.py  # Generic trunking protocol abstractions
│   └── voice.py     # Voice decoder abstractions
├── trunking/        # Trunking system configuration and control
│   ├── system.py    # Trunking system orchestration
│   ├── control_channel.py  # Control channel monitoring
│   ├── voice_channel.py    # Voice channel recording
│   ├── manager.py   # Trunking system manager
│   ├── identifiers.py # IdentifierCollection (SDRTrunk-inspired)
│   ├── event_tracker.py # P25EventTracker (call state machine)
│   ├── network_config.py # Network configuration monitoring
│   ├── duplicate_detector.py # Duplicate call detection
│   ├── cc_scanner.py # Control channel scanner
│   └── api.py       # Trunking REST API endpoints
└── dsp/             # Signal processing
    ├── fm.py        # WBFM/NBFM demodulation
    ├── am.py        # AM/SSB demodulation
    ├── sam.py       # Synchronous AM demodulation
    ├── filters.py   # FIR/IIR filter design
    ├── agc.py       # Automatic gain control
    ├── rds.py       # RDS decoder (FM broadcast)
    ├── pocsag.py    # POCSAG paging decoder
    └── fec/         # Forward error correction
        ├── bch.py   # BCH codes
        ├── golay.py # Golay codes
        └── trellis.py # Trellis/Viterbi decoding
```

**Key Data Flow:**
1. `Capture` owns an SDR device and runs IQ sampling in a dedicated thread
2. IQ samples are broadcast to `Channel` objects for demodulation
3. `Channel` performs DSP (freq shift → demod → filter → AGC) and broadcasts audio
4. Audio/IQ/FFT data is pushed to WebSocket subscribers via asyncio queues
5. Encoders provide MP3/Opus/AAC streams when requested

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
- `recovery`: SDRplay and IQ watchdog recovery settings
- `trunking`: Trunking system configuration (see `backend/wavecapsdr/trunking/`)

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

Build and install:
```bash
cd ../SoapySDRPlay3
mkdir -p build && cd build
cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ..
make -j4
sudo make install
```

## SDRplay Service Recovery

The SDRplay API service can become stuck, causing captures to hang in "starting" state.

**Symptoms:**
- `SoapySDRUtil --find` hangs indefinitely
- Spectrum analyzer shows "starting" but never updates
- SDRplay device not detected

**Automatic Recovery:**
WaveCap-SDR has proactive health monitoring that detects stuck states and attempts service restart.

**Manual Recovery (Linux/systemd):**
```bash
# Via API
curl -X POST http://localhost:8087/api/v1/devices/sdrplay/restart-service

# Via CLI
sudo systemctl restart sdrplay
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

## References

- **SDRTrunk** (https://github.com/DSheirer/sdrtrunk) - Java P25 trunking reference
  - Event tracking and channel pool management patterns
  - P25TrafficChannelManager architecture
  - LRRP/GPS location coordinate encoding

## SDRTrunk-Inspired Patterns

Located in `backend/wavecapsdr/trunking/`:

### IdentifierCollection (`identifiers.py`)
Flexible metadata management with immutable/mutable variants:
- `Identifier` - Immutable identifier with value, role, and form
- `IdentifierCollection` - Immutable collection with type-safe queries
- `MutableIdentifierCollection` - Builder pattern for constructing collections
- `TalkerAliasManager` - Cache and enrich identifiers with aliases

### P25EventTracker (`event_tracker.py`)
Call state machine with staleness detection:
- `P25CallEvent` - Complete call metadata with lifecycle tracking
- `P25EventTracker` - Manages single call state with dual-source timing
- `P25EventTrackerManager` - Manages all trackers for a trunking system
- 2-second staleness threshold, duplicate grant detection, talker change detection

### NetworkConfigurationMonitor (`network_config.py`)
System configuration tracking from control channel:
- `FrequencyBand` - Frequency band definitions (IDEN_UP)
- `SiteStatus` - Current site info (RFSS_STS_BCAST)
- `AdjacentSite` - Neighbor site tracking
- `P25NetworkConfigurationMonitor` - Comprehensive system configuration

### DuplicateCallDetector (`duplicate_detector.py`)
Duplicate event suppression:
- `DuplicateCallDetector` - Time-based duplicate detection
- `FrequencyBasedDuplicateDetector` - Frequency allocation tracking
- Prevents rapid-fire duplicate grants from flooding the system
