# WaveCap‑SDR

A standalone SDR server for the WaveCap ecosystem. It exposes a network API and bundled web UI so other services (and humans) can list devices, start/stop tuned captures, create multiple demod channels from a single device stream, stream IQ or audio, and record to disk.

- Spec: see `docs/api-spec.md` (authoritative) for scope, APIs, and milestones.
- Contribution & workflow: see `AGENTS.md` for coding principles, testing expectations, and repo conventions.
- SA-GRN P25 Phase 1 spec: `docs/sa-grn-p25-spec.md`.

## Status
Active development. Core functionality is implemented and tested with RTL-SDR Blog V4 and SDRplay RSPdx-R2 via SoapySDR.

## Capabilities
- Device enumeration and capture lifecycle (multi-device)
- Multi-channel demodulation (FM/AM/SSB) with AGC, filters, and squelch
- Digital voice modes: P25, DMR, NXDN, D-Star, YSF (various levels of support)
- P25 Phase 1 trunking system support with voice following and recording
- Spectrum/FFT display and scanner modes (sequential, priority, activity)
- Streaming: IQ (int16/f32), PCM audio, MP3/Opus/AAC via ffmpeg, WebSocket and HTTP
- RDS decoding (FM broadcast) and POCSAG paging decoder
- Web UI for monitoring, tuning, and channel management

## Getting Started

### Prerequisites
- Python 3.9+ (project supports 3.9 to 3.14)
- SoapySDR with device modules installed (system-level)
  - For RTL-SDR: `SoapyRTLSDR` module (or use built-in direct RTL driver)
  - For SDRplay: `SoapySDRPlay3` module and SDRplay API (custom fork recommended - see below)
- System packages: `python3-soapysdr` or equivalent
- Node.js 20+ (required by `./start-app.sh`, which builds the frontend)
- ffmpeg (required for MP3/Opus/AAC audio streaming)

**SDRplay Users:** This project requires a custom SoapySDRPlay3 driver with multi-device serialization fixes:
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

### Quick Start

1. **Install system dependencies**:

   **Ubuntu/Debian:**
   ```bash
   sudo apt-get install python3-soapysdr soapysdr-tools
   # For RTL-SDR:
   sudo apt-get install soapysdr-module-rtlsdr
   # For SDRplay (requires separate download from SDRplay.com):
   # Install SDRplay API .deb, then soapysdr-module-sdrplay3
   ```

   **macOS (Homebrew):**
   ```bash
   brew install soapysdr node
   # For RTL-SDR:
   brew install librtlsdr soapyrtlsdr
   # For SDRplay: Install SDRplay API from SDRplay.com, then:
   # brew install soapysdrplay3
   ```

   > **Note:** Homebrew's SoapySDR includes Python bindings for its bundled Python version.
   > The startup script will use Homebrew's Python automatically if available.

2. **Run the server (recommended)**:
```bash
./start-app.sh
```

3. **Verify device detection**:
```bash
scripts/soapy-find.sh
```

4. **Run test harness** (KEXP 90.3 FM Seattle):
```bash
# Single device (RTL-SDR):
DEVICE_ARGS="driver=rtlsdr" bash scripts/harness-kexp.sh

# Single device (SDRplay):
DEVICE_ARGS="driver=sdrplay" bash scripts/harness-kexp.sh

# All devices simultaneously:
bash scripts/harness-multi-fm.sh
```

Test output WAV files are saved to `backend/harness_out/`.

### Running the Server

#### Quick Start (Recommended)

Use the startup script from the root directory:

```bash
# Linux/macOS
./start-app.sh

# Windows (PowerShell)
.\start-app.ps1
```

The scripts automatically set up the virtual environment, install dependencies, and start the server with sensible defaults (host: 0.0.0.0, port: 8087).

#### Custom Configuration

Use environment variables to customize the server:

```bash
# Custom host/port
HOST=127.0.0.1 PORT=8088 ./start-app.sh

# Specific device
DEVICE_ARGS="driver=rtlsdr" ./start-app.sh

# All options
HOST=0.0.0.0 PORT=8087 DRIVER=soapy CONFIG=backend/config/wavecapsdr.yaml ./start-app.sh
```

For Windows PowerShell:
```powershell
$env:HOST="127.0.0.1"; $env:PORT=8088; .\start-app.ps1
```

#### Manual Start

If you prefer to run the server manually:

```bash
cd backend
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --upgrade pip
pip install fastapi uvicorn httpx websockets pyyaml numpy scipy slowapi
PYTHONPATH=. python -m wavecapsdr --bind 0.0.0.0 --port 8087
```

Then visit `http://localhost:8087/` for the web UI catalog page, or use the API directly (see `docs/api-spec.md`).

### Building the Frontend (Development)

`./start-app.sh` builds the frontend automatically. To rebuild after making changes:

```bash
cd frontend
npm ci
npm run build
# Copy built files to backend
cp -r dist/* ../backend/wavecapsdr/static/
```

## Trunking System Support

WaveCap-SDR includes comprehensive P25 Phase 1 trunking support with automatic voice following:

- **Control Channel Monitoring**: Tracks trunking control channel for voice grants
- **Voice Following**: Automatically tunes voice recorders to follow active calls
- **Talkgroup Filtering**: Priority-based talkgroup selection and filtering
- **Call Recording**: Records voice calls with metadata (talkgroup, timestamp, identifiers)
- **Network Configuration**: Monitors system parameters (WACN, System ID, NAC, sites)
- **Event Tracking**: Call state machine with staleness detection (SDRTrunk-inspired)
- **Duplicate Detection**: Suppresses duplicate voice grants

Supported P25 modulations:
- **C4FM**: Standard P25 Phase 1 (12.5 kHz channel spacing)
- **CQPSK/LSM**: Simulcast-compatible modulation (used by many large metro systems)

Configuration is done via `backend/config/wavecapsdr.yaml` in the `trunking.systems` section. See the `sa_grn` example for a complete configuration.

## Relation to WaveCap
- WaveCap (control/UI) lives in `~/speaker/WaveCap` (also symlinked as `~/speaker/smart-speaker`). WaveCap‑SDR provides the radio server component and a lightweight web UI for local monitoring/tuning. Together they form one product.

## Repository Layout
- `backend/` — Python server implementation
  - `wavecapsdr/` — main package
    - `devices/` — SDR driver abstractions (soapy, rtl, sdrplay_proxy, fake)
    - `decoders/` — digital decoders (P25, DMR, POCSAG, IMBE/AMBE voice, LRRP)
    - `dsp/` — signal processing (FM/AM/SSB demod, AGC, filters, RDS, POCSAG, P25 C4FM/CQPSK)
    - `trunking/` — P25 Phase 1 trunking system (control channel, voice following, recording)
    - `static/` — web UI files (built from frontend/)
    - `api.py` — REST/WebSocket endpoints
    - `app.py` — application setup and static file serving
    - `capture.py` — capture and channel management
    - `scanner.py` — frequency scanner (sequential, priority, activity)
    - `encoders.py` — audio encoders (MP3, Opus, AAC via ffmpeg)
    - `harness.py` — test harness
    - `sdrplay_recovery.py` — SDRplay service health monitoring and recovery
    - `state_broadcaster.py` — WebSocket state updates
    - `error_tracker.py` — error collection for UI visibility
  - `tests/` — pytest test suite (459+ tests)
  - `config/` — configuration files and presets
- `frontend/` — React/TypeScript web UI
  - `src/` — source code (components, hooks, types)
  - `dist/` — build output (copied to backend/wavecapsdr/static/)
- `docs/` — documentation
  - `api-spec.md` — API reference, data models, and milestones
  - `antenna.md` — multi-antenna device configuration
  - `configuration.md` — runtime options
  - `troubleshooting.md` — common issues and solutions
- `scripts/` — helper scripts
  - `soapy-*.sh` — SoapySDR utilities with timeout wrappers
  - `harness-*.sh` — test harness convenience wrappers
  - `run-with-timeout.sh` — generic command timeout wrapper
  - `fix-sdrplay.sh` — SDRplay service recovery script
- `.claude/skills/` — Claude Code skills for development and debugging

## Documentation

- **API Reference**: See `docs/api-spec.md`
- **Antenna Configuration**: See `docs/antenna.md`
- **Configuration**: See `docs/configuration.md`
- **Troubleshooting**: See `docs/troubleshooting.md`
- **Development Guidelines**: See `AGENTS.md`
- **Claude Code Instructions**: See `CLAUDE.md`
- **Trunking Configuration**: See `trunking.systems` section in `backend/config/wavecapsdr.yaml`

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

# Via script
scripts/fix-sdrplay.sh

# Via CLI (Linux/systemd)
sudo systemctl restart sdrplay

# Via CLI (macOS/launchd - requires sudoers configuration)
sudo /bin/launchctl kickstart -kp system/com.sdrplay.service
```

## License
TBD
