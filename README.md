# WaveCap‑SDR

A standalone server that encapsulates SDR device control, capture, and demodulation for the WaveCap ecosystem. It exposes a simple network API so other services can list devices, start/stop tuned captures, create multiple demod channels from a single device stream, stream IQ or audio, and record to disk.

- Spec: see `SPEC.md` (authoritative) for scope, APIs, and milestones.
- Contribution & workflow: see `AGENTS.md` for coding principles, testing expectations, and repo conventions.

## Status
**Alpha** — Core functionality implemented. Device enumeration, IQ streaming, WBFM demodulation, and web UI are working. Tested with RTL-SDR Blog V4 and SDRplay RSPdx-R2 via SoapySDR.

### Recent Updates (2025-10-26)
- Added web UI with catalog page and embedded audio players
- Multi-format audio streaming (PCM16 and F32)
- HTTP streaming endpoint for VLC and browser compatibility
- FM demodulation performance optimization (14x speedup with scipy)
- Multi-device support with simultaneous RTL-SDR and SDRplay operation

## Getting Started

### Prerequisites
- Python 3.9+ (tested with 3.13, 3.14)
- SoapySDR with device modules installed (system-level)
  - For RTL-SDR: `SoapyRTLSDR` module
  - For SDRplay: `SoapySDRPlay3` module and SDRplay API
- System packages: `python3-soapysdr` or equivalent
- Node.js 20+ (optional, for rebuilding the frontend)

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

2. **Set up Python environment**:
```bash
cd backend
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --upgrade pip
pip install fastapi uvicorn httpx websockets pyyaml numpy scipy slowapi
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
source .venv/bin/activate
PYTHONPATH=. python -m wavecapsdr --bind 0.0.0.0 --port 8087
```

Then visit `http://localhost:8087/` for the web UI catalog page, or use the API directly (see `SPEC.md`).

### Building the Frontend (Development)

The frontend is pre-built in `backend/wavecapsdr/static/`. To rebuild after making changes:

```bash
cd frontend
npm ci
npm run build
# Copy built files to backend
cp -r dist/* ../backend/wavecapsdr/static/
```

## Relation to WaveCap
- WaveCap (control/UI) lives in `~/speaker/WaveCap` (also symlinked as `~/speaker/smart-speaker`). WaveCap‑SDR provides the radio server component. Together they form one product; this repo intentionally contains no frontend.

## Repository Layout
- `backend/` — Python server implementation
  - `wavecapsdr/` — main package
    - `devices/` — SDR driver abstractions (soapy, rtl, fake)
    - `dsp/` — signal processing (FM demodulation)
    - `static/` — web UI files (built from frontend/)
    - `api.py` — FastAPI REST/WebSocket endpoints
    - `app.py` — FastAPI application and static file serving
    - `capture.py` — capture and channel management
    - `harness.py` — test harness
  - `tests/` — pytest test suite
  - `config/` — configuration files and presets
- `frontend/` — React/TypeScript web UI
  - `src/` — source code (components, hooks, types)
  - `dist/` — build output (copied to backend/wavecapsdr/static/)
- `docs/` — documentation
  - `configuration.md` — runtime options (planned)
  - `troubleshooting.md` — common issues and solutions
- `scripts/` — helper scripts
  - `soapy-*.sh` — SoapySDR utilities with timeout wrappers
  - `harness-*.sh` — test harness convenience wrappers
  - `run-with-timeout.sh` — generic command timeout wrapper

## Documentation

- **API Reference**: See `SPEC.md`
- **Development Guidelines**: See `AGENTS.md`
- **Troubleshooting**: See `docs/troubleshooting.md`

## License
TBD
