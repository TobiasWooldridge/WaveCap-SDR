# WaveCap‑SDR — Product Specification (Authoritative)

Status: Initial draft (updated for multi-channel per device)

## Overview
WaveCap‑SDR is a standalone server that encapsulates all Software‑Defined Radio (SDR) control and signal processing for the WaveCap ecosystem. It exposes a network API for device discovery, tuning, capture, and demodulation so other services (e.g., WaveCap in `~/speaker/WaveCap`) can consume radio streams or recorded artifacts without bundling radio drivers or DSP logic.

Primary goals:
- Provide a stable API to manage SDR hardware and capture sessions.
- Offer demodulation pipelines (e.g., FM/AM/SSB) with typed configuration.
- Enable streaming of IQ and/or demodulated audio over the network.
- Record to disk with predictable naming and metadata.
- Isolate hardware/driver dependencies behind a narrow interface.

Non‑goals:
- Building end‑user UI in this repo.
- Long‑term storage/archival management beyond local recordings.
- Complex multi‑tenant auth; default is local‑only, opt‑in token auth.

## Abstraction & Components
- Device Layer: thin abstraction over SDR drivers (SoapySDR) to enumerate devices, query capabilities, and produce a continuous IQ stream for a configured tuner.
- Capture: a configured device stream (center frequency + sample rate) representing one active tuner. A capture owns the device.
- Channel(s): one or more demodulation pipelines created within a capture. Each channel frequency‑translates within the capture bandwidth, demodulates (e.g., NBFM/WBFM/AM/SSB), and emits audio. Multiple channels can share a single capture to cover multiple VHF marine channels simultaneously.
- Control Plane: HTTP/JSON API to manage devices, captures, and channels.
- Data Plane: WebSocket (and optional HTTP chunked) streaming for IQ (per capture) and audio (per channel).
- Storage: optional local recording for IQ (per capture) and/or audio (per channel) with sidecar metadata.
- Observability: structured logs and metrics (per device, capture, and channel).

Terminology
- Capture: owns the physical SDR device and defines RF center and sample rate.
- Channel: a demod chain within a capture, defined by a target RF frequency (or offset), mode, and squelch.

## API Surface (draft)
Base path: `/api/v1`

- GET `/devices`
  - Lists available devices with id, label, capabilities (freq range, gains, sample rates).

Captures (own devices)
- GET `/captures`
  - List active captures.
- POST `/captures`
  - Body: `{ deviceId?, centerHz, sampleRate, gain?, bandwidth?, ppm?, record?: { iq?: { enabled: bool, path?: string } } }`
  - Creates a capture; returns `{ id, deviceId, state: "created", centerHz, sampleRate }`.
- POST `/captures/{id}/start`
  - Starts device streaming; returns `{ id, state: "running" }`.
- POST `/captures/{id}/stop`
  - Stops device streaming and finalizes IQ recording.
- GET `/captures/{id}`
  - Returns capture status, config, and stats.
- DELETE `/captures/{id}`
  - Tears down the capture and all channels.
- WS `/stream/captures/{id}/iq`
  - Streams capture IQ frames (e.g., `iq16` or `f32`).

Channels (demod within a capture)
- GET `/captures/{id}/channels`
  - List channels for a capture.
- POST `/captures/{id}/channels`
  - Body: `{ freqHz? , offsetHz?, mode: "nbfm"|"wbfm"|"am"|"ssb", squelchDb?: number, audioRate?: number, record?: { audio?: { enabled: bool, path?: string } } }`
  - Creates a channel; `freqHz` is absolute RF; `offsetHz` is relative to `centerHz`.
- POST `/channels/{chanId}/start`
  - Starts demod pipeline.
- POST `/channels/{chanId}/stop`
  - Stops demod pipeline and finalizes audio recording.
- GET `/channels/{chanId}`
  - Returns channel status, config, and stats.
- DELETE `/channels/{chanId}`
  - Removes the channel.
- WS `/stream/channels/{chanId}`
  - Streams audio frames (e.g., `pcm16`).

Health
- GET `/health`
  - Liveness and readiness report; includes attached devices summary.

Error handling:
- JSON errors: `{ error: { code, message, details? } }` with appropriate HTTP status.
- Common codes: `DEVICE_BUSY`, `INVALID_CONFIG`, `NO_DEVICE`, `UNSUPPORTED_MODE`.

## Data Model (high‑level)
- Device: `{ id, driver, label, freqRange: [minHz,maxHz], sampleRates: number[], gains: string[], antenna?: string[] }`
- Capture: `{ id, deviceId, state, centerHz, sampleRate, stats }`
- Channel: `{ id, captureId, mode, targetHz|offsetHz, state, audioRate, squelchDb?, stats }`
- Stats: `{ startedAt, bytesOut, framesOut, overruns, underruns, dropped, snr?, squelch? }`

## Deployment & Running
WaveCap-SDR provides convenient startup scripts for all platforms:

- **Linux/macOS**: `./start-app.sh` from repository root
- **Windows**: `.\start-app.ps1` from repository root (PowerShell)

These scripts automatically:
- Create and configure Python virtual environment with system SoapySDR integration
- Install required dependencies (FastAPI, uvicorn, httpx, websockets, pyyaml, numpy, scipy)
- Start the server with sensible defaults

Behavior on startup:
- If `device.driver=soapy` and no SoapySDR devices are detected at runtime, the server falls back to the built‑in Fake driver so the app remains usable without hardware.
- If no captures are configured in `config/wavecapsdr.yaml`, the server initializes a single default capture (using the first preset if present) but does not auto‑start it. This ensures the UI/API display a capture immediately while avoiding hardware hangs.
- Capture definitions are materialized even when the requested SDR cannot be opened (USB permissions, unplugged hardware, etc.). Such captures surface with `state=failed` and automatically retry acquisition so configs never vanish simply because the radio was offline during boot.
- The fallback/default capture seeds channel entries from the preset offsets so the UI always shows at least one channel to start/stop even before any manual configuration.
- When a capture transitions to `running`, any existing channels auto-start so streaming clients can immediately subscribe. Channels created while a capture is running also auto-start.

Configuration via environment variables:
- `HOST`: bind address (default: `0.0.0.0`)
- `PORT`: port number (default: `8087`)
- `DRIVER`: SDR driver backend (default: `soapy`)
- `DEVICE_ARGS`: specific device selection (optional, e.g., `"driver=rtlsdr,serial=00000001"`)
- `CONFIG`: path to YAML configuration file (optional)

## Configuration
- File: `config/wavecapsdr.yaml` (or environment variables). Document all options in `docs/configuration.md`.
- Examples:
  - `bindAddress` (default `127.0.0.1`), `port` (default `8087`).
  - `auth.token` for simple bearer auth (optional, disabled by default).
  - `recording.rootDir` and filename pattern.
  - `limits.maxConcurrentCaptures`, `limits.maxChannelsPerCapture`, `limits.maxSampleRate`.
- Each successful save creates/refreshes a sibling `wavecapsdr.yaml.bak` so operators can recover if a UI action overwrote capture configs unexpectedly.

## Security
- Default bind is loopback only. Provide opt‑in network exposure with explicit config.
- Optional bearer token for control and streaming endpoints.
- Reject unknown/unsafe driver parameters; validate all user inputs.

## Observability
- Structured logs with per‑session correlation id.
- Metrics counters/gauges: device attach/detach, session states, overruns, throughput.

## Performance Targets (initial)
- Single device at 2–3 MS/s with low overrun rate on modest hardware.
- Streaming end‑to‑end latency under 200 ms for audio pipelines when feasible.

## Concurrency & Resource Rules
- A device may be owned by at most one active capture (one tuner) at a time.
- A capture can host multiple channels as long as each channel’s target frequency lies within the capture passband and resource limits are respected.
- Enforce backpressure on streaming clients; drop oldest frames if needed and report drops.

## Milestones
1. MVP: device list, single session start/stop, IQ streaming over WS, local recording.
2. Add WBFM demod + audio streaming (PCM16).
3. Add basic auth token and configuration file.
4. Add metrics and structured logging.
5. Add additional demod modes and fixtures for testing without hardware.

## Out of Scope (for now)
- Multi‑node clustering and remote device brokers.
- Advanced digital decoders beyond core analog modes.

## Open Questions
- Which driver stack(s) to support first (e.g., RTL‑SDR, SoapySDR)? RESOLVED: SoapySDR primary, targeting RTL‑SDR and SDRplay RSPdx‑r2 via their Soapy modules.
- Preferred on‑disk container for IQ (raw vs. WAV container with metadata)?
- Transport framing for IQ over WS (binary frames chunked by N samples vs. length‑prefixed)?
- Channel scheduler behavior when requested targets drift outside passband (auto retune vs. error).

## Integration with WaveCap
- WaveCap (app/control plane) lives in `~/speaker/WaveCap` (symlinked as `~/speaker/smart-speaker`). WaveCap‑SDR exposes radio functionality so WaveCap can orchestrate higher‑level workflows (e.g., configuring multiple marine channels, recording policies). Keep integration docs and example flows in WaveCap and reference this spec for the SDR API.

## Hardware Support (initial)
- RTL-SDR (via Soapy RTL driver) — enumerate, tune, stream IQ at common rates (250 kS/s–2.4 MS/s).
- SDRplay RSPdx-r2 (via SoapySDRPlay3) — enumerate, tune, stream IQ; demod pipelines operate on the IQ stream.
