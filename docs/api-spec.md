# WaveCap‑SDR — Product Specification (Authoritative)

Status: Alpha (updated 2025-12-25)

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

## API Surface
Base path: `/api/v1`

### Devices
- GET `/devices`
  - Lists available devices with id, label, capabilities (freq range, gains, sample rates), nickname, shorthand.
- GET `/devices/{deviceId}/name`
  - Get device nickname and shorthand.
- PATCH `/devices/{deviceId}/name`
  - Set custom nickname for device.

### Recipes
- GET `/recipes`
  - List capture creation recipes (presets for common use cases).

### Frequency Identification
- GET `/frequency/identify?frequency_hz={hz}`
  - Identify a frequency and return suggested name, band, description.

### Captures (own devices)
- GET `/captures`
  - List active captures.
- POST `/captures`
  - Body: `{ deviceId?, centerHz, sampleRate, gain?, bandwidth?, ppm?, antenna?, deviceSettings?, elementGains?, streamFormat?, dcOffsetAuto?, iqBalanceAuto?, createDefaultChannel?, name? }`
  - Creates a capture; returns full capture model.
- GET `/captures/{id}`
  - Returns capture status, config, and stats.
- PATCH `/captures/{id}`
  - Update capture configuration (frequency, gain, bandwidth, etc.). Hot-reconfigures running captures.
- POST `/captures/{id}/start`
  - Starts device streaming and auto-starts all channels.
- POST `/captures/{id}/stop`
  - Stops device streaming and finalizes IQ recording.
- DELETE `/captures/{id}`
  - Tears down the capture and all channels.

### Channels (demod within a capture)
- GET `/captures/{id}/channels`
  - List channels for a capture.
- POST `/captures/{id}/channels`
  - Body: `{ mode, offsetHz?, audioRate?, squelchDb?, name?, notchFrequencies?, ...filterParams }`
  - Creates a channel with configurable DSP parameters.
- GET `/channels/{chanId}`
  - Returns channel status, config, and signal metrics (RSSI, SNR).
- PATCH `/channels/{chanId}`
  - Update channel configuration including all DSP parameters.
- POST `/channels/{chanId}/start`
  - Starts demod pipeline.
- POST `/channels/{chanId}/stop`
  - Stops demod pipeline and finalizes audio recording.
- DELETE `/channels/{chanId}`
  - Removes the channel.

### Scanner (frequency scanning)
- POST `/scanners`
  - Create scanner: `{ captureId, scanList, mode, dwellTimeMs, priorityFrequencies, priorityIntervalS, squelchThresholdDb, lockoutFrequencies, pauseDurationMs }`
- GET `/scanners`
  - List all scanners.
- GET `/scanners/{id}`
  - Get scanner status including current frequency, hits, lockouts.
- PATCH `/scanners/{id}`
  - Update scanner configuration.
- DELETE `/scanners/{id}`
  - Delete scanner.
- POST `/scanners/{id}/start`
  - Start scanning.
- POST `/scanners/{id}/stop`
  - Stop scanning.
- POST `/scanners/{id}/pause`
  - Pause scanning.
- POST `/scanners/{id}/resume`
  - Resume from pause.
- POST `/scanners/{id}/lock`
  - Lock on current frequency.
- POST `/scanners/{id}/unlock`
  - Unlock and resume scanning.
- POST `/scanners/{id}/lockout`
  - Add current frequency to lockout list.
- DELETE `/scanners/{id}/lockout/{freq}`
  - Remove frequency from lockout.
- DELETE `/scanners/{id}/lockouts`
  - Clear all lockouts.

### Trunking Systems (P25 trunking)
- GET `/trunking/systems`
  - List all trunking systems.
- POST `/trunking/systems`
  - Create system: `{ id, name, protocol, modulation?, controlChannels, centerHz, sampleRate, deviceId?, gain?, maxVoiceRecorders?, recordingPath?, squelchDb?, autoStart?, talkgroups? }`
- GET `/trunking/systems/{id}`
  - Get system details including NAC, system ID, site ID, decode rate, active calls.
- DELETE `/trunking/systems/{id}`
  - Remove trunking system.
- POST `/trunking/systems/{id}/start`
  - Start control channel monitoring and voice tracking.
  - C4FM control-channel decode uses soft-decision trellis when available.
  - Fails fast when `sampleRate`/`centerHz` are invalid or control channels fall outside the capture bandwidth.
- POST `/trunking/systems/{id}/stop`
  - Stop trunking system.
- GET `/trunking/systems/{id}/talkgroups`
  - List configured talkgroups.
- POST `/trunking/systems/{id}/talkgroups`
  - Import/add talkgroups.
- GET `/trunking/systems/{id}/calls/active`
  - List active calls for this system.
- GET `/trunking/calls`
  - List all active calls across all systems.
- GET `/trunking/vocoders`
  - Check IMBE/AMBE vocoder availability.
- GET `/trunking/systems/{id}/locations`
  - Get GPS locations from LRRP messages.
- GET `/trunking/systems/{id}/voice-streams`
  - List active voice streams for playback.
- GET `/trunking/recipes`
  - List pre-configured trunking system templates.

### Streaming
- WS `/stream/captures/{id}/iq`
  - Streams capture IQ frames (binary: `iq16` or `f32`).
- WS `/stream/captures/{id}/spectrum`
  - Streams FFT/spectrum data (JSON) for waterfall/spectrum analyzer.
- WS `/stream/channels/{chanId}?format={pcm16|f32}`
  - Streams binary PCM audio (no JSON envelope). Defaults to `pcm16`; `f32` is available for analysis clients. Encoded formats (MP3/Opus/AAC) are only available via the HTTP endpoints below.
- GET `/stream/channels/{chanId}.pcm?format={pcm16|f32}`
  - HTTP streaming for VLC and other players.
- GET `/stream/channels/{chanId}.mp3`
  - HTTP MP3 streaming.
- GET `/stream/channels/{chanId}.opus`
  - HTTP Opus streaming.
- GET `/stream/channels/{chanId}.aac`
  - HTTP AAC streaming.
- Encoder endpoints (MP3/Opus/AAC) stream raw encoded frames over chunked HTTP. Responses include `Cache-Control: no-cache` and `X-Audio-Rate` headers; clients must handle mid-stream disconnects and reissue requests if a channel is stopped/restarted. Encoders spin up lazily per-channel and are shared across all subscribers for a given format; when the last subscriber disconnects, the encoder shuts down automatically and subsequent subscribers will incur a new encoder spawn.
- Message shape for encoded streams:
  - Body: continuous encoded payload (MP3 frames, Ogg Opus, or AAC ADTS). Frame boundaries are not aligned to HTTP chunks; clients must parse frames from the bytestream.
  - Headers: `X-Audio-Rate` (Hz), `X-Audio-Format` (for PCM streams), `X-Audio-Channels: 1`.
- Encoder expectations:
  - Up to 32 packets are buffered per subscriber queue; overflow drops the oldest packet and increments the drop counter surfaced in `/health` streaming stats.
  - Encoder subprocesses (ffmpeg) use the channel’s configured `audioRate` and default to 128 kbps CBR. Bitrate selection will be exposed via config later; until then, consumers should not assume VBR support.
- WS `/stream/trunking/{systemId}`
  - Real-time trunking events (grants, denials, registrations) for a specific system.
- WS `/stream/trunking`
  - Trunking events for all systems.
- GET `/stream/trunking/{systemId}/voice/{streamId}.pcm`
  - HTTP streaming for trunked voice channels.

### Health
- GET `/health`
  - Comprehensive health check with device, capture, and channel status.

Error handling:
- JSON errors: `{ error: { code, message, details? } }` with appropriate HTTP status.
- Common codes: `DEVICE_BUSY`, `INVALID_CONFIG`, `NO_DEVICE`, `UNSUPPORTED_MODE`.

## Data Model

### Device
```
{ id, driver, label, freqMinHz, freqMaxHz, sampleRates[], gains[], gainMin, gainMax,
  bandwidthMin, bandwidthMax, ppmMin, ppmMax, antennas[], nickname?, shorthand? }
```

### Capture
```
{ id, deviceId, state, centerHz, sampleRate, gain?, bandwidth?, ppm?, antenna?,
  deviceSettings?, elementGains?, streamFormat?, dcOffsetAuto?, iqBalanceAuto?,
  errorMessage?, name?, autoName? }
```
States: `created`, `starting`, `running`, `stopping`, `stopped`, `failed`

### Channel
```
{ id, captureId, mode, state, offsetHz, audioRate, squelchDb?, name?, autoName?,
  signalPowerDb?, rssiDb?, snrDb?, ...dspConfig }
```
States: `created`, `running`, `stopped`

### Channel DSP Configuration
FM filters:
- `enableDeemphasis`, `deemphasisTauUs` (1-200 µs)
- `enableMpxFilter`, `mpxCutoffHz`
- `enableFmHighpass`, `fmHighpassHz`
- `enableFmLowpass`, `fmLowpassHz`

AM/SSB filters:
- `enableAmHighpass`, `amHighpassHz`
- `enableAmLowpass`, `amLowpassHz`
- `enableSsbBandpass`, `ssbBandpassLowHz`, `ssbBandpassHighHz`
- `ssbMode` ("usb" | "lsb")
- `ssbBfoOffsetHz` (BFO offset for centering voice)

SAM (Synchronous AM):
- `samSideband` ("dsb" | "usb" | "lsb")
- `samPllBandwidthHz` (10-200 Hz)

AGC:
- `enableAgc`, `agcTargetDb`, `agcAttackMs`, `agcReleaseMs`

Noise reduction:
- `enableNoiseReduction`, `noiseReductionDb` (spectral noise suppression, 3-30 dB)
- `notchFrequencies[]` (up to 10 frequencies for interference rejection)

### Scanner
```
{ id, captureId, state, currentFrequency, currentIndex, scanList[], mode,
  dwellTimeMs, priorityFrequencies[], priorityIntervalS, squelchThresholdDb,
  lockoutList[], pauseDurationMs, hits[] }
```
States: `stopped`, `scanning`, `paused`, `locked`
Modes: `sequential`, `priority`, `activity`

### Recipe
```
{ id, name, description, category, centerHz, sampleRate, gain?, bandwidth?,
  channels[], allowFrequencyInput?, frequencyLabel? }
```

### Trunking System
```
{ id, name, protocol, deviceId?, state, controlChannelState, controlChannelFreqHz?,
  nac?, systemId?, rfssId?, siteId?, decodeRate, activeCalls, talkgroups[] }
```
Protocol: `p25_phase1` | `p25_phase2`
States: `stopped`, `starting`, `running`, `failed`

Trunking stats (in `stats` field on system responses):
- `control_monitor.tsbk_rejected`: count of TSBK blocks dropped due to invalid/out-of-range fields.

Fast-fail validation:
- Non-finite IQ/audio samples and out-of-range trunking fields are dropped early to prevent propagating bogus values.

### Active Call
```
{ callId, talkgroupId, talkgroupName, frequency, startTime, duration,
  sourceId?, sourceName?, encrypted, emergency, priority, phase2? }
```

### Talkgroup
```
{ tgid, name, alphaTag?, category?, priority, record, monitor }
```

## Deployment & Running
WaveCap-SDR provides convenient startup scripts for all platforms:

- **Linux/macOS**: `./start-app.sh` from repository root
- **Windows**: `.\start-app.ps1` from repository root (PowerShell)

These scripts automatically:
- Create and configure Python virtual environment with system SoapySDR integration
- Install required dependencies (FastAPI, uvicorn, httpx, websockets, pyyaml, numpy, scipy, slowapi)
- Start the server with sensible defaults

Behavior on startup:
- If `device.driver=soapy` and no SoapySDR devices are detected at runtime, the server falls back to the built‑in Fake driver so the app remains usable without hardware.
- If no captures are configured in `config/wavecapsdr.yaml` (with optional overrides in `config/wavecapsdr.local.yaml`), the server initializes a single default capture (using the first preset if present) but does not auto‑start it. This ensures the UI/API display a capture immediately while avoiding hardware hangs.
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
- Files: `config/wavecapsdr.yaml` with optional overrides in `config/wavecapsdr.local.yaml` (gitignored), plus environment variables. Document all options in `docs/configuration.md`.
- Examples:
  - `bindAddress` (default `127.0.0.1`), `port` (default `8087`).
  - `auth.token` for simple bearer auth (optional, disabled by default).
  - `recording.rootDir` and filename pattern.
  - `limits.maxConcurrentCaptures`, `limits.maxChannelsPerCapture`, `limits.maxSampleRate`.
- Each successful save creates/refreshes a sibling `wavecapsdr.local.yaml.bak` (or `wavecapsdr.yaml.bak` if no local file is used) so operators can recover if a UI action overwrote capture configs unexpectedly.

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

### Completed
1. ✅ **MVP**: Device enumeration, capture start/stop, IQ streaming over WebSocket.
2. ✅ **WBFM/NBFM demod**: Audio streaming (PCM16, F32, MP3, Opus, AAC).
3. ✅ **Configuration**: YAML config file, bearer token auth, recipe presets.
4. ✅ **Web UI**: Spectrum analyzer, waterfall display, channel cards, creation wizard.
5. ✅ **Multi-device**: Simultaneous operation of RTL-SDR and SDRplay devices.
6. ✅ **DSP filters**: Deemphasis, highpass/lowpass, notch filters, AGC, noise blanker.
7. ✅ **Scanner**: Sequential/priority/activity scan modes with lockout management.
8. ✅ **Signal metrics**: RSSI/SNR measurement, S-meter display.
9. ✅ **Click-to-tune**: Interactive spectrum with frequency tooltip.
10. ✅ **AM/SSB modes**: Synchronous AM (SAM) with PLL, USB/LSB with BFO.
11. ✅ **RDS decoder**: Station name (PS), radio text (RT), PTY, TA/TP flags.
12. ✅ **POCSAG decoder**: Pager message decoding for numeric and alphanumeric formats.
13. ✅ **P25 trunking**: Phase 1/2 control channel decoding, voice channel tracking, TSBK parsing.
14. ✅ **Trunking API**: Complete REST/WebSocket API for P25 systems, talkgroups, and call tracking.
15. ✅ **IMBE/AMBE vocoders**: Vocoder integration for digital voice playback (external codec support).

### In Progress
16. **Digital voice modes**: NXDN, D-Star, YSF demodulation (stubs exist, full implementation pending).
17. **DMR trunking**: DMR Tier 3 trunking support with CSBK decoding.

### Planned
18. **Stereo FM**: 19 kHz pilot detection, L-R decoding for stereo broadcast.
19. **CW decoder**: Morse code recognition and text extraction.

## Out of Scope (for now)
- Multi‑node clustering and remote device brokers.
- Full DMR, NXDN, D-Star, and YSF voice demodulation (stubs in place).
- Commercial trunking protocols (EDACS, LTR, Motorola SmartZone).

## Encoders, Fixtures, and Round-trip Testing

- Audio encoders (MP3, Opus, AAC) are orchestrated per-channel and start only when a subscriber connects to the corresponding encoded endpoint. The same encoder instance fans out to all subscribers of that format; disconnecting all listeners tears it down.
- Fixture guidance:
  - IQ/audio fixtures live under `backend/harness_out/` when generated via the harness; keep short (≤10 s) to speed regression runs.
  - Preserve the JSON harness report emitted to stdout alongside any WAV/encoded assets so tests can assert RMS/peak without reprocessing audio.
- Round-trip workflow (no hardware required):
  1. Start the harness with the Fake driver: `cd backend && . .venv/bin/activate && PYTHONPATH=. python -m wavecapsdr.harness --start-server --driver fake --preset tone --duration 3 --out harness_out`.
  2. Capture PCM over WebSocket or download an encoded stream (e.g., `timeout 5s curl -o harness_out/channel.mp3 http://127.0.0.1:8087/api/v1/stream/channels/<chanId>.mp3`).
  3. Validate RMS/peak against the harness report; for encoded assets, decode with `ffmpeg -i harness_out/channel.mp3 -f wav -` and compare spectra to the saved PCM WAV for drift/levels.

## Demodulation Modes

| Mode | Status | Description |
|------|--------|-------------|
| `wbfm` | ✅ Complete | Wideband FM (broadcast) with deemphasis, 150 kHz deviation, RDS support |
| `nbfm` | ✅ Complete | Narrowband FM (VHF/UHF comms) with 5 kHz deviation, POCSAG paging |
| `am` | ✅ Complete | Envelope detection with configurable bandwidth and AGC |
| `sam` | ✅ Complete | Synchronous AM with carrier PLL for improved audio quality |
| `ssb` | ✅ Complete | USB/LSB with bandpass filter, BFO offset, and AGC |
| `raw` | ✅ Complete | Pass-through IQ samples for external processing |
| `p25` | ✅ Complete | P25 Phase 1/2 with IMBE/AMBE voice codec support and full trunking |
| `dmr` | 🚧 Partial | 4-FSK demodulation, CSBK decoding, voice codec integration in progress |
| `nxdn` | 🚧 Stub | Accepted as mode, demodulation not yet implemented |
| `dstar` | 🚧 Stub | Accepted as mode, demodulation not yet implemented |
| `ysf` | 🚧 Stub | Accepted as mode, demodulation not yet implemented |

P25 trunking specifics:
- TSBK CRC uses CRC-16 CCITT with init 0x0000, 16-bit flush, and final XOR 0xFFFF.
- Status symbols are stripped every 36 dibits (positions 36, 72, 108, ...).
- LDU voice frames are parsed using a 864-dibit window; NID decoding skips the status symbol at dibit 10.
- IDEN_UP channel identifiers are cached per system and can be preseeded via `channel_identifiers` in trunking config to resolve voice frequencies during weak CRC periods.

## Open Questions
- Preferred on‑disk container for IQ (raw vs. WAV container with metadata)?
- Channel scheduler behavior when requested targets drift outside passband (auto retune vs. error).

## Integration with WaveCap
- WaveCap (app/control plane) lives in `~/speaker/WaveCap` (symlinked as `~/speaker/smart-speaker`). WaveCap‑SDR exposes radio functionality so WaveCap can orchestrate higher‑level workflows (e.g., configuring multiple marine channels, recording policies). Keep integration docs and example flows in WaveCap and reference this spec for the SDR API.

## Hardware Support (initial)
- RTL-SDR (via Soapy RTL driver) — enumerate, tune, stream IQ at common rates (250 kS/s–2.4 MS/s).
- SDRplay RSPdx-r2 (via SoapySDRPlay3) — enumerate, tune, stream IQ; demod pipelines operate on the IQ stream.
