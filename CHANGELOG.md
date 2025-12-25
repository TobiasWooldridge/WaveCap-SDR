# Changelog

All notable changes to WaveCap-SDR are documented in this file.

## 2025-12-25 - P25 Trunking and Digital Voice Enhancements

**P25 Trunking Improvements:**
- Fixed center frequency recentering when control channel changes
- Fixed test suite by adding `auto_start` parameter to trunking API
- Improved P25 voice channel following and call event tracking
- Fixed critical P25 decoder bugs for TSBK decoding and frame synchronization
- Added read-only channels UI for trunking-managed captures
- Added audio playback and stream URLs to trunking UI
- Fixed P25 trunking TSBK decoding and callback routing based on SDRTrunk algorithm comparison

**Digital Voice Modes:**
- Added NXDN, D-Star, and YSF digital voice mode stubs
- Wired DMR AMBE+2 voice decoding to stateful processing path
- Wired P25 IMBE voice decoding to stateful processing path

**Demodulation Enhancements:**
- Added SAM (Synchronous AM) demodulation mode with PLL carrier recovery
- Made SSB BFO offset frequency configurable
- Fixed C4FM demodulator scaling and thresholds for P25 decoding

**FFT and Performance:**
- Added pluggable FFT backends for improved spectrum analyzer performance
- Added FFT Max FPS setting in UI
- Added per-capture DSP executor for CPU isolation
- Improved threading model to use ThreadPoolExecutor for all DSP processing

**Development and Infrastructure:**
- Improved linting and type checking infrastructure with Python 3.10+ modernization
- Added configuration warnings system for capture lint errors
- Fixed UI bugs and enhanced error handling
- Added frontend logging system and WebSocket debug instrumentation

## 2025-12-19 - Trunking Event Broadcasting and UI Improvements

**Trunking System Fixes:**
- Fixed trunking event broadcast scheduling to prevent race conditions
- Route trunking voice recorder audio via event loop for proper async handling
- Schedule trunking voice channel tasks on main event loop
- Await trunking recorder release on stop to prevent resource leaks

**UI and Developer Experience:**
- Replaced positional dropdowns with modal dialogs and fixed component bugs
- Added device refresh button and periodic device re-enumeration
- Enabled ESLint and cleaned up React hook dependencies
- Hardened capture handling and API error parsing
- Updated test documentation with accurate commands
- Refreshed repository documentation

## 2025-12-12 - Trunking UI Unification

**User Interface:**
- Unified Radio and Trunking tabs into single tab bar for better UX
- Show device name in trunking system tabs
- Improved tab navigation and state management

## 2025-12-08 to 2025-12-11 - P25 Trunking System (Phase 1-7)

**Complete P25 Trunking Implementation:**
- **Phase 1-2**: Added P25 DSP core implementation for Phase I trunked radio support
- **Phase 3**: Added P25 trunking system infrastructure (manager, system state, configuration)
- **Phase 4-5**: Added P25 vocoder integration (IMBE/AMBE) and trunking REST API
- **Phase 6**: Added P25 trunking frontend UI components with real-time call tracking
- **Phase 7**: Integrated P25 trunking with SDR capture system for end-to-end functionality
- Added P25 frame decoder and TSBK parser for Phase II trunking support
- Added Radio/Trunking mode selector to main application UI
- Added `~/.local/bin` to PATH in start-app.sh for DSD-FME vocoder support
- Implemented event tracking, duplicate detection, and network configuration monitoring

**Trunking Architecture:**
- `IdentifierCollection`: Flexible metadata management with immutable/mutable variants
- `P25EventTracker`: Call state machine with staleness detection
- `NetworkConfigurationMonitor`: System configuration tracking from control channel
- `DuplicateCallDetector`: Time-based duplicate event suppression
- Voice channel following with automatic frequency tracking
- Control channel scanning and frequency band management

## 2025-12-07 - Performance, UI Polish, and SDRplay Resilience

**Performance Optimizations:**
- Optimized UI rendering performance with React memoization
- Added configurable FFT settings and adaptive FPS for spectrum analyzer
- Improved spectrum data processing pipeline

**UI Enhancements:**
- Added compact VolumeSlider component for cleaner audio controls
- Made AudioWaveform expand to full container width
- Stop audio playback when switching between radio tabs
- Improved component layout and responsive design

**SDRplay Improvements:**
- Added SDRplay resilience improvements with automatic service recovery
- Added proactive health monitoring with timeout detection
- Implemented rate-limited restart logic (max 5 restarts/hour, 60s cooldown)
- Added SA-GRN P25 trunking system support and configuration

**Code Quality:**
- Fixed type annotations in device drivers and decoders
- Fixed type annotations in RTL-SDR driver
- Improved error handling and logging

## 2025-10-26 - Antenna Selection

Added antenna selection for multi-antenna SDR devices (e.g., SDRplay RSPdx-R2). Pass `antenna` parameter in capture creation. See `docs/antenna.md` for details.

## 2025-10-26 - Web UI and Streaming

Added web UI catalog page with embedded audio players, multi-format streaming (PCM16/F32), HTTP streaming endpoint, and 14x FM demodulation speedup via scipy.

## 2025-10-26 - Alpha Release

Initial release with device enumeration, IQ streaming, WBFM demodulation, WebSocket audio streaming, and multi-channel support. Tested with RTL-SDR Blog V4 and SDRplay RSPdx-R2.

**Bug fixes**: SoapySDR compatibility (async event loop, StreamResult handling, SoapySDRKwargs access)

**Known issues**: Audio requires active WebSocket client; multi-device gain calibration is slow
