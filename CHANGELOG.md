# Changelog

## 2025-10-26 - Alpha Release: Core Functionality

### Features Implemented
- Device enumeration via SoapySDR for multiple simultaneous devices
- IQ sample streaming from RTL-SDR and SDRplay devices
- WBFM demodulation pipeline
- WebSocket audio streaming (PCM16)
- Multi-channel support (multiple demod channels from single capture)
- Test harness with automatic gain calibration
- Timeout wrappers for SoapySDR utilities

### Bug Fixes

#### Critical Fixes for SoapySDR Python Bindings Compatibility

**1. Channel Audio Processing (capture.py:307-323)**
- **Issue**: Attempted to call `asyncio.create_task()` from non-async context (capture thread)
- **Impact**: Channel demodulation never started; no audio output despite IQ samples flowing
- **Fix**: Changed to `asyncio.run_coroutine_threadsafe()` to properly schedule channel processing from the synchronous capture thread to the async event loop
- **Root cause**: The capture thread runs in a separate thread without an event loop, so direct async calls fail

**2. SoapySDR Stream API (soapy.py:28-50)**
- **Issue**: Code assumed `readStream()` returned a tuple, but it returns a `StreamResult` object
- **Impact**: TypeError when comparing `StreamResult` to integer: `'<' not supported between instances of 'StreamResult' and 'int'`
- **Fix**: Added handling for `StreamResult` objects with `.ret` and `.flags` attributes
- **Additional issue**: `SDR_READ_FLAG_NONE` constant not available in all Python bindings
- **Fix**: Use integer constants directly instead of named constants that may not exist

**3. Device Enumeration (soapy.py:121-157)**
- **Issue**: `SoapySDRKwargs` object doesn't support `.get()` method despite being dict-like
- **Impact**: AttributeError during device enumeration; multi-device operations failed
- **Fix**: Changed from `args.get("key", default)` to `args["key"] if "key" in args else default`

### Tested Hardware
- RTL-SDR Blog V4 (Rafael Micro R828D tuner)
- SDRplay RSPdx-R2
- Simultaneous operation verified on both devices

### Known Issues
- Audio demodulation requires active WebSocket client (no server-side-only recording yet)
- Multi-device auto-gain calibration can be slow (probes multiple gain settings per device)
- Event loop coordination between capture thread and async API needs refinement

### Testing
- Test harness validates: device open, IQ sample flow, audio demodulation, level thresholds
- Scripts provide timeout wrappers to prevent hangs
- WAV file output for manual verification
- Tested with KEXP 90.3 FM (Seattle) as reference signal

### Dependencies
- Python 3.9+ (tested with 3.13)
- SoapySDR system package with Python bindings
- FastAPI, uvicorn, httpx, websockets, numpy, scipy
- Device-specific modules: SoapyRTLSDR, SoapySDRPlay3

### Documentation
- Updated README with setup instructions
- Added `docs/troubleshooting.md` with common issues and solutions
- Added changelog for tracking changes
