# Changelog

## 2025-10-26 - Antenna Selection

Added antenna selection for multi-antenna SDR devices (e.g., SDRplay RSPdx-R2). Pass `antenna` parameter in capture creation. See `ANTENNA_IMPLEMENTATION.md` for details.

## 2025-10-26 - Web UI and Streaming

Added web UI catalog page with embedded audio players, multi-format streaming (PCM16/F32), HTTP streaming endpoint, and 14x FM demodulation speedup via scipy.

## 2025-10-26 - Alpha Release

Initial release with device enumeration, IQ streaming, WBFM demodulation, WebSocket audio streaming, and multi-channel support. Tested with RTL-SDR Blog V4 and SDRplay RSPdx-R2.

**Bug fixes**: SoapySDR compatibility (async event loop, StreamResult handling, SoapySDRKwargs access)

**Known issues**: Audio requires active WebSocket client; multi-device gain calibration is slow
