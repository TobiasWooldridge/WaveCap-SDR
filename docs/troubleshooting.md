# Troubleshooting Guide

## Common Issues and Solutions

### SoapySDR Python Bindings

**Issue**: The SoapySDR Python bindings vary across distributions and versions. Some expose named constants like `SOAPY_SDR_RX`, while others don't.

**Solution**: The codebase uses integer constants directly where needed and handles both tuple and `StreamResult` return types from `readStream()`.

**Key compatibility notes**:
- Use `args["key"]` instead of `args.get("key")` for `SoapySDRKwargs` objects
- `readStream()` returns `StreamResult` with `.ret` and `.flags` attributes (not a tuple)
- Flag constants may not be available; use bit values directly (e.g., `1 << 1` for overflow)

### Device Detection

**Symptom**: `scripts/soapy-find.sh` times out or hangs

**Cause**: SoapySDRUtil can hang when probing devices with driver issues

**Solution**:
- The soapy wrapper scripts use timeouts (default 10s)
- Increase timeout: `MAX_SECONDS=30 scripts/soapy-find.sh`
- Check USB permissions: device files should be accessible (group `plugdev` on Ubuntu)

**Symptom**: Devices detected but can't open

**Common causes**:
1. Another process has the device open (check with `lsof | grep -i rtl`)
2. Kernel driver attached (RTL-SDR dongles often attach `dvb_usb_rtl28xxu`)
3. USB permissions issue

**Solution**:
```bash
# Kill processes using the device:
pkill -9 rtl_
pkill -9 -f wavecapsdr

# Check which process has the device:
lsof | grep -i sdr
```

### Audio Streaming Issues

**Symptom**: IQ samples are being read but no audio output

**Debugging**:
1. Check that WebSocket connection is established (log shows "connection open")
2. Verify channel is in "running" state
3. Check for event loop in the capture thread

**Known issue**: The channel processing requires an active event loop from a WebSocket subscriber. If the WebSocket disconnects early, demodulation may not occur.

### USB Device Permissions

On Linux, SDR devices need proper permissions. Add user to `plugdev` group:

```bash
sudo usermod -a -G plugdev $USER
# Log out and back in for group changes to take effect
```

Or create a udev rule:
```bash
# /etc/udev/rules.d/99-sdr.rules
SUBSYSTEM=="usb", ATTRS{idVendor}=="0bda", ATTRS{idProduct}=="2838", GROUP="plugdev", MODE="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="1df7", ATTRS{idProduct}=="3010", GROUP="plugdev", MODE="0666"
```

Then reload:
```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### SDRplay-specific Issues

**Issue**: SDRplay device not detected

**Requirements**:
1. SDRplay API v3.x installed (download from sdrplay.com)
2. `SoapySDRPlay3` module installed
3. Service must be running: `sudo systemctl start sdrplay`

**Verify**:
```bash
SoapySDRUtil --find="driver=sdrplay"
```

### Test Harness Timeouts

**Symptom**: Harness scripts time out even though device is working

**Cause**: The harness expects audio data within a timeout period. Weak signals or configuration issues can cause this.

**Solutions**:
- Increase duration: `DURATION=10 bash scripts/harness-kexp.sh`
- Adjust gain: `GAIN=30 bash scripts/harness-kexp.sh`
- Use auto-gain: `scripts/harness-kexp.sh` (default behavior)
- Check signal strength for your location (KEXP 90.3 FM is Seattle-area)

### Python Environment Issues

**Symptom**: `ModuleNotFoundError: No module named 'SoapySDR'`

**Solution**: Create venv with `--system-site-packages`:
```bash
python3 -m venv --system-site-packages .venv
```

This allows the venv to access system-installed SoapySDR Python bindings.

### Background Process Management

**Symptom**: Multiple test runs fail with "device busy"

**Solution**: Kill all related processes:
```bash
pkill -9 -f "wavecapsdr|uvicorn|python.*harness"
```

Check for orphaned processes:
```bash
ps aux | grep -E "wavecapsdr|rtl_|sdr"
```

## Debugging Tips

### Enable Verbose Logging

Add debug output to capture thread:
```python
import sys
print(f"DEBUG: message", file=sys.stderr, flush=True)
```

### Check SoapySDR Device Info

```bash
SoapySDRUtil --probe="driver=rtlsdr"
SoapySDRUtil --probe="driver=sdrplay"
```

### Monitor USB Activity

```bash
# Watch USB device connections:
sudo dmesg -w | grep -i usb

# Check current USB devices:
lsusb
```

### Test IQ Data Flow

The harness validates:
1. Device opens successfully
2. IQ samples are being read (check for "Read 4096 samples" in debug logs)
3. Audio is demodulated and has sufficient level (RMS > 0.003, peak > 0.05)
4. WAV files are written to `backend/harness_out/`

If WAV files are created but empty or very quiet, check:
- Antenna connection
- Frequency accuracy (RTL-SDR may need PPM correction)
- Local signal strength for the test frequency

## Known Limitations

- **Multi-device coordination**: While multiple devices can be used simultaneously, there's currently no built-in mechanism to prevent frequency conflicts or coordinate gain settings across devices.

- **Event loop dependency**: Channel demodulation requires an active WebSocket client to provide the event loop context. Server-side recording without a client is not yet implemented.

- **PPM correction**: RTL-SDR devices may require frequency correction. Set via `ppm` parameter in capture configuration.

## Getting Help

For issues not covered here:
1. Check `SPEC.md` for API details
2. Review `AGENTS.md` for development guidelines
3. Check SoapySDR documentation: https://github.com/pothosware/SoapySDR/wiki
4. Verify device-specific requirements (RTL-SDR, SDRplay) on vendor sites
