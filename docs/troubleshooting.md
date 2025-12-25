# Troubleshooting

## Common Issues

### Captures Stuck in "starting" State

**Symptoms**: Capture shows "starting" but never transitions to "running", spectrum display doesn't update.

**Causes**:
1. **SDRplay API service stuck** (most common for SDRplay devices)
2. **IQ watchdog timeout** (device not producing samples)
3. **Device in use by another process**

**Solutions**:

1. Check if SDRplay service is responsive:
   ```bash
   # Test with timeout (should complete in <5 seconds)
   timeout 5 SoapySDRUtil --find
   ```
   If this hangs, restart the SDRplay service:

   **macOS:**
   ```bash
   # Via API (recommended)
   curl -X POST http://localhost:8087/api/v1/devices/sdrplay/restart-service

   # Or manually
   sudo /bin/launchctl kickstart -kp system/com.sdrplay.service
   ```

   **Linux:**
   ```bash
   # Via API (recommended)
   curl -X POST http://localhost:8087/api/v1/devices/sdrplay/restart-service

   # Or manually
   sudo systemctl restart sdrplay
   ```

2. Check device availability:
   ```bash
   pkill -9 -f wavecapsdr   # Kill any processes using the device
   lsof | grep -i sdr        # Check what has it open
   ```

3. Check server logs for specific error messages:
   ```bash
   # Recent errors
   cd backend && tail -100 wavecapsdr.log | grep -i error
   ```

### Audio Not Playing

**Symptoms**: Capture running, spectrum updating, but no audio output.

**Cause**: Channels now process audio continuously (no WebSocket subscriber required).

**Solutions**:
1. Check channel state is "running" (not "created" or "stopped")
2. Verify squelch settings - signal may be below squelch threshold
3. Check signal strength in channel panel (RSSI should be > squelch_db)
4. Test with squelch disabled: `squelch_db: null` in channel config
5. Verify audio format is supported by your player (PCM16, F32, MP3, Opus, AAC)

### SDRplay Device Not Detected

**Symptoms**: USB device present but not shown in device list, enumeration timeout.

**Diagnosis**:
```bash
# Check USB device present
# macOS:
system_profiler SPUSBDataType 2>/dev/null | grep -B5 -A10 "1df7\|SDRplay"

# Linux:
lsusb | grep -i "1df7\|sdrplay"
```

**Fix**: See "Captures Stuck in 'starting' State" section above.

### SDRplay Service Health Monitoring

WaveCap-SDR includes automatic health monitoring for SDRplay devices:

```bash
# Check service health
curl http://localhost:8087/api/v1/devices/sdrplay/health

# Restart service
curl -X POST http://localhost:8087/api/v1/devices/sdrplay/restart-service
```

The system automatically detects stuck states during enumeration and attempts recovery.

## SoapySDR Compatibility

SoapySDR Python bindings vary across distributions. The codebase handles:
- Integer constants instead of named constants (e.g., `SOAPY_SDR_RX`)
- `StreamResult` objects with `.ret` and `.flags` attributes
- `SoapySDRKwargs` objects require `args["key"]` not `args.get("key")`

## Device Detection

**Timeout or hangs**: Increase timeout with `MAX_SECONDS=30 scripts/soapy-find.sh`

**Device detected but can't open**:
```bash
pkill -9 -f wavecapsdr   # Kill any processes using the device
lsof | grep -i sdr        # Check what has it open
```

## USB Permissions (Linux)

Add user to `plugdev` group:
```bash
sudo usermod -a -G plugdev $USER  # Log out/in after
```

Or create `/etc/udev/rules.d/99-sdr.rules`:
```bash
# RTL-SDR
SUBSYSTEM=="usb", ATTRS{idVendor}=="0bda", ATTRS{idProduct}=="2838", GROUP="plugdev", MODE="0666"
# SDRplay
SUBSYSTEM=="usb", ATTRS{idVendor}=="1df7", ATTRS{idProduct}=="3010", GROUP="plugdev", MODE="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="1df7", ATTRS{idProduct}=="3020", GROUP="plugdev", MODE="0666"

# Reload rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

## Custom SoapySDRPlay3 Driver

**Important**: This project requires a custom SoapySDRPlay3 driver with multi-device fixes:
- Repository: https://github.com/TobiasWooldridge/SoapySDRPlay3
- Includes API-level locking to prevent crashes on rapid config changes

Build and install:
```bash
cd ../SoapySDRPlay3
mkdir -p build && cd build
cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ..
make -j4
sudo make install
```

## Test Harness

Timeouts: Increase duration with `DURATION=10 bash scripts/harness-kexp.sh`

## Python Environment

Create venv with system packages for SoapySDR access:
```bash
python3 -m venv --system-site-packages .venv
```

## Process Management

Kill all related processes:
```bash
pkill -9 -f "wavecapsdr|uvicorn|python.*harness"
ps aux | grep -E "wavecapsdr|rtl_|sdr"  # Check for orphans
```

## Debugging

**End-to-end health check**:
```bash
.claude/skills/capture-health-check/check_health.sh
```

**SoapySDR device info**:
```bash
SoapySDRUtil --probe="driver=rtlsdr"
```

**USB monitoring**:
```bash
# macOS
system_profiler SPUSBDataType

# Linux
sudo dmesg -w | grep -i usb
lsusb
```

**Low/no audio output**:
- Check antenna connection
- Verify frequency accuracy (RTL-SDR PPM correction)
- Check local signal strength
- Verify squelch settings (signal must exceed squelch threshold)
- Test with squelch disabled

## Known Issues

1. **Audio requires active WebSocket client** (legacy, now fixed): Channels now process audio continuously
2. **Multi-device gain calibration is slow**: Initial device opening can take 10-15 seconds
3. **SDRplay service can become stuck**: Use automatic recovery or manual restart (see above)
