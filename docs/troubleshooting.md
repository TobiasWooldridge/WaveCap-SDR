# Troubleshooting

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

## Audio Streaming

IQ samples flowing but no audio: Check WebSocket connection is active and channel is in "running" state. Channel processing requires an active WebSocket subscriber.

## USB Permissions

Add user to `plugdev` group:
```bash
sudo usermod -a -G plugdev $USER  # Log out/in after
```

Or create `/etc/udev/rules.d/99-sdr.rules`:
```bash
SUBSYSTEM=="usb", ATTRS{idVendor}=="0bda", ATTRS{idProduct}=="2838", GROUP="plugdev", MODE="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="1df7", ATTRS{idProduct}=="3010", GROUP="plugdev", MODE="0666"
```

## SDRplay Setup

Requires SDRplay API v3.x, `SoapySDRPlay3` module, and service running:
```bash
sudo systemctl start sdrplay
SoapySDRUtil --find="driver=sdrplay"  # Verify
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

**SoapySDR device info**:
```bash
SoapySDRUtil --probe="driver=rtlsdr"
```

**USB monitoring**:
```bash
sudo dmesg -w | grep -i usb
lsusb
```

**Low/no audio output**: Check antenna connection, frequency accuracy (RTL-SDR PPM correction), and local signal strength.
