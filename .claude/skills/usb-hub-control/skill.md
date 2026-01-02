# USB Hub Control for WaveCap-SDR

This skill provides USB power cycling capabilities for SDR devices using uhubctl.

## When to Use This Skill

Use this skill when:
- SDR device is stuck or unresponsive
- SDRplay API service won't enumerate devices
- Need to reset USB devices without physical access
- Device shows in `lsusb` but not in SoapySDR

## Prerequisites

Requires `uhubctl` to be installed:
```bash
brew install uhubctl  # macOS
apt install uhubctl   # Linux
```

Your USB hub must support per-port power switching (PPPS). Most powered USB 3.0 hubs support this.

## Common Commands

### List all USB hubs and ports
```bash
sudo uhubctl
```

### Show specific hub
```bash
sudo uhubctl -l 2-1.1.4
```

### Power cycle specific ports
```bash
# Power off
sudo uhubctl -l 2-1.1.4 -p 3,4 -a off

# Wait for capacitors to discharge
sleep 3

# Power on
sudo uhubctl -l 2-1.1.4 -p 3,4 -a on
```

### Power cycle with automatic delay
```bash
sudo uhubctl -l 2-1.1.4 -p 3,4 -a cycle -d 5
```

## Known Device Locations (WaveCap-SDR Lab)

| Hub | Port | Device |
|-----|------|--------|
| 2-1.1.4 | 2 | RTL-SDR Blog V4 (0bda:2838) |
| 2-1.1.4 | 3 | SDRplay RSPdx-R2 (1df7:3060) |
| 2-1.1.4 | 4 | SDRplay RSPdx-R2 (1df7:3060) |

## Full SDRplay Reset Procedure

When SDRplay devices are stuck:

```bash
# 1. Stop any running SDR applications
pkill -f "python.*wavecapsdr"

# 2. Kill SDRplay service
sudo killall -9 sdrplay_apiService

# 3. Power cycle SDRplay USB ports
sudo uhubctl -l 2-1.1.4 -p 3,4 -a off
sleep 5
sudo uhubctl -l 2-1.1.4 -p 3,4 -a on
sleep 5

# 4. Restart SDRplay service
sudo /bin/launchctl kickstart -kp system/com.sdrplay.service
sleep 3

# 5. Verify enumeration
SoapySDRUtil --find="driver=sdrplay"
```

## Troubleshooting

### Hub not showing in uhubctl
- Check if hub supports PPPS: `lsusb -v 2>/dev/null | grep -A5 "Hub Descriptor"`
- Some hubs need firmware updates to enable PPPS

### Permission denied
- uhubctl requires root: use `sudo`
- Or set up udev rules (Linux) for passwordless access

### Device not reconnecting after power on
- Wait longer (5-10 seconds) for USB enumeration
- Check `dmesg` or `log show` for USB errors
- Try unplugging from hub entirely and using different port

### SDRplay still not enumerating after power cycle
- The SDRplay API service may have corrupted state
- Try: `sudo killall -9 sdrplay_apiService` before power cycle
- Verify USB connection: `system_profiler SPUSBDataType | grep -A10 "1df7"`

## Integration with fix-sdr-devices.sh

The main fix script at `scripts/fix-sdr-devices.sh` can be extended to use uhubctl:

```bash
# Add to fix-sdr-devices.sh
if command -v uhubctl &> /dev/null; then
    echo "Power cycling SDRplay USB ports..."
    sudo uhubctl -l 2-1.1.4 -p 3,4 -a cycle -d 5
fi
```

## Passwordless sudo for uhubctl

To allow uhubctl without password prompts, add to `/etc/sudoers.d/uhubctl`:

```
%admin ALL=(ALL) NOPASSWD: /opt/homebrew/bin/uhubctl
```

Create with:
```bash
sudo visudo -f /etc/sudoers.d/uhubctl
```
