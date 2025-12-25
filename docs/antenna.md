# Antenna Selection

Multi-antenna SDR devices (e.g., SDRplay RSPdx-R2) support antenna selection via the `antenna` parameter in capture creation.

## Implementation Status

Antenna selection is **fully implemented** and operational across the codebase:

- Backend API accepts `antenna` parameter in capture creation/update
- Device drivers (SoapySDR, SDRplay proxy) support antenna switching
- Frontend UI includes antenna selector dropdown for multi-antenna devices
- Configuration presets support antenna specification
- Trunking systems support antenna configuration

## Usage

### API

Pass `antenna` when creating a capture:
```bash
curl -X POST http://localhost:8087/api/v1/captures \
  -H "Content-Type: application/json" \
  -d '{"deviceId": "...", "centerHz": 90300000, "sampleRate": 2000000, "antenna": "Antenna B"}'
```

Update antenna on running capture:
```bash
curl -X PATCH http://localhost:8087/api/v1/captures/{id} \
  -H "Content-Type: application/json" \
  -d '{"antenna": "Antenna C"}'
```

### Configuration

Presets can specify antenna selection in `backend/config/wavecapsdr.yaml`:
```yaml
presets:
  kexp:
    center_hz: 91900000
    sample_rate: 1024000
    antenna: Antenna B  # SDRplay antennas use format "Antenna A/B/C"
```

Trunking systems also support antenna configuration:
```yaml
trunking:
  systems:
    sa_grn:
      antenna: Antenna B
      device_settings:
        rfnotch_ctrl: 'false'
```

### UI

The frontend automatically displays an antenna selector when a device reports multiple antennas. The selector appears in the Device Controls section of the Radio Panel.

## Antenna Naming

Different SDR devices use different naming conventions:

- **SDRplay RSPdx-R2**: `Antenna A`, `Antenna B`, `Antenna C`
- **RTL-SDR**: Single antenna, typically `RX` (no selection needed)
- **Other devices**: Check device enumeration response for available antennas

## RSPdx-R2 Antenna Recommendations

For the SDRplay RSPdx-R2 with specific antennas:

- **Antenna B (GRA-RH795)**: Best for FM broadcast (88-108 MHz)
  - Note: Bias-T (4.7V) is only available on Antenna B
- **Antenna C (SRH789)**: VHF amateur (144 MHz, 2m band)
- **Antenna A (TW-777BNC)**: UHF amateur (440 MHz, 70cm band)

## Implementation Details

Key files:
- `backend/wavecapsdr/models.py` - API models with antenna field
- `backend/wavecapsdr/api.py` - API endpoints with antenna validation
- `backend/wavecapsdr/devices/soapy.py` - SoapySDR antenna switching
- `backend/wavecapsdr/devices/sdrplay_proxy.py` - SDRplay antenna support
- `frontend/src/features/radio/TuningControls.tsx` - UI antenna selector
