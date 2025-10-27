# Antenna Selection

Multi-antenna SDR devices (e.g., SDRplay RSPdx-R2) support antenna selection via the `antenna` parameter in capture creation.

## Usage

Pass `antenna` when creating a capture:
```bash
curl -X POST http://localhost:8087/api/v1/captures \
  -H "Content-Type: application/json" \
  -d '{"deviceId": "...", "centerHz": 90300000, "sampleRate": 2000000, "antenna": "Ant B"}'
```

## RSPdx-R2 Antenna Recommendations

- **Ant B (GRA-RH795)**: Best for FM broadcast (88-108 MHz)
- **Ant C (SRH789)**: VHF amateur (144 MHz, 2m band)
- **Ant A (TW-777BNC)**: UHF amateur (440 MHz, 70cm band)

See `backend/config/wavecapsdr.yaml` for device-specific antenna configurations.
