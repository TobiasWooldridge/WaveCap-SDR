# Antenna Configuration Implementation

## Overview
Full antenna selection support has been implemented for SDR devices that support multiple antennas (e.g., SDRplay RSPdx-R2).

## Implementation Details

### 1. Device Layer
- **File**: `backend/wavecapsdr/devices/base.py`
  - Added `antenna: Optional[str]` parameter to `Device.configure()` protocol
  - Added `get_antenna()` method to return configured antenna

- **File**: `backend/wavecapsdr/devices/soapy.py`
  - Added `_antenna: Optional[str]` field to `_SoapyDevice` dataclass
  - Modified `configure()` to accept and store antenna preference
  - Modified `start_stream()` to:
    - Use configured antenna if specified
    - Otherwise use first available antenna as fallback
    - Store actual antenna in use for later retrieval
  - Implemented `get_antenna()` to return current antenna

### 2. Capture Layer
- **File**: `backend/wavecapsdr/capture.py`
  - Added `antenna: Optional[str]` field to `CaptureConfig` dataclass
  - Added `antenna: Optional[str]` field to `Capture` dataclass to store actual antenna in use
  - Modified `Capture._run_thread()` to:
    - Pass antenna to device configuration
    - Store actual antenna after stream starts
  - Modified `CaptureManager.create_capture()` to accept antenna parameter

### 3. API Layer
- **File**: `backend/wavecapsdr/models.py`
  - Added `antenna: Optional[str]` to `CreateCaptureRequest` model
  - Added `antenna: Optional[str]` to `CaptureModel` response model

- **File**: `backend/wavecapsdr/api.py`
  - Updated all capture endpoints to handle antenna field:
    - `POST /api/v1/captures` - accepts antenna in request
    - `GET /api/v1/captures` - returns antenna in response
    - `GET /api/v1/captures/{cid}` - returns antenna
    - `POST /api/v1/captures/{cid}/start` - returns antenna after start
    - `POST /api/v1/captures/{cid}/stop` - returns antenna

### 4. Web UI
- **File**: `backend/wavecapsdr/static/index.html`
  - Added antenna display row in channel detail table (line 246-249)
  - Shows antenna in bold when available
  - Conditionally rendered (only shows if antenna is set)

### 5. Configuration
- **File**: `backend/config/wavecapsdr.yaml`
  - Added `antennas` section documenting RSPdx-R2 antenna ports:
    - **Ant A**: TW-777BNC fixed stub (2m/70cm, 144/440 MHz)
    - **Ant B**: GRA-RH795 telescoping (**best for FM broadcast 88-108 MHz**)
    - **Ant C**: SRH789 telescoping (2m/70cm, 144/440 MHz)
  - Updated KEXP preset to specify `antenna: "Ant B"`

## Usage

### Creating a Capture with Antenna Selection

```bash
curl -X POST http://localhost:8087/api/v1/captures \
  -H "Content-Type: application/json" \
  -d '{
    "deviceId": "driver=sdrplay,serial=240309F070",
    "centerHz": 90300000,
    "sampleRate": 2000000,
    "antenna": "Ant B"
  }'
```

Response:
```json
{
  "id": "c1",
  "deviceId": "driver=sdrplay,serial=240309F070",
  "state": "created",
  "centerHz": 90300000,
  "sampleRate": 2000000,
  "antenna": null  // Will be set when capture starts
}
```

After starting the capture, the actual antenna in use will be returned:
```json
{
  "antenna": "Ant B"
}
```

### Antenna Recommendations

For **FM Broadcast (88-108 MHz)**:
- Use **Ant B (GRA-RH795)** - wideband telescoping antenna optimized for this range

For **VHF Amateur Radio (144 MHz, 2m band)**:
- Use **Ant C (SRH789)** - optimized for 2m/70cm

For **UHF Amateur Radio (440 MHz, 70cm band)**:
- Use **Ant A (TW-777BNC)** - fixed stub optimized for VHF/UHF

## Testing

All implementation tests pass:
```bash
cd backend
PYTHONPATH=. .venv/bin/python ../test_antenna.py
```

Output:
```
✓ _SoapyDevice dataclass accepts _antenna field
✓ Device.configure() accepts antenna parameter
✓ CaptureManager.create_capture() accepts antenna parameter
✓ CreateCaptureRequest includes antenna field
✓ CaptureModel includes antenna field
✅ All antenna support tests passed!
```

## Files Modified

1. `backend/wavecapsdr/devices/base.py` - Protocol definition
2. `backend/wavecapsdr/devices/soapy.py` - SoapySDR implementation
3. `backend/wavecapsdr/capture.py` - Capture management
4. `backend/wavecapsdr/models.py` - API models
5. `backend/wavecapsdr/api.py` - API endpoints
6. `backend/wavecapsdr/static/index.html` - Web UI display
7. `backend/config/wavecapsdr.yaml` - Antenna configuration

## Web UI Display

When viewing the catalog page at `http://localhost:8087/`, channels will now show:
```
Frequency:        90.3 MHz
Mode:             WBFM
Audio Sample Rate: 48 kHz (48000 Hz)
SDR Sample Rate:  2.00 MHz (2000000 Hz)
Offset:           0 Hz
Antenna:          Ant B    ← NEW: Shows which antenna is in use
Capture ID:       c1
PCM16 Stream URL: http://localhost:8087/api/v1/stream/channels/ch1.pcm
```

## Backward Compatibility

- Antenna parameter is optional in all APIs
- Existing captures without antenna specification continue to work
- Devices without antenna selection (e.g., RTL-SDR) return `null` for antenna
- First available antenna is used as fallback if none specified
