# WaveCap-SDR Development Journal

## 2026-01-02: SDRplay Enumeration Debug + RTL-SDR Tuning

### What We Were Doing

1. **Primary Goal**: Get RTL-SDR tuned for SA-GRN trunking and debug why it previously got more details than the RSPdx radios

2. **SDRplay Enumeration Issue** (blocking):
   - SDRplay devices not enumerating - `SoapySDRUtil --find` hangs
   - Service logs show: `USBInterfaceOpen: another process has device opened` and `pthread_mutex_destroy: Resource busy`
   - Tried: USB power cycling, service restarts, killing orphaned processes
   - **Resolution**: Rebooting to clear stuck kernel-level USB state

3. **RTL-SDR Decimation Fix** (completed):
   - Found RTL-SDR was using wrong decimation (120:1 fixed) giving only 20 kHz / 4.2 SPS
   - Fixed in `backend/wavecapsdr/trunking/system.py` - now adaptive based on sample rate
   - RTL-SDR now gets 48 kHz / 10.0 SPS (correct for P25)

4. **Pending After Reboot**:
   - Verify SDRplay enumeration works
   - Start WaveCap-SDR server
   - Compare RTL-SDR vs RSPdx decode rates on SA-GRN
   - Debug weak signal decode (error_metric ~26, need <15 for CRC pass)

### Key Files Modified (uncommitted)

- `backend/wavecapsdr/api.py` - Send empty devices list in WebSocket snapshot
- `backend/wavecapsdr/trunking/api.py` - Add deviceName to SystemResponse
- `frontend/src/hooks/useSelectedRadio.ts` - Use deviceName from API
- `frontend/src/types/trunking.ts` - Add deviceName field

### Key Files Modified (previous session, already committed)

- `backend/wavecapsdr/trunking/system.py` - Adaptive decimation fix
- `backend/wavecapsdr/trunking/control_channel.py` - Lowered sync threshold to 100

### Commands to Resume After Reboot

```bash
# 1. Verify SDRplay enumeration
SoapySDRUtil --find

# 2. Start server
cd /Users/thw/Projects/WaveCap-SDR
./start-app.sh

# 3. Check trunking status
curl -s http://localhost:8087/api/v1/trunking/systems | python3 -m json.tool
```

### USB Hub Layout (for reference)

```
Hub 2-1.1.4 (Generic USB2.1 Hub):
  Port 2: RTL-SDR Blog V4 (0bda:2838)
  Port 3: SDRplay RSPdx-R2 (1df7:3060)
  Port 4: SDRplay RSPdx-R2 (1df7:3060)
```

### Config Notes

- RTL-SDR trunking config in `backend/config/wavecapsdr.yaml` ~line 932
- RTL-SDR gain set to 49.6 (was 35.0)
- Sample rate: 2.4 MHz

---

## 2026-01-02 (continued): SDRplay Lock Leak Bug Found

### Investigation Summary

After server reboot, SDRplay enumeration still hangs. Investigated recent changes:

1. **SoapySDR commit 5fa9c45**: "Fix Device::make() hang by using async futures"
2. **SoapySDRPlay3 commit 7b85bb7**: "Add timeout to SDRplay API lock"

### Bug Found: SdrplayApiLockGuard Lock Leak

In `SoapySDRPlay3/SoapySDRPlay.hpp`, the `SdrplayApiLockGuard` has a critical bug:

```cpp
// BUGGY CODE (lines 63-73):
std::future<void> lockFuture = std::async(std::launch::async, []() {
    sdrplay_api_LockDeviceApi();
});
auto status = lockFuture.wait_for(std::chrono::milliseconds(timeoutMs));
if (status == std::future_status::timeout) {
    --lockDepth;
    throw std::runtime_error("SDRplay API lock timed out...");
}
```

**Problem**: When timeout occurs:
1. `lockFuture` goes out of scope, destructor does NOT wait
2. Async thread continues trying to acquire lock
3. When lock is eventually acquired, nobody releases it
4. **Lock leaked forever! All subsequent calls hang!**

### Fix Applied

Changed to spawn a cleanup thread on timeout:

```cpp
// In timeout case:
std::thread([lockFuture]() {
    try {
        lockFuture->get();  // Wait for lock to be acquired
        sdrplay_api_UnlockDeviceApi();  // Immediately release it
    } catch (...) {}
}).detach();
```

### To Complete

1. Run `cd ../SoapySDRPlay3/build && sudo make install`
2. Restart WaveCap-SDR server
3. Verify SDRplay enumeration works

### RTL-SDR Decoding Issue (Still Unresolved)

- Sync found: 93%+ rate
- Error metric: consistently 27-30 (need <15 for CRC)
- NACs decoded as random (should be 0x2e8)
- Tested PPM: -50, -30, 0, +30 - no improvement
- Tested polarity: normal and forced reversed - no improvement
- Issue appears to be symbol timing or something more fundamental

---

## 2026-01-03: SDRplay Fix Complete, RTL-SDR Issue Persists

### SDRplay Enumeration Fixed

1. **Lock leak bug fix** - Applied fix to SoapySDRPlay3 SdrplayApiLockGuard
2. **Library rpath fix** - Added `/usr/local/lib` to module rpath:
   ```bash
   sudo install_name_tool -add_rpath /usr/local/lib ~/.local/lib/SoapySDR/modules0.8-3/libsdrPlaySupport.so
   ```
3. **Created fix script** - `scripts/fix-sdrplay-full.sh`:
   - Kills SDRplay service
   - Power-cycles USB ports via uhubctl
   - Restarts service
   - Verifies enumeration
4. **Passwordless sudo** configured for fix script

### SDRplay Enumeration Working

Both RSPdx-R2 devices now enumerate:
- SDRplay Dev0 RSPdx-R2 240309F070
- SDRplay Dev1 RSPdx-R2 240305E670

### RTL-SDR Issue Still Present

Despite correct sample rate (48 kHz, 10 SPS), RTL-SDR still has ~0.1% TSBK CRC pass rate:

| Metric | Value |
|--------|-------|
| State | synced/locked |
| tsbk_attempts | 840+ |
| tsbk_crc_pass | 1 (0.1%) |
| NAC | None |

**Additional Testing:**
- Tried multiple control channel frequencies (413.3250, 413.4125 MHz)
- Scan shows sync detected on 413.4125 MHz but decode still fails
- All measured SNR values low (4-12 dB)

**Possible Causes:**
1. RTL-SDR sample rate clock drift causing symbol timing errors
2. Weak signal at this location
3. Hardware issue with this specific RTL-SDR
4. Sample rate not exactly 2.4 MHz (crystal oscillator error affects sample rate, not just frequency)

**Next Steps:**
- Try recording IQ to file and analyzing offline
- Compare with SDRTrunk on same RTL-SDR
- Test with different RTL-SDR hardware
- Try higher gain settings

---

## 2026-01-03 (continued): Static Analysis of SDRplay Trunking Failures

### Problem Summary

SDRplay trunking systems fail to start - captures get stuck in "starting" state for 45+ seconds, then fail. Pattern observed:
1. SDRplay enumeration times out or returns no devices
2. `Device::make()` times out after 10 seconds
3. `SDRplay API lock timed out` errors
4. Capture stuck in "starting" for 45+ seconds
5. System enters "searching" state but never finds control channel

### Architecture Review

The SDRplay device flow involves multiple layers of locking and subprocess isolation:

```
TrunkingSystem.start()
  └─ CaptureManager.create_capture()
       └─ Capture._run_thread()
            └─ SoapyDriver.open()
                 └─ SDRplayProxyDevice (subprocess isolation)
                      ├─ sdrplay_lock.py (file-based cross-process lock)
                      └─ SDRplayWorker subprocess
                           └─ SoapySDR.Device.make() (10s timeout)
                                └─ SoapySDRPlay3
                                     └─ SdrplayApiLockGuard (C++, 5s timeout)
                                          └─ sdrplay_api_LockDeviceApi()
```

### Key Files Analyzed

| File | Purpose |
|------|---------|
| `backend/wavecapsdr/devices/soapy.py` | SoapySDR driver, SDRplay health tracking |
| `backend/wavecapsdr/devices/sdrplay_proxy.py` | Subprocess proxy for SDRplay isolation |
| `backend/wavecapsdr/devices/sdrplay_worker.py` | Worker subprocess that opens SDRplay device |
| `backend/wavecapsdr/devices/sdrplay_lock.py` | Cross-process file-based lock |
| `backend/wavecapsdr/sdrplay_recovery.py` | Automatic service recovery |
| `backend/wavecapsdr/trunking/system.py` | Trunking system startup |

### Root Causes Identified

#### 1. SDRplay Service State Corruption

When `Device::make()` times out in a subprocess:
1. Worker subprocess is killed by parent (`process.terminate()`)
2. SDRplay API service internal state is corrupted
3. The C++ `SdrplayApiLockGuard` fix releases the *application-level* lock
4. But the service itself (`sdrplay_apiService`) may still hold internal state
5. Subsequent device opens fail because service is in bad state

**Evidence:**
- `sdrplay_worker.py:383`: `SoapySDR.Device.make(device_args, DEVICE_OPEN_TIMEOUT_US)` with 10s timeout
- When timeout occurs, subprocess is killed but service remains corrupted

#### 2. macOS Recovery May Be Failing Silently

The automatic recovery in `sdrplay_recovery.py` uses:
```python
subprocess.run(["sudo", "-n", "launchctl", "kickstart", ...], ...)
```

The `-n` flag requires passwordless sudo. If not configured for these commands, recovery silently fails and falls back to `killall` which also needs passwordless sudo.

**File:** `sdrplay_recovery.py:144-179`

#### 3. Missing USB Power Cycle in Automatic Recovery

The `fix-sdrplay-full.sh` script works because it:
1. Kills ALL service instances (`killall -9`)
2. **Power-cycles USB ports via uhubctl** (critical!)
3. Restarts service fresh
4. Verifies enumeration

The automatic recovery only restarts the service - it doesn't power-cycle USB.

**File:** `scripts/fix-sdrplay-full.sh:17-24`

#### 4. Cooldown May Be Insufficient

The file-based lock in `sdrplay_lock.py` has a 1-second cooldown:
```python
if elapsed < cooldown:
    sleep_time = cooldown - elapsed
    time.sleep(sleep_time)
```

**File:** `sdrplay_lock.py:113-116`

The SDRplay API may need more time to stabilize after device operations.

### Potential Fixes

| Fix | Description | Complexity |
|-----|-------------|------------|
| **1. Configure passwordless sudo for recovery** | Add launchctl/killall to sudoers for WaveCap | Low |
| **2. Integrate USB power cycling into recovery** | Call uhubctl in `sdrplay_recovery.py` | Medium |
| **3. Increase cooldown period** | Change from 1s to 2-3s in `sdrplay_lock.py` | Low |
| **4. Sequential trunking startup** | Start SDRplay trunking systems one at a time | Medium |
| **5. Pre-flight health check** | Verify enumeration works before Device::make() | Medium |
| **6. Call fix script from recovery** | Use the working `fix-sdrplay-full.sh` in automatic recovery | Low |

### Immediate Workaround

Before starting WaveCap-SDR server with SDRplay trunking:
```bash
sudo scripts/fix-sdrplay-full.sh
```

### Recommended Fix: Integrate Fix Script into Recovery

Modify `sdrplay_recovery.py` to call the fix script instead of individual commands:

```python
def _restart_macos(self) -> bool:
    fix_script = Path(__file__).parent.parent / "scripts" / "fix-sdrplay-full.sh"
    if fix_script.exists():
        result = subprocess.run(
            ["sudo", "-n", str(fix_script)],
            capture_output=True,
            timeout=30,
        )
        return result.returncode == 0
    # Fallback to existing logic...
```

### GitHub Issues Created

Created `../SoapySDRPlay3/GITHUB_ISSUES.md` with 6 proposed stress tests:
1. Multi-device concurrent access
2. Rapid open/close cycles
3. API lock timeout recovery
4. Long-running stability
5. Enumeration under load
6. Service crash recovery

Also copied `fix-sdrplay-full.sh` to `../SoapySDRPlay3/scripts/` and updated its README.
