# Subprocess Worker Architecture for SDRplay Multi-Device Support

## Problem Statement

The SDRplay API v3.15 has a fundamental limitation: **only ONE device can be selected per process**. When `sdrplay_api_SelectDevice()` is called for a second device, it fails with "Device already selected". This is not a bug in SoapySDRPlay3 - it's an SDRplay API design decision.

**Current behavior:**
- First capture with SDRplay device works fine
- Second capture with another SDRplay device fails with "Init() failed: sdrplay_api_Fail"
- The SelectDevice call fails at sdrplay_api.cpp:409

## Solution: Subprocess Worker per SDRplay Device

Each SDRplay device will run in its own isolated subprocess, bypassing the API's single-device limitation. IQ samples will be transferred via shared memory (zero-copy) with command/control via multiprocessing pipes.

## Architecture Design

### Current Data Flow (Single Process)
```
Capture._run_thread()  →  device.read()  →  IQ samples (numpy array)
                       →  process channels  →  broadcast to WebSocket
```

### New Data Flow (Subprocess Workers)
```
Main Process:
  Capture._run_thread()  →  SharedMemory.read()  →  IQ samples (zero-copy)
                         →  process channels     →  broadcast to WebSocket

Subprocess Worker:
  SDRplayWorker.run()  →  SoapySDR.Device()  →  read IQ samples
                       →  SharedMemory.write()  →  signal main process
```

## Implementation Plan

### 1. Create SDRplayWorker class
**File:** `backend/wavecapsdr/devices/sdrplay_worker.py`

```python
class SDRplayWorker:
    """Subprocess worker for isolated SDRplay device control.

    Runs in a separate process to bypass SDRplay API single-device limit.
    Communicates via:
    - SharedMemory for IQ data (zero-copy transfer)
    - Pipe for commands (configure, start, stop)
    - Pipe for status updates (overflow, errors)
    """

    def __init__(self, device_args: str, shm_name: str, cmd_pipe: Connection, status_pipe: Connection):
        self.device_args = device_args
        self.shm_name = shm_name
        self.cmd_pipe = cmd_pipe
        self.status_pipe = status_pipe

    def run(self):
        """Main loop: process commands, stream IQ to shared memory."""
        # Open device (isolated API context)
        sdr = SoapySDR.Device(self.device_args)
        stream = None

        while True:
            # Check for commands
            if self.cmd_pipe.poll():
                cmd = self.cmd_pipe.recv()
                if cmd['type'] == 'configure':
                    self._configure(sdr, cmd)
                elif cmd['type'] == 'start':
                    stream = self._start_stream(sdr)
                elif cmd['type'] == 'stop':
                    self._stop_stream(sdr, stream)
                    stream = None
                elif cmd['type'] == 'shutdown':
                    break

            # Stream IQ samples if active
            if stream:
                samples, overflow = self._read_samples(stream)
                self._write_to_shm(samples)
                if overflow:
                    self.status_pipe.send({'type': 'overflow'})
```

### 2. Create SDRplayProxyDevice class
**File:** `backend/wavecapsdr/devices/sdrplay_proxy.py`

```python
class SDRplayProxyDevice(Device):
    """Proxy for SDRplay device running in subprocess.

    Presents the same Device interface but routes all calls
    to the subprocess worker via IPC.
    """

    def __init__(self, info: DeviceInfo, device_args: str):
        self.info = info
        self.device_args = device_args
        self._worker_process = None
        self._shm = None
        self._cmd_pipe = None
        self._status_pipe = None

    def _start_worker(self):
        """Launch subprocess worker."""
        # Create shared memory for IQ buffer
        shm_size = 262144 * 8  # 256K complex64 samples
        self._shm = SharedMemory(create=True, size=shm_size)

        # Create IPC pipes
        self._cmd_pipe, worker_cmd = Pipe()
        worker_status, self._status_pipe = Pipe()

        # Launch worker process
        self._worker_process = Process(
            target=sdrplay_worker_main,
            args=(self.device_args, self._shm.name, worker_cmd, worker_status)
        )
        self._worker_process.start()

    def configure(self, **kwargs):
        self._cmd_pipe.send({'type': 'configure', **kwargs})

    def start_stream(self) -> StreamHandle:
        self._cmd_pipe.send({'type': 'start'})
        return SDRplayProxyStream(self._shm, self._status_pipe)
```

### 3. Create SDRplayProxyStream class
**File:** `backend/wavecapsdr/devices/sdrplay_proxy.py`

```python
class SDRplayProxyStream(StreamHandle):
    """Stream handle that reads from subprocess's shared memory."""

    def __init__(self, shm: SharedMemory, status_pipe: Connection):
        self._shm = shm
        self._status_pipe = status_pipe
        self._read_idx = 0

    def read(self, num_samples: int) -> tuple[np.ndarray, bool]:
        """Read IQ samples from shared memory (zero-copy view)."""
        # Create numpy array backed by shared memory
        buffer = np.ndarray(
            shape=(num_samples,),
            dtype=np.complex64,
            buffer=self._shm.buf
        )

        # Check for overflow status
        overflow = False
        while self._status_pipe.poll():
            msg = self._status_pipe.recv()
            if msg['type'] == 'overflow':
                overflow = True

        return buffer.copy(), overflow  # Copy since worker may overwrite
```

### 4. Modify SoapyDriver.open() to use proxy for SDRplay
**File:** `backend/wavecapsdr/devices/soapy.py`

```python
def open(self, id_or_args: Optional[str] = None) -> Device:
    args = id_or_args or self._cfg.device_args or ""

    # For SDRplay, use subprocess proxy to bypass API limitation
    if "sdrplay" in args.lower():
        from .sdrplay_proxy import SDRplayProxyDevice

        # Build DeviceInfo (can query via subprocess)
        info = self._build_device_info_subprocess(args)

        # Return proxy that manages subprocess lifecycle
        return SDRplayProxyDevice(info, args)
    else:
        # Non-SDRplay devices use direct SoapySDR access
        sdr = SoapySDR.Device(args)
        # ... existing code ...
```

### 5. SharedMemory Ring Buffer Protocol

```
Ring Buffer Layout (per worker):
┌────────────────────────────────────────────────────────────┐
│ Header (64 bytes)                                          │
│  - write_idx: uint64  (samples written by worker)          │
│  - read_idx: uint64   (samples consumed by main process)   │
│  - sample_rate: uint32                                     │
│  - flags: uint32 (overflow, error)                         │
│  - reserved: padding                                       │
├────────────────────────────────────────────────────────────┤
│ Data Buffer (2MB = 262144 * 8 bytes)                       │
│  - 262144 complex64 samples                                │
│  - Circular write/read with wrap-around                    │
└────────────────────────────────────────────────────────────┘
```

### 6. Cleanup and Lifecycle Management

The proxy device must properly manage subprocess lifecycle:
- Start worker on first configure() or start_stream()
- Monitor worker health via status pipe
- Terminate worker gracefully on close()
- Handle worker crash/timeout with automatic restart

## Files to Create/Modify

1. **NEW:** `backend/wavecapsdr/devices/sdrplay_worker.py`
   - `SDRplayWorker` class (subprocess entry point)
   - `sdrplay_worker_main()` function
   - Shared memory protocol implementation

2. **NEW:** `backend/wavecapsdr/devices/sdrplay_proxy.py`
   - `SDRplayProxyDevice` class
   - `SDRplayProxyStream` class
   - Worker process management

3. **MODIFY:** `backend/wavecapsdr/devices/soapy.py`
   - Update `SoapyDriver.open()` to use proxy for SDRplay
   - Add `_build_device_info_subprocess()` helper

4. **MODIFY:** `backend/wavecapsdr/capture.py`
   - No changes needed - proxy is transparent to Capture

## Testing Strategy

1. **Unit tests:**
   - Test shared memory ring buffer read/write
   - Test worker subprocess lifecycle
   - Test proxy device interface

2. **Integration tests:**
   - Single SDRplay device via proxy
   - Two SDRplay devices simultaneously
   - Mixed SDRplay + RTL-SDR captures

3. **Manual testing:**
   - Start two captures with different SDRplay RSPdx-R2 devices
   - Verify both show spectrums updating
   - Verify audio streams work for channels on both

## Estimated Complexity

- **sdrplay_worker.py:** ~200 lines
- **sdrplay_proxy.py:** ~300 lines
- **soapy.py modifications:** ~50 lines
- **Total:** ~550 lines of new/modified code

## Rollback Plan

The subprocess approach is opt-in for SDRplay only:
- RTL-SDR and other devices are unaffected
- If issues arise, simply remove the proxy check in `SoapyDriver.open()`
- Existing single-SDRplay-device workflows continue to work
