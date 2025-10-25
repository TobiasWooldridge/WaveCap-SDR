Configuration — WaveCap‑SDR

Location: `config/wavecapsdr.yaml` by default. Overrides via env vars with prefix `WAVECAPSDR__SECTION__KEY`.

Server
- `server.bind_address` (string, default `127.0.0.1`): Bind address.
- `server.port` (int, default `8087`): HTTP port.
- `server.auth_token` (string, optional): If set, HTTP requires `Authorization: Bearer <token>`. WebSocket accepts header or `?token=` query.

Stream
- `stream.default_transport` (enum `ws`|`http`, default `ws`): Default stream transport.
- `stream.default_format` (enum `iq16`|`f32`|`pcm16`, default `iq16`): Default frame format.
- `stream.default_audio_rate` (int, default `48000`): Default audio sample rate for demod channels.

Limits
- `limits.max_concurrent_captures` (int, default 2): Maximum concurrent device captures.
- `limits.max_channels_per_capture` (int, default 8): Maximum channels within a single capture.
- `limits.max_sample_rate` (int, optional): Upper limit on requested sample rate.

Device
- `device.driver` (enum `soapy`|`fake`, default `soapy`): Driver selection. Use `soapy` for RTL‑SDR and RSPdx‑r2 via SoapySDR; `fake` for tests.
- `device.device_args` (string, optional): Soapy device args, e.g. `driver=rtlsdr` or `driver=sdrplay,serial=...`.

Presets (for harness and quick start)
- `presets.<name>.center_hz` (float): RF center frequency in Hz.
- `presets.<name>.sample_rate` (int): Sample rate in samples per second.
- `presets.<name>.offsets` (array<float>): Channel offsets relative to center in Hz.
- Example:
  - `presets.kexp.center_hz: 90300000`
  - `presets.kexp.sample_rate: 2000000`
  - `presets.kexp.offsets: [0.0]`

Environment overrides examples
- `WAVECAPSDR__SERVER__PORT=8089`
- `WAVECAPSDR__DEVICE__DRIVER=fake`
- `WAVECAPSDR__SERVER__AUTH_TOKEN=secret123`
