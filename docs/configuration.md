Configuration — WaveCap‑SDR

Location: `config/wavecapsdr.yaml` by default, with optional local overrides in `config/wavecapsdr.local.yaml` (gitignored). The server loads the base config first, then overlays the local file. Overrides via env vars with prefix `WAVECAPSDR__SECTION__KEY`.
Whenever the server persists changes (e.g., when tuning a preset through the UI) it first writes a sibling backup `wavecapsdr.local.yaml.bak` (or `wavecapsdr.yaml.bak` if no local file is used), so you can recover prior settings if a save goes sideways.

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
- `device.driver` (enum `soapy`|`fake`|`rtl`, default `soapy`): Driver selection. Use `soapy` for RTL‑SDR and RSPdx‑r2 via SoapySDR; `rtl` for direct RTL‑SDR driver; `fake` for tests.
- `device.device_args` (string, optional): Soapy device args, e.g. `driver=rtlsdr` or `driver=sdrplay,serial=...`.
- `device.show_fake_device` (bool, default `false`): Show fake/test device even when real devices are available (for development).

Recovery
- `recovery.sdrplay_service_restart_enabled` (bool, default `true`): Enable automatic SDRplay service restart on failure.
- `recovery.sdrplay_service_restart_cooldown` (float, default `60.0`): Minimum seconds between service restart attempts.
- `recovery.max_service_restarts_per_hour` (int, default `5`): Maximum service restarts allowed per hour.
- `recovery.sdrplay_operation_cooldown` (float, default `0.5`): Minimum seconds between SDRplay device operations (configure, close).
- `recovery.iq_watchdog_enabled` (bool, default `true`): Enable IQ sample watchdog (restart capture if no samples).
- `recovery.iq_watchdog_timeout` (float, default `30.0`): Watchdog timeout in seconds.

Antennas (device-specific presets)
- `antennas.<device_id>.<preset_name>` (string): Preferred antenna for a specific device and preset combination.
- Example:
  - `antennas.sdrplay_240309F070.fm_broadcast: "Ant B"`
  - `antennas.sdrplay_240309F070.vhf_amateur: "Ant C"`

Device Names
- `device_names.<device_id>` (string): Custom human-readable name for a device.
- Example:
  - `device_names."driver=rtlsdr,serial=00000001": "RTL-SDR Blog V4"`
  - `device_names."driver=sdrplay,serial=240309F070": "RSPdx-R2 F070"`

Recipes
- `recipes.<name>.name` (string): Display name for the recipe.
- `recipes.<name>.description` (string): Help text describing the recipe.
- `recipes.<name>.category` (string): Category grouping (e.g., "Marine", "Aviation", "Broadcast").
- `recipes.<name>.center_hz` (float): Default RF center frequency in Hz.
- `recipes.<name>.sample_rate` (int): Sample rate in samples per second.
- `recipes.<name>.gain` (float, optional): Default RF gain in dB.
- `recipes.<name>.bandwidth` (float, optional): Default hardware filter bandwidth in Hz.
- `recipes.<name>.channels` (array, optional): List of channel definitions to create.
  - `channels[].offset_hz` (float): Channel offset from center in Hz.
  - `channels[].name` (string): Channel display name.
  - `channels[].mode` (string, default `"wbfm"`): Demodulation mode (`wbfm`, `nbfm`, `am`, `sam`, `ssb`, `raw`, `p25`, `dmr`, `nxdn`, `dstar`, `ysf`).
  - `channels[].squelch_db` (float, default `-60`): Squelch level in dB.
  - `channels[].enable_pocsag` (bool, default `false`): Enable POCSAG pager decoding (NBFM only).
  - `channels[].pocsag_baud` (int, default `1200`): POCSAG baud rate (512, 1200, 2400).
- `recipes.<name>.allow_frequency_input` (bool, default `false`): Allow user to customize frequency.
- `recipes.<name>.frequency_label` (string, optional): Label for frequency input field.

Trunking Systems
- `trunking.systems.<system_id>` (dict): Trunking system configuration.
  - See `backend/wavecapsdr/trunking/` for detailed configuration options.
  - P25 Phase 1 trunking is supported with voice channel following.
  - SA-GRN P25 Phase 1 spec: `docs/sa-grn-p25-spec.md`.
  - Control channel frequencies must fall within `center_hz ± (sample_rate / 2)` and `sample_rate` must be > 0.
  - `channel_identifiers` (dict|array, optional): Seed IDEN_UP channel band data so voice grants can resolve even when CRC is weak.
    - Map form (keyed by identifier 0-15):
      - `base_freq_mhz` (float): Base frequency in MHz.
      - `channel_spacing_khz` (float): Channel spacing in kHz.
      - `bandwidth_khz` (float, default 12.5): Channel bandwidth in kHz.
      - `tx_offset_mhz` (float, default 0.0): TX offset in MHz.
    - Values are merged with cached IDEN_UP data persisted under `~/.wavecapsdr/trunking_state/<system_id>.json`.

Captures (auto-start on boot)
- `captures[]` (array): List of captures to auto-start on server launch.
  - `preset` (string): Name of preset to use.
  - `device_id` (string, optional): Specific device to use (if omitted, uses any available device).

Driver fallback
- When `device.driver` is `soapy` but no devices are detected (or Soapy fails to initialize), the server automatically falls back to the Fake driver so you can exercise the UI/API without hardware.
- Capture entries stay resident even if the requested SDR cannot be opened at startup (permissions, unplugged device, etc.). They surface with state `failed` and will retry automatically once the radio is available, so configs are never discarded silently.

Presets (for harness and quick start)
- `presets.<name>.center_hz` (float): RF center frequency in Hz.
- `presets.<name>.sample_rate` (int): Sample rate in samples per second.
- `presets.<name>.offsets` (array<float>, optional): Channel offsets relative to center in Hz.
- `presets.<name>.gain` (float, optional): RF gain in dB.
- `presets.<name>.bandwidth` (float, optional): Hardware filter bandwidth in Hz.
- `presets.<name>.ppm` (float, optional): Frequency correction in PPM.
- `presets.<name>.antenna` (string, optional): Antenna selection (e.g., "Antenna A", "Antenna B").
- `presets.<name>.squelch_db` (float, optional): Default squelch level in dB.
- `presets.<name>.device_settings` (dict, optional): SoapySDR device-specific settings (key-value pairs passed to writeSetting).
- `presets.<name>.element_gains` (dict, optional): Per-element gains (e.g., `{"LNA": 20, "VGA": 15}`).
- `presets.<name>.stream_format` (string, optional): Stream format preference ("CF32", "CS16", "CS8").
- `presets.<name>.dc_offset_auto` (bool, default `true`): Enable automatic DC offset correction.
- `presets.<name>.iq_balance_auto` (bool, default `true`): Enable automatic IQ balance correction.
- Example:
  - `presets.kexp.center_hz: 90300000`
  - `presets.kexp.sample_rate: 2000000`
  - `presets.kexp.offsets: [0.0]`
  - `presets.kexp.gain: 21.1`
  - `presets.kexp.antenna: "Antenna B"`

Environment overrides examples
- `WAVECAPSDR__SERVER__PORT=8089`
- `WAVECAPSDR__DEVICE__DRIVER=fake`
- `WAVECAPSDR__SERVER__AUTH_TOKEN=secret123`

Harness / CLI helpers (not config-backed)
- `python -m wavecapsdr.harness --message-spec <file>`: encode a JSON/YAML IMBE message spec to `harness_out/message.bin` and `harness_out/message.wav` (overridable via `--message-bytes-out` / `--message-wav-out`). Use `--message-stream-ws <url>` and `--message-chunk-ms <ms>` to push PCM16 chunks into an existing WebSocket audio consumer.
- `python -m wavecapsdr.cli message --spec <file> --out-bytes <file> [--out-wav <file>] [--stream-ws <url>]`: standalone encoder variant without starting the harness.
