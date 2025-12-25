# P25 Outstanding Issues (WaveCap-SDR)

This file summarizes the remaining P25 trunking gaps and proposed fixes.

## High Priority

- **Phase II/TDMA is not end-to-end**
  - Evidence: `backend/wavecapsdr/decoders/p25.py:1382` (Phase I decoder),
    `backend/wavecapsdr/trunking/system.py:215`,
    `backend/wavecapsdr/trunking/voice_channel.py:169`
  - Fix: Add TDMA framing/timeslot selection in the demod/decoder path and
    propagate timeslot into `VoiceChannel`/`VoiceDecoder`. If Phase II is not
    ready, align API/docs to "Phase I only" until slots are supported.
- **Capture start/restart can conflict with trunking ownership**
  - Evidence: `backend/wavecapsdr/api.py:1157`, `backend/wavecapsdr/api.py:1217`
  - Fix: Block starting or restarting a capture if the device is owned by a
    trunking system, and add an explicit guard to `restart_capture` similar to
    `start_capture`/`stop_capture`.

## Medium Priority

- **Decoder stop scheduling can drop stop coroutines**
  - Evidence: `backend/wavecapsdr/capture.py:541`
  - Fix: Only return early after `run_coroutine_threadsafe` if the scheduling
    succeeds, otherwise fall through to a running loop or `asyncio.run`.

## Low Priority

- **`TrunkingSystem.process_tsbk` uses wrong parser signature**
  - Evidence: `backend/wavecapsdr/trunking/system.py:1061`
  - Fix: Update to `TSBKParser.parse(opcode, mfid, data)` or remove unused path.

## Open Questions

- Clarify whether Phase II support is expected to be functional now, or whether
  docs/API should be scoped to Phase I until TDMA/timeslot logic is implemented.
- If Phase II is kept: decide how to select timeslot (config, grant parsing,
  or auto-detect) and how to expose it in API events.
- Should non-trunking captures be blocked from starting if a trunking capture
  already owns the same device?
