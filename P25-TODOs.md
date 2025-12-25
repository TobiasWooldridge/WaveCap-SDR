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

## Open Questions

- Clarify whether Phase II support is expected to be functional now, or whether
  docs/API should be scoped to Phase I until TDMA/timeslot logic is implemented.
- If Phase II is kept: decide how to select timeslot (config, grant parsing,
  or auto-detect) and how to expose it in API events.
