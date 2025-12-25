# P25 Trunking TODOs (WaveCap-SDR)

This file summarizes the remaining P25 trunking gaps and proposed fixes.

## High Priority Gaps

- Phase II/TDMA is not end-to-end.
  - Evidence: `backend/wavecapsdr/decoders/p25.py:1382` (Phase I decoder),
    `backend/wavecapsdr/trunking/system.py:215`,
    `backend/wavecapsdr/trunking/voice_channel.py:169`
  - Fix: Add TDMA framing/timeslot selection in the demod/decoder path and
    propagate timeslot into `VoiceChannel`/`VoiceDecoder`. If Phase II is not
    ready, align API/docs to "Phase I only" until slots are supported.

## Medium Priority Gaps

- Recording-related config is defined but not applied in the trunking path.
  `record`, `record_unknown`, `min_call_duration`, `recording_path`,
  `audio_gain`, `squelch_db` do not gate recording or file output.
  - Evidence: `backend/wavecapsdr/trunking/config.py:126`,
    `backend/wavecapsdr/trunking/system.py:1055`,
    `backend/wavecapsdr/trunking/voice_channel.py:168`
  - Fix: Implement per-call recording, honor `record_*` flags, apply
    `audio_gain/squelch_db`, and enforce `min_call_duration`.
- Location cache exists, but LRRP/ELC data is not wired into decoder flow.
  - Evidence: `backend/wavecapsdr/trunking/system.py:1537`,
    `backend/wavecapsdr/decoders/lrrp.py:266`
  - Fix: Decode LRRP/ELC from PDU/LDU frames and call
    `TrunkingSystem.update_radio_location`. This requires:
    1. Extracting Link Control words from LDU1/LDU2 frames in p25.py
    2. Parsing ELC GPS data via lrrp.py's `decode_gps_from_elc()`
    3. Implementing PDU frame parsing for LRRP packets

## Tests to Add

- Phase II timeslot handling tests (if/when implemented).

## Notes / Open Questions

- Clarify whether Phase II support is expected to be functional now, or whether
  docs/API should be scoped to Phase I until TDMA/timeslot logic is implemented.
- If Phase II is kept: decide how to select timeslot (config, grant parsing,
  or auto-detect) and how to expose it in API events.
