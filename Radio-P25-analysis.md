# Radio + P25 Review (Functionality + Trunking Integration)

This document summarizes findings from a code review of the radio system
(captures/channels) and how it integrates with the P25 trunking system.

## Scope Reviewed
- Radio system: captures, channels, P25 decode path, voice following
  - `backend/wavecapsdr/capture.py`
  - `backend/wavecapsdr/api.py`
- Trunking system control-channel + voice follow integration
  - `backend/wavecapsdr/trunking/system.py`
  - `backend/wavecapsdr/trunking/control_channel.py`
  - `backend/wavecapsdr/decoders/p25.py`

## Resolved Issues

The following issues from the original review have been fixed:

1. **P25 control-channel voice following dead code** - The `on_grant` callback
   was wired but never invoked. Fixed by removing the wiring and adding comments
   noting that trunking systems use their own VoiceRecorder pool.

2. **Trunking captures could be stopped by API** - Starting a normal capture
   could auto-stop a trunking capture on the same device. Fixed by adding
   `trunking_system_id` field to captures and protecting trunking-owned captures
   from start/stop/delete via the capture API.

3. **Voice-following duplication** - Trunking control channels were created with
   voice-following enabled by default. Fixed by passing
   `enable_voice_following=False` when trunking creates its control channel.

4. **Control channel decode duplication** - Both P25 decoder and
   ControlChannelMonitor were processing TSBKs. Fixed by removing the channel's
   `on_tsbk` wiring - now only ControlChannelMonitor decodes (supports Phase I/II).

5. **Roaming recenter offsets** - Active voice recorders kept old offsets after
   roaming. Fixed by adding `VoiceRecorder.update_center_frequency()` that's
   called when roaming changes the capture center.

6. **Channel decoder shutdown** - Used `asyncio.run` as fallback which fails
   under an active event loop. Fixed by trying `get_running_loop()` first before
   falling back to `asyncio.run`.

## Open Questions / Scope Decisions

- Should P25 voice following exist at the capture layer at all?
  - **Decision**: No - trunking systems manage their own voice channels via
    VoiceRecorder pool. The capture-level `_handle_trunking_grant` infrastructure
    remains but is not wired up.

- Should trunking captures be locked from direct capture start/stop/delete?
  - **Decision**: Yes - the API now returns 409 Conflict if attempting to
    start/stop/delete a capture owned by a trunking system.
