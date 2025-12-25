# Debugging P25 (Control Channel + Trunking)

This is a repeatable approach to isolate RF issues from demod/FEC and trunking
integration problems. Keep it small, reproducible, and instrumented.

## Goals
- Determine whether failures are RF/antenna, demod (timing/phase/equalizer),
  or FEC/bit parsing.
- Keep changes incremental and reversible.
- Prefer offline IQ fixtures to avoid hardware variability.

## Baseline Repro (offline first)
1. Use a known good IQ capture from `reference-captures/` (or capture a new
   short file with clear sync).
2. Run the decode path against the file with stable settings (no tuning
   changes). Keep config consistent.
3. Confirm frame sync and NID extraction before touching FEC.

## Demod Isolation Checklist
- **Sample rate sanity**: Confirm decimation lands on 4800 sym/s (or an exact
  multiple feeding the symbol timing loop).
- **Symbol timing**: Log timing error and verify it converges. Excess jitter
  usually points to loop gains or incorrect interpolator steps.
- **Carrier tracking**: Log CFO estimate and lock time. If CFO oscillates,
  reduce loop bandwidth or validate NCO sign/units.
- **Dibit mapping**: Verify mapping and polarity. Try all 4 phase rotations
  and polarity flips against a known-good capture.
- **Status symbols**: Ensure status symbol stripping and interleaving order
  match P25 specs (common source of TSBK errors with good sync).
- **LSM equalization** (CQPSK/LSM only): Verify equalizer taps are applied,
  and compare with a C4FM demod path on the same IQ.

## FEC + TSBK Validation
- **NID/BCH**: Confirm BCH decode success before TSBK. If BCH is flaky, fix
  demod/sync first.
- **Trellis decode**: Count bit errors per block; dump pre/post trellis
  streams for a few frames.
- **CRC/TSBK parse**: Validate deinterleave and CRC. If CRC fails but raw
  errors are low, check parser bit ordering.

## Trunking Integration Checks
- Ensure the control channel capture owns the SDR (avoid starting another
  capture on the same device).
- Verify control channel updates (grant/explicit/UU) are applied to channel
  tracking and voice retune logic.
- Confirm voice recorders retune when capture center frequency changes.

## Instrumentation to Add (short-lived)
- CFO estimate, timing error, and EVM per second.
- Trellis error counts and CRC pass rates.
- Debug dumps of dibits and interleaver output (rate-limited).

## Suggested Experiments
1. Compare C4FM vs CQPSK demod output on the same IQ.
2. Sweep symbol timing and carrier loop bandwidths (small increments).
3. Force each phase rotation/polarity for a few frames and log CRC hits.
4. Run with and without LSM equalization to isolate multipath sensitivity.

## Tooling Notes
- Use the timeout wrapper for long-running commands:
  `scripts/run-with-timeout.sh --seconds 60 -- <command>`
- If probing hardware, use the Soapy wrappers:
  `scripts/soapy-find.sh` or `scripts/soapy-probe.sh driver=sdrplay`

## Exit Criteria
- Stable NID/BCH and TSBK CRC pass rate on offline IQ.
- Control channel grants reliably schedule voice channels.
- Voice decode stable after retunes and capture center changes.
