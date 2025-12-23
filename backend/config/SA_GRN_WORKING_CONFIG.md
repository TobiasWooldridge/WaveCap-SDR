# SA GRN Working Configuration

## What Works

After extensive debugging, the following configuration successfully receives P25 frame syncs from SA GRN in Woodcroft, South Australia.

## Key Settings

| Setting | Value | Notes |
|---------|-------|-------|
| **Modulation** | `c4fm` | C4FM (4-level FSK), NOT CQPSK/LSM |
| **Gain** | `30` | Lower gain works better than 59 |
| **SDRplay Device** | `240305E670` | May need to test both devices |
| **Antenna** | `Antenna A` | With 36-45cm telescoping antenna |
| **Center Frequency** | `415.5 MHz` | Covers 412.5 - 418.5 MHz |
| **Sample Rate** | `6 MHz` | Wide enough for all control channels |

## Control Channel Frequencies

From [SAScan.net.au](https://www.sascan.net.au) and [RadioReference](https://www.radioreference.com/db/sid/8145):

| Frequency | Site | Notes |
|-----------|------|-------|
| 415.25 MHz | Sellicks | South of Adelaide |
| 414.8875 MHz | Mt Barker | Mt Barker Summit |
| 414.75 MHz | Birdwood | McVitties Hill |
| 413.35 MHz | Hackham West | Closest to Woodcroft |
| 413.175 MHz | Bull Creek | Site 084 |
| 413.3125 MHz | Adelaide Main | CBD |

## YAML Configuration

```yaml
trunking:
  systems:
    sa_grn:
      id: sa_grn
      name: SA GRN (South Australia Government Radio Network)
      protocol: p25_phase1
      modulation: c4fm  # C4FM works, not CQPSK/LSM
      system_id: 988
      wacn: 781056
      control_channels:
        - 415.25 MHz    # Sellicks
        - 414.8875 MHz  # Mt Barker
        - 414.75 MHz    # Birdwood
        - 413.350 MHz   # Hackham West
        - 413.175 MHz   # Bull Creek
        - 413.3125 MHz  # Adelaide Main
      center_hz: 415.5 MHz
      sample_rate: 6000000
      device_id: driver=sdrplay,label=SDRplay Dev0 RSPdx-R2 240305E670,serial=240305E670
      gain: 30  # Lower gain works better
      antenna: Antenna A
      device_settings:
        rfnotch_ctrl: 'false'
        dabnotch_ctrl: 'false'
      control_channel_timeout: 30  # Seconds per frequency
```

## Debugging Notes

### What Didn't Work
- High gain (59 dB) - causes signal issues
- CQPSK/LSM modulation - uniform symbol distribution
- Wrong control channel frequencies from initial research

### What Fixed It
1. **Lower gain (30 dB)** - raw_mean went from 0.0005 to 0.003
2. **C4FM modulation** - standard P25 Phase 1 modulation
3. **Correct control channel frequencies** - from SAScan.net.au

### Verification
- Frame syncs: `syncs=129+`
- Frame types detected: LDU1 (voice), TSDU (control)
- Symbol distribution: Non-uniform when on active frequency

## Current Status (Dec 2024)

**Working:**
- Frame sync detection: 900+ syncs detected
- TSDU (control channel) frames detected
- LDU1 (voice) frames detected
- C4FM symbol thresholds: Fixed at [-2.0, 0.0, 2.0]

**Code Fixes Applied:**

1. **FM Discriminator Scaling** (`decoders/p25.py:546`):
   - Changed from `max_deviation=1800` to `deviation_hz=600`
   - Now correctly maps ±1800 Hz to ±3.0 symbol values

2. **Adaptive Thresholds** (`decoders/p25.py:678-722`):
   - Changed from quartile-based to fixed thresholds
   - Thresholds converge toward [-2.0, 0.0, 2.0] for C4FM
   - Noise detection (std>10) prevents threshold drift

**Remaining Issue:**
- TSBK FEC still has ~23 bit errors (max correctable ~11)
- Symbol std is ~8-13 instead of expected ~2.5
- Likely due to weak signal or symbol timing issues

**Possible Next Steps:**
1. Improve symbol timing recovery (Gardner TED tuning)
2. Add AGC before FM discriminator to normalize signal level
3. Compare with dsp/p25/c4fm.py constellation gain approach

## Sources

- [RadioReference SA-GRN](https://www.radioreference.com/db/sid/8145)
- [SAScan SA GRN Sites](https://www.sascan.net.au/?page=infPages/GRN_Sites)
- [RadioReference Forums - SAGRN Site IDs](https://forums.radioreference.com/threads/sagrn-site-ids.281911/)
