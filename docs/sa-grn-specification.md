# SA-GRN (South Australian Government Radio Network) Specification

## Overview

The SA-GRN (South Australian Government Radio Network, also written SAGRN) is a statewide P25 Phase 1 trunked radio network providing mission-critical communications for South Australia's emergency services and government agencies. Operational since 2000, it is one of the largest public safety networks in the Southern Hemisphere.

## System Identifiers

| Parameter | Hex | Decimal |
|-----------|-----|---------|
| WACN | BEE00 | 781056 |
| System ID | 3DC | 988 |
| NAC | Varies by site | - |

## Technical Specifications

### Protocol

- **Standard**: APCO Project 25 (P25) Phase I
- **Air Interface**: APCO-25 Common Air Interface Exclusive
- **Infrastructure**: Motorola Astro-25

### Frequency Band

- **Band**: UHF Harmonised Government Spectrum
- **Range**: 412-422 MHz
- **Control Channels**: Multiple, distributed across sites (412.4875 - 422.1375 MHz)
- **Channel Spacing**: 12.5 kHz

### Modulation

**The SA-GRN uses LSM (Linear Simulcast Modulation), also known as CQPSK.**

This is because:
1. The network is built on Motorola Astro-25 with IP-based infrastructure
2. It covers 265,000+ km² with 240+ sites - large-area coverage requiring simulcast
3. Motorola IP Simulcast uses LSM/CQPSK modulation (not C4FM)
4. LSM allows greater site separation and better audio quality in overlap zones

When configuring receivers for SA-GRN:
- Use **LSM/CQPSK** modulation (not C4FM)
- C4FM is used by subscriber equipment (portables, mobiles) when transmitting
- Base stations transmit in CQPSK/LSM for simulcast compatibility

### Sample Rate Requirements

For reliable control channel decoding:
- Minimum: 48 kHz (per-channel)
- Recommended: 50 kHz for 10.4 samples per symbol (9600 baud)
- System sample rates of 5.5-6.0 MHz work well for wideband capture

## Network Architecture

### Coverage

- **Area**: 265,000+ km² (approximately 98% of South Australia by area)
- **Population**: 99.5%+ coverage
- **Offshore**: 20 km coastal coverage
- **Sites**: ~240 transmission sites

### Site Types

The network uses a mixture of:
- **IP Simulcast cells** - Multiple sites transmitting identical content with LSM modulation
- **Individual trunked sites** - Standalone sites with local coverage

Major sites include:
- Adelaide Main GRN Site
- Mt Lofty
- Port Adelaide (Pt Adelaide)
- One Tree Hill
- Mt Barker
- Cherry Gardens
- Gawler
- Mt Gambier

### Channel Identifiers

SA-GRN uses channel identifiers to define frequency bands. Known identifiers:

| ID | Base Freq (MHz) | Spacing (kHz) | Bandwidth (kHz) | TX Offset (MHz) |
|----|-----------------|---------------|-----------------|-----------------|
| 0 | 412.475 | 12.5 | 12.5 | 0.0 |
| 1 | 415.125 | 12.5 | 12.5 | 0.0 |
| 2 | 420.0125 | 6.25 | 12.5 | +5.2 |
| 3 | 467.5 | 6.25 | 12.5 | -10.0 |

## User Agencies

### Primary Users

- **SAPOL** - South Australia Police (encrypted)
- **SA Ambulance Service** (encrypted)
- **MFS** - Metropolitan Fire Service
- **CFS** - Country Fire Service
- **SES** - State Emergency Service
- **SA Fisheries** (encrypted)

### Encryption

- Police (SAPOL), Ambulance, and Fisheries use encrypted digital communications
- Fire services (MFS, CFS) and SES typically use unencrypted digital
- Some legacy analog operations still exist on the network

## Paging Network

SA-GRN includes a dedicated paging network for emergency dispatch:
- **Frequency**: 148.8125 MHz
- **Protocol**: POCSAG
- **Users**: Ambulance, MFS, CFS, SES dispatch

## WaveCap-SDR Configuration

### Recommended Settings

```yaml
trunking:
  systems:
    sa_grn:
      protocol: p25_phase1
      modulation: lsm          # CRITICAL: Must be LSM for simulcast
      center_hz: 415.0 MHz     # Adjust based on target control channels
      sample_rate: 6000000     # 6 Msps for wide capture
      bandwidth: 5500000
      gain: 20-30              # Adjust for your antenna/location
      wacn: 781056             # 0xBEE00
      system_id: 988           # 0x3DC
      control_channels:
        - frequency: 413.3125 MHz
          name: Adelaide Main GRN Site
        - frequency: 414.7625 MHz
          name: Mt Lofty
        # Add more as needed
```

### Modulation Selection

| Scenario | Modulation Setting |
|----------|-------------------|
| SA-GRN (simulcast) | `lsm` |
| Standard P25 (non-simulcast) | `c4fm` |

**Never use C4FM for SA-GRN** - the system uses IP Simulcast infrastructure which transmits in LSM/CQPSK. Using C4FM will result in zero TSBK decodes.

## History

- **2000**: Network launched, replacing 28 separate legacy systems
- **2000s**: Telstra constructed and initially operated the network
- **2010**: Motorola entered support contract
- **2015**: $175M AUD contract awarded to Motorola for upgrade and management
- **Present**: Ongoing expansion and modernization

## References

- [RadioReference SA-GRN Database](https://www.radioreference.com/db/sid/8145)
- [RadioReference SA-GRN Wiki](https://wiki.radioreference.com/index.php/South_Australian_Government_Radio_Network_(SA-GRN))
- [SA Scan - SA GRN Information](https://www.sascan.net.au/?page=infPages/GRN_GenInfo)
- [Wikipedia - Government Radio Networks in Australia](https://en.wikipedia.org/wiki/Government_Radio_Network_(Australia))
- [Tait Communications - C4FM vs LSM Explained](https://www.taitcommunications.com/en/about-us/news/2018/01/31/what-is-the-difference-between-c4fm-and-lsm-modulation-for-p25)

## Document History

- 2025-01-01: Initial specification created
