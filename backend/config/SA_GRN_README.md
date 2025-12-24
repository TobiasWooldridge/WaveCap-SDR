# SA GRN (South Australia Government Radio Network) Configuration

## Overview

The SA-GRN is a P25 Phase 1 trunked radio network serving South Australian emergency services and government agencies. It has been operational since 2000 and covers over 265,000 square kilometres with approximately 240 sites.

## System Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Protocol | P25 Phase 1 | APCO-25 Common Air Interface |
| Modulation | C4FM | Standard 4-level FSK |
| System ID | 0x3DC (988) | |
| WACN | 0xBEE00 (781056) | |
| NAC (Adelaide Main) | 0x3D1 (977) | Site-specific |

## Control Channel Frequencies

Frequencies verified from RadioReference.com (December 2024):

### Adelaide Main Site (Site 002)
- **412.5625 MHz** - Primary control channel

### Trott Park Site
- 412.825 MHz
- 412.95 MHz
- 413.075 MHz
- 413.325 MHz
- 413.45 MHz
- 415.325 MHz
- 415.95 MHz
- 416.45 MHz
- 416.95 MHz

## SDR Configuration

- **Center Frequency**: 415.5 MHz
- **Sample Rate**: 6 MHz (covers 412.5 - 418.5 MHz)
- **Recommended Gain**: 40-59 dB (adjust based on local conditions)

## Sources

1. **RadioReference SA-GRN System Profile**
   - URL: https://www.radioreference.com/db/sid/8145
   - Contains system-wide information and site listings

2. **RadioReference Adelaide Main Site**
   - URL: https://www.radioreference.com/db/site/32061
   - Site 002, NAC 3D1, 25 frequencies in 412-419 MHz range

3. **RadioReference Forums - SAGRN Site IDs**
   - URL: https://forums.radioreference.com/threads/sagrn-site-ids.281911/
   - Community-contributed frequency and LCN data for Trott Park site

4. **SAScan - SA Government Radio Network**
   - URL: https://www.sascan.net.au/?page=infPages/GRN_GenInfo
   - General system information and coverage details

5. **Wikipedia - Government Radio Networks in Australia**
   - URL: https://en.wikipedia.org/wiki/Government_Radio_Network_(Australia)
   - Background on Australian government radio networks

## Notes

- The SA-GRN uses Motorola Astro-25 network equipment
- Voice channels may be in the 800 MHz range (separate from control channels)
- The system includes a statewide paging network on 148.8125 MHz
- Coverage extends 20km offshore along the state's coast

## Last Updated

December 2024
