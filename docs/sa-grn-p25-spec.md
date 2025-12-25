# SA-GRN P25 Phase 1 Radio Spec

## Scope
This document captures SA-GRN (South Australian Government Radio Network)
Project 25 Phase I radio parameters and control-channel-capable frequencies
using only public internet sources.

Primary sources used here are RadioReference (system + sites tables) and
SAScan (general network information).

## System Summary (RadioReference)
- System Name: South Australian Government Radio Network (SA-GRN) Project 25
- Location: Various, VIC SA NSW
- System Type: Project 25 Phase I
- System Voice: APCO-25 Common Air Interface Exclusive
- System ID: Sysid: 3DC WACN: BEE00
- Last Updated: March 22, 2025, 12:34 pm UTC   [Changed Site # 072 (Lochiel) to 72 (Lochiel)]

## Phase 1 Trunking Notes (RadioReference)
- System Type: Project 25 Phase I.
- System Voice: APCO-25 Common Air Interface Exclusive.
- WACN/SysID: BEE00 / 3DC.
- Control-channel-capable frequencies are marked with a trailing "c" on the
  RadioReference Sites and Frequencies table (control channel can rotate).

## Phase 1 Air Interface (Wikipedia)
- Phase 1 radios use C4FM modulation at 4,800 baud (2 bits per symbol),
  yielding 9,600 bits per second total channel throughput.
- Receivers designed for C4FM can also demodulate CQPSK signals.

## System Context (SAScan)
- The SA-GRN is a shared P25 Phase 1 trunked radio network serving emergency
  services and other agencies. (SAScan General Network Information)
- It is part of South Australia's critical infrastructure and has been in use
  since 2000. (SAScan General Network Information)
- The network provides wide-area communications and interoperability between
  agencies. (SAScan General Network Information)
- SA-GRN uses some 240 sites across metropolitan and country South Australia to
  meet coverage requirements. (SAScan General Network Information)
- Simplex services are available as part of SA-GRN. (SAScan General Network Information)

## WaveCap-SDR Trunking Config Mapping (Online Sources Only)
This table maps SA-GRN/P25 facts from public sources to WaveCap trunking
config fields. Items marked "Not specified" are not provided by the sources
listed in this document.

| WaveCap config option | Online-sourced detail | Source |
|---|---|---|
| `trunking.systems.<id>.protocol` | P25 Phase I | RadioReference system page |
| `trunking.systems.<id>.modulation` | Phase 1 uses C4FM; CQPSK is compatible | Wikipedia (Project 25) |
| `trunking.systems.<id>.system_id` | Sysid 3DC | RadioReference system page |
| `trunking.systems.<id>.wacn` | WACN BEE00 | RadioReference system page |
| `trunking.systems.<id>.control_channels` | Control-channel-capable frequencies listed in RR sites table | RadioReference sites table |
| `trunking.systems.<id>.talkgroups` | Talkgroup IDs and labels listed on RR system page | RadioReference system page |
| `trunking.systems.<id>.nac` | Site-specific; RR site pages may list NAC (often N/A) | RadioReference site pages |
| `trunking.systems.<id>.center_hz` | Not specified in public sources | N/A |
| `trunking.systems.<id>.sample_rate` | Not specified in public sources | N/A |
| `trunking.systems.<id>.bandwidth` | Not specified in public sources | N/A |
| `trunking.systems.<id>.gain` | Not specified in public sources | N/A |
| `trunking.systems.<id>.antenna` | Not specified in public sources | N/A |
| `trunking.systems.<id>.device_id` | Not specified in public sources | N/A |
| `trunking.systems.<id>.device_settings` | Not specified in public sources | N/A |
| `trunking.systems.<id>.control_channel_timeout` | Not specified in public sources | N/A |
| `trunking.systems.<id>.max_voice_recorders` | Not specified in public sources | N/A |
| `trunking.systems.<id>.min_call_duration` | Not specified in public sources | N/A |
| `trunking.systems.<id>.squelch_db` | Not specified in public sources | N/A |
| `trunking.systems.<id>.record_unknown` | Not specified in public sources | N/A |
| `trunking.systems.<id>.recording_path` | Not specified in public sources | N/A |

## Control-Channel-Capable Frequencies (RadioReference)
The list below uses the RadioReference site table. Frequencies marked
with a trailing "c" on that page are control-channel-capable; actual
control channels may rotate among them. Data retrieved: 2025-12-25 05:22 UTC.

### Control Channels by Site (RR)
| RFSS | Site | Name | County | Control Channels (MHz) |
|------|------|------|--------|------------------------|
| 1 (1) | 001 (1) | Mt lofty | Statewide, SA | 414.7625, 415.7625, 417.7625, 468.0000 |
| 1 (1) | 002 (2) | Adelaide Main GRN Site | Central - Greater Adelaide & Mt Lofty Ranges, SA | 413.3125 |
| 1 (1) | 003 (3) | Pt Adelaide | Central - Greater Adelaide & Mt Lofty Ranges, SA | 414.6875, 417.6875 |
| 1 (1) | 004 (4) | Banksia Park - Tea Tree Gully | Central - Greater Adelaide & Mt Lofty Ranges, SA | 414.5875, 415.8375 |
| 1 (1) | 005 (5) | Trott Park | Statewide, SA | 416.4500 |
| 1 (1) | 006 (6) | One Tree Hill | UNKNOWN, SA | 413.2125, 415.9625 |
| 1 (1) | 007 (7) | Belair | UNKNOWN, SA | 420.3625 |
| 1 (1) | 008 (8) | Cherry Gardens | UNKNOWN, SA | 413.2250 |
| 1 (1) | 009 (9) | Hackham West | Central - Greater Adelaide & Mt Lofty Ranges, SA | 413.3500 |
| 1 (1) | 010 (A) | Gawler | Statewide, SA | 414.7875, 417.0375 |
| 1 (1) | 011 (B) | Mt Barker | Central - Greater Adelaide & Mt Lofty Ranges, SA | 413.3875 |
| 1 (1) | 012 (C) | Mt Terrible | Central - Greater Adelaide & Mt Lofty Ranges, SA | 415.2500 |
| 1 (1) | 013 (D) | Lobethal | UNKNOWN, SA | 420.1375 |
| 1 (1) | 014 (E) | White Hill - Murray Bridge | Central - Greater Adelaide & Mt Lofty Ranges, SA | 413.1875 |
| 1 (1) | 016 (10) | Kadina | UNKNOWN, SA | 413.3625, 416.3625, 416.8625 |
| 1 (1) | 017 (11) | Browns Rd (Monash) | Murray - Riverlands, SA | 413.2500, 416.2500, 416.7500 |
| 1 (1) | 045 (2D) | Coonawarra | Southeast - Lower South East, SA | 413.3500, 416.3500, 416.8500 |
| 1 (1) | 065 (41) | Murtho Rd (Paringa) | Murray - Riverlands, SA | 413.1250, 416.1250, 416.6250 |
| 1 (1) | 066 (42) | Ramco | UNKNOWN, SA | 413.4000 |
| 1 (1) | 070 (46) | Mt Gambier | Southeast - Lower South East, SA | 422.1375 |
| 1 (1) | 072 (48) | Kapunda | Statewide, SA | 413.3500 |
| 1 (1) | 074 (4A) | Virginia | Statewide, SA | 413.4125 |
| 1 (1) | 075 (4B) | Williamstown | Statewide, SA | 415.0375, 415.7875 |
| 1 (1) | 077 (4D) | Glenelg | Statewide, SA | 415.1625 |
| 1 (1) | 084 (54) | Bull Creek | Central - Greater Adelaide & Mt Lofty Ranges, SA | 413.1750 |
| 1 (1) | 089 (59) | Yankalilla | Central - Greater Adelaide & Mt Lofty Ranges, SA | 413.2500 |
| 1 (1) | 096 (60) | Rieger Building | Central - Greater Adelaide & Mt Lofty Ranges, SA | 412.9875 |
| 1 (1) | 097 (61) | New Royal Adelaide Hospital | Central - Greater Adelaide & Mt Lofty Ranges, SA | 412.4875 |
| 1 (1) | 100 (64) | Bookpurnong Tce (Loxton) | Murray - Riverlands, SA | 413.0000, 416.5000, 417.0000 |
| 1 (1) | 103 (67) | Manting Rd (Mindarie) | Murray - Murrylands, SA | 412.9625, 416.4625, 416.9625 |
| 1 (1) | 104 (68) | Talinga | Statewide, SA | 413.3375 |
| 1 (1) | 121 (79) | Vardon Tce (Lameroo) | Murray - Riverlands, SA | 413.2125, 416.2125, 416.7125 |
| 1 (1) | 122 (7A) | Bone Rd (Pinnaroo) | Murray - Riverlands, SA | 413.0875, 416.0875, 416.5875 |
| 1 (1) | 126 (7E) | Unknown | UNKNOWN, SA | 420.1750, 420.4250, 420.6750 |
| 1 (1) | 140 (8C) | SAGRN site | Central - Greater Adelaide & Mt Lofty Ranges, SA | 420.2375 |
| 1 (1) | 201 (C9) | Mount Compass | UNKNOWN, SA | 413.3750 |
| 1 (1) | 202 (CA) | Trial Hill | UNKNOWN, SA | 413.2875 |
| 1 (1) | 203 (CB) | Nitschke Hill | UNKNOWN, SA | 413.1375 |
| 1 (1) | 204 (CC) | Port Augusta | UNKNOWN, SA | 413.0875 |
| 2 (2) | 080 (50) | State Admin Center (CBD) | Central - Greater Adelaide & Mt Lofty Ranges, SA | 420.2500 |
| 3 (3) | 072 (48) | Lochiel | UNKNOWN, SA | 413.0000, 416.2500, 416.5000 |
| 3 (3) | 073 (49) | Clare | Statewide, SA | 413.0000 |
| 3 (3) | 080 (50) | Cowell | Statewide, SA | 413.3250 |
| 3 (3) | 086 (56) | Cummins | Statewide, SA | 413.4375 |

### Unique Control-Channel-Capable Frequencies (MHz)
412.4875, 412.9625, 412.9875, 413.0000, 413.0875, 413.1250, 413.1375, 413.1750, 413.1875, 413.2125, 413.2250, 413.2500, 413.2875, 413.3125, 413.3250, 413.3375, 413.3500, 413.3625, 413.3750, 413.3875, 413.4000, 413.4125, 413.4375, 414.5875, 414.6875, 414.7625, 414.7875, 415.0375, 415.1625, 415.2500, 415.7625, 415.7875, 415.8375, 415.9625, 416.0875, 416.1250, 416.2125, 416.2500, 416.3500, 416.3625, 416.4500, 416.4625, 416.5000, 416.5875, 416.6250, 416.7125, 416.7500, 416.8500, 416.8625, 416.9625, 417.0000, 417.0375, 417.6875, 417.7625, 420.1375, 420.1750, 420.2375, 420.2500, 420.3625, 420.4250, 420.6750, 422.1375, 468.0000

## Sources
- RadioReference system page: https://www.radioreference.com/db/sid/8145
- RadioReference sites table: https://www.radioreference.com/db/sid/8145 (Sites and Frequencies)
- SAScan General Network Information: https://www.sascan.net.au/?page=infPages/GRN_GenInfo
- SAScan SA GRN sites & frequencies: https://www.sascan.net.au/?page=infPages/GRN_Sites
- Wikipedia Project 25: https://en.wikipedia.org/wiki/Project_25
