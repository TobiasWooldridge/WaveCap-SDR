# WaveCap-SDR TODO List

Based on comprehensive SDR feature audit conducted 2025-11-11

---

## 📊 Executive Summary

### Current State
WaveCap-SDR has **excellent architectural foundations** with a modern web interface, flexible DSP pipeline, and multi-device support. The application successfully implements:
- WBFM/NBFM demodulation (fully functional)
- Real-time spectrum analysis and waterfall display
- Multi-channel architecture with independent demodulation
- Audio streaming in multiple formats (PCM, MP3, Opus, AAC)
- Frequency bookmarks and wizard-based setup
- AGC implementation for AM/SSB modes
- ✅ **S-Meter Display** - Visual signal strength with S1-S9+60dB scale (completed 2025-11-13)
- ✅ **Active Notch Filters** - Multi-frequency interference rejection (completed 2025-11-13)
- ✅ **Click-to-Tune on Spectrum** - Interactive spectrum with frequency tooltip (completed 2025-11-13)

### Top Priority Gaps
1. **Scanner Mode** - Automated frequency scanning with signal detection
2. **Frequency History & Memory Banks** - Quick recall of recent frequencies and saved configurations
3. **Complete AM/SSB** - Current implementations need refinement for production use
4. **Digital Mode Codecs** - P25/DMR voice decoders (IMBE/AMBE)
5. **CW Decoder** - Morse code support

### Implementation Roadmap
- **Phase 1 (1-2 weeks):** Quick wins with immediate user impact
- **Phase 2 (4-6 weeks):** Core radio features
- **Phase 3 (6-8 weeks):** Digital mode completion
- **Phase 4 (2-4 weeks):** Advanced features and polish

---

## 🎯 Phase 1: Quick Wins (1-2 weeks)

High-value features using existing infrastructure. Minimal risk, immediate user benefit.

### ✅ S-Meter Display (COMPLETED)
**Status:** ✅ FULLY IMPLEMENTED
**Completed:** 2025-11-13
**Files:**
- `backend/wavecapsdr/capture.py` - RSSI/SNR calculation from IQ samples (lines 182-206)
- `frontend/src/components/primitives/SMeter.react.tsx` - S-meter component
- `frontend/src/components/CompactChannelCard.react.tsx` - Integrated S-meter display

**Implementation:**
- ✅ Convert RSSI (dBFS) to S-units with proper scale mapping
- ✅ Visual analog meter with S1-S9+60dB scale and tick marks
- ✅ Color coding: orange (S1-S4), yellow (S5-S6), green (S7-S9), red (S9+20+)
- ✅ Shows both visual meter and numeric RSSI/SNR values
- ✅ Backend calculates RSSI/SNR every processing cycle
- ✅ Displays automatically in all channel cards

---

### ✅ Active Notch Filter (COMPLETED)
**Status:** ✅ FULLY IMPLEMENTED
**Completed:** 2025-11-13
**Files:**
- `backend/wavecapsdr/dsp/filters.py` - `notch_filter()` function with Q=30 (lines 156-198)
- `backend/wavecapsdr/models.py` - `notch_frequencies: list[float]` in ChannelConfig
- `backend/wavecapsdr/capture.py` - Integrated in all demodulation modes (FM/AM/SSB)
- `backend/wavecapsdr/dsp/{fm.py,am.py}` - Notch filters applied in DSP chain
- `frontend/src/components/CompactChannelCard.react.tsx` - Full notch filter UI (lines 800-846)

**Implementation:**
- ✅ Support multiple notch frequencies per channel (max 10)
- ✅ UI: Input field + "Add" button with validation
- ✅ Shows list of active notches with remove button
- ✅ Default Q factor: 30 (narrow notch, high selectivity)
- ✅ Chains multiple notch filters in series
- ✅ Frequency validation: 0-20kHz range
- ✅ Applied to WBFM, NBFM, AM, and SSB modes

**Use Cases:** Remove power line hum (60 Hz, 120 Hz), carrier interference, birdie tones

---

### Frequency History & Memory Banks
**Status:** Not implemented
**Effort:** 8-10 hours
**Files:**
- `backend/wavecapsdr/models.py` - Add FrequencyHistory model
- `backend/wavecapsdr/api.py` - Add `/api/history` and `/api/memory-banks` endpoints
- `frontend/src/hooks/useFrequencyHistory.ts` - New hook
- `frontend/src/components/BookmarkManager.react.tsx` - Add "Recent" tab

**Implementation:**
- **History:** Store last 100 tuned frequencies with timestamps in localStorage
- **Memory Banks:** Save/recall complete capture configs (frequency + all channels + modes)
- UI: "Recent" tab shows chronological list with quick-tune
- "Save to Memory" button saves entire capture state with name
- "Load Memory" recalls saved configuration

**Wizard Integration:**
- Add "Load from Memory" option at wizard start
- "Save as Memory Bank" button at wizard completion

---

### ✅ Click-to-Tune on Spectrum (COMPLETED)
**Status:** ✅ FULLY IMPLEMENTED
**Completed:** 2025-11-13
**Files:**
- `frontend/src/components/primitives/SpectrumAnalyzer.react.tsx` - Click handler and tooltip (lines 521-571, 657-694)
- `frontend/src/App.tsx` - handleFrequencyClick updates capture (lines 211-223)

**Implementation:**
- ✅ Make spectrum canvas clickable
- ✅ Calculate frequency from click X position
- ✅ Update capture center_hz via API call with toast notification
- ✅ Visual feedback: cursor crosshair over spectrum
- ✅ Show frequency tooltip on hover with "Click to tune" hint
- ✅ Tooltip displays frequency in MHz with 4 decimal precision
- ✅ Smooth hover tracking with mousemove/mouseleave handlers

**Additional Features for Future:**
- Drag-to-create-channel: click+drag on spectrum → creates new channel at offset
- Snap-to-peak option (tune to strongest signal near click)

---

## 🔥 Phase 2: Core Features (4-6 weeks)

Essential radio functionality that matches user expectations from SDR applications.

### Scanner Mode
**Status:** Not implemented
**Effort:** 2-3 weeks
**Priority:** CRITICAL - Most requested feature
**Files:**
- `backend/wavecapsdr/scanner.py` - New ScannerService class
- `backend/wavecapsdr/models.py` - Add ScanConfig model
- `backend/wavecapsdr/api.py` - Add `/api/scanners` endpoints
- `frontend/src/components/ScannerControl.react.tsx` - New component

**Implementation:**

**Backend:**
```python
class ScanMode(str, Enum):
    SEQUENTIAL = "sequential"  # A→B→C→A
    PRIORITY = "priority"      # Check priority freq every N seconds
    ACTIVITY = "activity"      # Pause on active signal (squelch-based)

class ScanConfig(BaseModel):
    scan_list: list[float]  # Frequencies to scan (Hz)
    mode: ScanMode
    dwell_time_ms: int = 500  # Time per frequency
    priority_frequencies: list[float] = []
    priority_interval_s: int = 5
    squelch_threshold_db: float = -60.0  # For activity mode
    lockout_frequencies: list[float] = []  # Skip these
```

**Scan Logic:**
- Sequential: Cycle through scan_list with dwell_time delay
- Priority: Check priority_frequencies every priority_interval_s
- Activity: When squelch opens, pause scan for dwell_time

**UI Features:**
- Frequency list editor (add/remove/import from bookmarks)
- Scan speed slider (100ms - 5s per freq)
- Priority channel checkbox per frequency
- Lockout button (temporarily skip frequency)
- Activity log showing hits

**Wizard Integration:**
- New "Scanner Wizard" in CreateCaptureWizard
- Pre-built scan lists:
  - "Marine VHF Scan" (Ch 16, 9, 13, 68, 72)
  - "Aviation Scan" (118.0 - 137.0 MHz, 25 kHz steps)
  - "Public Safety Scan" (User's local frequencies)
  - "Shortwave Broadcast" (Common SW stations)
- Wizard collects: frequency range OR manual list, scan speed, mode

---

### Complete AM/SSB Implementation
**Status:** Basic stubs exist in `dsp/am.py`
**Effort:** 2-3 weeks
**Files:**
- `backend/wavecapsdr/dsp/am.py` - Enhance AM/SSB demodulation
- `backend/wavecapsdr/models.py` - Add AM/SSB-specific parameters

**AM Improvements:**
- Synchronous AM detection (carrier lock for reduced fading)
- Better AGC tuning for broadcast/aviation voice
- Optional ECSS (Exalted Carrier Single Sideband) mode
- Carrier offset display (show +/- Hz from center)
- Add selectable bandwidth (2.5 kHz, 5 kHz, 8 kHz, 10 kHz)

**SSB Improvements:**
- Sharper bandpass filters: 300-2700 Hz for voice, 200-500 Hz for CW
- Fine-tune frequency control (±10 Hz matters for SSB clarity)
- BFO (Beat Frequency Oscillator) adjustment in UI
- AGC optimized for SSB voice (faster attack, slower release)
- USB/LSB auto-detection from frequency (USB above 10 MHz, LSB below)

**Parameters to Add:**
```python
class AMConfig(BaseModel):
    sync_detection: bool = False  # Synchronous AM
    bandwidth_hz: int = 5000

class SSBConfig(BaseModel):
    sideband: Literal["usb", "lsb", "auto"] = "auto"
    bfo_offset_hz: int = 0  # Fine-tune
    cw_filter: bool = False  # Narrow 500 Hz filter
```

**Wizard Integration:**
- "Shortwave Listening" recipe category
- "Ham Radio SSB" recipes (14.200 MHz USB, 7.200 MHz LSB, etc.)
- "AM Broadcast" recipes with sync AM option
- Wizard auto-selects sideband based on frequency

---

### Stereo FM Decoding
**Status:** WBFM is mono only
**Effort:** 1-2 weeks
**Files:**
- `backend/wavecapsdr/dsp/fm.py` - Add stereo pilot detection and L-R decoding
- `backend/wavecapsdr/models.py` - Add `stereo: bool` to ChannelConfig

**Implementation:**
- Detect 19 kHz pilot tone
- Demodulate 38 kHz L-R subcarrier (double frequency)
- Reconstruct L and R channels: L = (L+R) + (L-R), R = (L+R) - (L-R)
- Output dual-channel audio stream
- Fallback to mono if pilot weak/absent

**Wizard Integration:**
- "FM Radio" recipes enable stereo by default
- Checkbox in channel settings: "Stereo (if available)"

---

### RDS Decoder for FM
**Status:** Not implemented (57 kHz subcarrier removed)
**Effort:** 2-3 weeks
**Files:**
- `backend/wavecapsdr/dsp/rds.py` - New RDS decoder module
- `backend/wavecapsdr/models.py` - Add RDS data fields
- `frontend/src/components/CompactChannelCard.react.tsx` - Display RDS info

**Implementation:**
- Extract 57 kHz RDS subcarrier before MPX filtering
- BPSK demodulation and bit synchronization
- Decode RDS groups:
  - Group 0A/0B: Program Service name (PS) - 8 characters
  - Group 2A/2B: Radio Text (RT) - 64 characters
  - Group 4A: Clock Time
  - PTY (Program Type): News, Rock, Classical, etc.
- Error correction with syndrome lookup

**UI Display:**
- Station name badge (PS)
- Scrolling text for RT
- Program type icon

**Wizard Integration:**
- Auto-enabled for WBFM mode
- No user configuration needed

---

## 🟣 Phase 3: Digital Modes (6-8 weeks)

Complete existing digital mode skeletons and add new decoders.

### P25 & DMR Voice Codec Integration
**Status:** Framework exists, voice silent (TODOs at capture.py:555, 567)
**Effort:** 3-4 weeks
**Files:**
- `backend/wavecapsdr/dsp/p25.py` - Integrate IMBE decoder
- `backend/wavecapsdr/dsp/dmr.py` - Integrate AMBE decoder
- `backend/requirements.txt` - Add codec libraries

**Implementation:**

**P25:**
- Integrate `imbe_vocoder` library (open-source Python bindings)
- Decode voice frames → IMBE codewords → PCM audio
- Handle frame types: HDU, LDU1, LDU2, TDU
- Extract encryption indicator (show "Encrypted" badge if E-bit set)

**DMR:**
- Integrate `mbelib` via Python bindings (check licensing)
- Decode AMBE+2 codewords → PCM audio
- Support dual-slot time division (Slot 1 & Slot 2)
- Decode Color Code and slot timing

**Licensing Note:**
- IMBE: Patent expired, open-source implementations available
- AMBE: May require licensing; alternative open implementations exist

**Wizard Integration:**
- "Trunked Radio" category in wizard
- Recipes: "P25 Conventional", "DMR Repeater"
- Auto-configure for 12.5 kHz channel spacing

---

### P25/DMR Trunking Auto-Follow
**Status:** TSBK/CSBK decoding exists, voice following missing (capture.py:579)
**Effort:** 2-3 weeks
**Files:**
- `backend/wavecapsdr/dsp/p25.py` - Add trunking controller
- `backend/wavecapsdr/dsp/dmr.py` - Add trunking controller
- `backend/wavecapsdr/models.py` - Add TrunkingConfig

**Implementation:**

**P25 Trunking:**
- Monitor control channel for TSBK grant messages
- On grant: Create temporary channel at voice frequency
- Follow talkgroup across frequency changes
- Decode NAC (Network Access Code) and talk group IDs
- Display talkgroup name (from user-provided database)

**DMR Trunking:**
- Monitor control channel for CSBK messages
- Parse Channel Grant, Channel Assignment
- Follow talk group across time slots and frequencies
- Decode Color Code

**Configuration:**
```python
class TrunkingConfig(BaseModel):
    control_channel_hz: float
    talkgroup_filter: Literal["all", "whitelist", "blacklist"] = "all"
    talkgroup_whitelist: list[int] = []
    talkgroup_blacklist: list[int] = []
    auto_create_channels: bool = True
    max_voice_channels: int = 10
```

**Wizard Integration:**
- New "Trunking Wizard"
- Step 1: Select system type (P25 Phase I/II, DMR Tier III)
- Step 2: Enter control channel frequency
- Step 3: Configure talkgroup filters
- Step 4: Import talkgroup database (CSV)
- Auto-create tracking channels

---

### CW (Morse Code) Decoder
**Status:** Listed in Mode enum but not implemented
**Effort:** 1-2 weeks
**Files:**
- `backend/wavecapsdr/dsp/cw.py` - New CW decoder
- `backend/wavecapsdr/models.py` - Add CWConfig
- `frontend/src/components/CompactChannelCard.react.tsx` - Show decoded text

**Implementation:**
- Narrow bandpass filter (500 Hz centered on tone)
- Envelope detection and AGC
- Edge detection (key-down/key-up timing)
- Dit/dah classification (dah = 3× dit length)
- Character spacing (3× dit) and word spacing (7× dit)
- ITU Morse table lookup
- Real-time text output

**Auto-Tuning:**
- FFT to detect tone frequency (400-1000 Hz typical)
- Auto-adjust filter center to match tone

**UI:**
- Decoded text display (scrolling)
- WPM (words per minute) indicator
- Tone frequency display

**Wizard Integration:**
- "Ham Radio CW" recipe with common CW frequencies
- Auto-enable narrow filter (500 Hz)
- Suggested frequencies: 3.525, 7.025, 14.025, 21.025, 28.025 MHz

---

### Additional Decoders

#### CTCSS/DCS Tone Decoder
**Effort:** 1 week
**Use Case:** Sub-audible tone squelch for repeater access

**Implementation:**
- Detect sub-audible tones (67-254 Hz CTCSS)
- Decode DCS codes (Digital Coded Squelch)
- Display detected tone/code
- Tone squelch mode (unmute only on matching tone)
- Tone search mode (scan all CTCSS tones)

#### PSK31 Decoder
**Effort:** 2-3 weeks
**Use Case:** Popular digital mode on HF amateur bands

**Implementation:**
- Phase shift keying demodulation (31.25 baud)
- Varicode decoding (variable-length character encoding)
- AFC (Automatic Frequency Control)
- Waterfall tuning indicator
- Text display with color-coded RX/TX

---

## 🌟 Phase 4: Polish (2-4 weeks)

Advanced features and UX refinements.

### Noise Blanker
**Effort:** 1 week
**Files:** `backend/wavecapsdr/dsp/filters.py`

**Implementation:**
- Detect impulse noise (threshold-based)
- Blank samples exceeding threshold
- Configurable threshold and blanking duration
- UI toggle and sensitivity slider

**Use Cases:** Ignition noise, power tools, lightning static

---

### Band Plan Overlay
**Effort:** 1-2 weeks
**Files:**
- `backend/wavecapsdr/bands.py` - Band plan database
- `frontend/src/components/primitives/SpectrumAnalyzer.react.tsx` - Overlay rendering

**Implementation:**
- Frequency range database:
  - Amateur radio bands (160m, 80m, 40m, 20m, 15m, 10m, 6m, 2m, 70cm)
  - Broadcast bands (AM, FM, SW)
  - Aviation (118-137 MHz)
  - Marine VHF (156-162 MHz)
  - Public safety (150-174 MHz, 450-470 MHz)
- Color-coded regions on spectrum
- Labels appear on hover
- Click region → auto-tune and configure

**Wizard Integration:** Recipe categories align with band plan regions

---

### Advanced Spectrum Features

#### Peak Hold Mode
**Effort:** 4-6 hours
**Implementation:** Track maximum value per FFT bin, decay slowly over time

#### Averaging Mode
**Effort:** 4-6 hours
**Implementation:** Moving average of last N FFT frames, configurable N

#### Zoom & Pan
**Effort:** 1 week
**Implementation:**
- Click+drag to select frequency range → zoom
- Pan with arrow keys or drag
- Reset zoom button
- Frequency span control (full/±5 MHz/±1 MHz/±500 kHz)

#### Adjustable FFT Size
**Effort:** 4-6 hours
**Options:** 512, 1024, 2048, 4096, 8192 bins
**Trade-off:** Larger FFT = better resolution but slower updates

#### Power Scale Control
**Effort:** 4-6 hours
**Implementation:** Manual min/max dB sliders, auto-scale toggle

---

## 🔧 Wizard Integration Strategy

### Enhancements to CreateCaptureWizard

**New Recipe Categories:**
- **Scanning** → Opens ScannerWizard
- **Trunked Radio** → Opens TrunkingWizard
- **CW/Digital Modes** → Pre-configured for narrow filters

**Per-Recipe Options (Step 2):**
- Checkbox: "Enable scanner mode" (if recipe has frequency list)
- Slider: "Squelch level" with visual preview
- Dropdown: "Preferred demodulation mode"

**Post-Creation Flow (Step 3):**
- Show spectrum with hint: "Click to add channels"
- Quick action: "Save to memory bank?"
- Quick action: "Start scanning?"

### New Wizards

#### ScannerWizard
**Steps:**
1. Scan type: Sequential / Priority / Activity
2. Add frequencies: Manual entry, import bookmarks, range entry
3. Configure: Scan speed, squelch, priority list
4. Confirm and start

**Pre-built Scan Lists:**
- Marine VHF (16, 9, 13, 68, 69, 71, 72, 73)
- Aviation (118.0-137.0 MHz, 25 kHz steps)
- Weather (162.400, 162.425, 162.450, 162.475, 162.500, 162.525, 162.550 MHz)
- Public Safety (user-configurable)

#### TrunkingWizard
**Steps:**
1. System type: P25 Phase I, P25 Phase II, DMR Tier III
2. Control channel frequency
3. Talkgroup filter: All, whitelist, blacklist
4. Import talkgroup CSV (optional)
5. Confirm and start tracking

---

## 🎯 Current Strengths (Keep These!)

**Architecture:**
- Clean async/await backend with proper error handling
- Multi-capture architecture allows concurrent operations
- Flexible DSP pipeline (easy to add new filters/demodulators)
- Type-safe Pydantic models for API contracts

**UI/UX:**
- Modern React with React Query (excellent caching/invalidation)
- Real-time WebSocket updates for spectrum/audio
- Recipe-based wizard system (intuitive for beginners)
- Toast notifications and skeleton loading states

**Device Support:**
- SoapySDR abstraction (RTL-SDR, SDRplay, HackRF, etc.)
- Hot reconfiguration (frequency/gain without restart)
- Device naming system with shorthand

**Audio Quality:**
- Multiple codec support (PCM, MP3, Opus, AAC)
- Web Audio API playback
- HTTP stream URLs for external players

---

## ✅ Completed Features (Reference)

These items are already implemented and working well:

- [x] WBFM/NBFM demodulation (fully functional)
- [x] Spectrum analyzer with real-time FFT
- [x] Waterfall display with color schemes
- [x] Frequency bookmarks with notes
- [x] Toast notification system
- [x] Multi-capture tab management
- [x] Device nickname system
- [x] Capture/channel naming (auto and manual)
- [x] Error boundaries and error recovery
- [x] Rate limiting and CORS
- [x] Recipe-based wizard system
- [x] AGC implementation (for AM/SSB)
- [x] Squelch control (basic implementation)

---

## 🟡 Medium Priority (UX Improvements)

### Navigation
- [ ] Breadcrumb navigation (Device > Capture > Channel)
- [ ] Horizontal scrolling for capture tabs overflow
- [ ] Search/filter for channels
- [ ] Capture tab management (close, reorder)
- [ ] Keyboard navigation for tabs

### Visual Design
- [ ] Design system with typography scale
- [ ] Spacing scale (xs, sm, md, lg, xl)
- [ ] Standardize icon sizes (16, 24, 32)
- [ ] Move inline styles to CSS classes
- [ ] CSS custom properties for theming

### Component Organization
- [ ] Extract reusable primitives (Card, Modal, FormField)
- [ ] Split RadioTuner into smaller components
- [ ] Extract components from App.tsx
- [ ] Reduce code duplication

### Accessibility
- [ ] Add ARIA labels for icon-only buttons
- [ ] Add aria-describedby for form hints
- [ ] Add role="tablist" for capture tabs
- [ ] Add aria-live regions for updates
- [ ] Ensure keyboard accessibility
- [ ] Add focus-visible styling
- [ ] WCAG color contrast check
- [ ] Alt text for icons

### Mobile Responsiveness
- [ ] Fix horizontal overflow on NumericSelector
- [ ] Stack two-column layout on mobile
- [ ] Convert modals to bottom sheets on mobile
- [ ] Minimum 44x44px touch targets (mostly done)

---

## 🟢 Nice to Have (Future Enhancements)

### Advanced Features
- [ ] IQ constellation diagram
- [ ] Eye diagram for digital signals
- [ ] Signal quality metrics dashboard (EVM, BER)
- [ ] Frequency database with band plans (see Phase 4)
- [ ] Multi-device simultaneous operation
- [ ] Remote SDR server support (already architecturally possible)

### Digital Mode Expansion
- [ ] RTTY decoder (Baudot code)
- [ ] FT8/FT4 decoder (WSJT-X integration)
- [ ] APRS decoder (packet radio with mapping)
- [ ] ADS-B decoder (aircraft tracking)
- [ ] POCSAG/FLEX pager decoder

### Configuration
- [ ] Settings UI (edit YAML via web interface)
- [ ] Config validation with helpful errors
- [ ] Import/export configurations
- [ ] Configuration profiles system
- [ ] Inline help for all settings

### User Experience
- [ ] Theming support (dark mode already default, add light mode)
- [ ] Keyboard shortcuts (global)
- [ ] PWA support (offline capabilities)
- [ ] Multi-language support (i18n)

---

## 🐛 Known Issues

- [ ] Capture tabs overflow horizontally without scrolling
- [ ] NumericSelector overflows on narrow screens
- [x] White border on selected tabs ✅ FIXED
- [x] Frequency selector overflow ✅ FIXED
- [x] Channel cards too narrow ✅ FIXED
- [x] Tab alignment in navbar ✅ FIXED
- [x] Device shorthand names ✅ FIXED
- [x] Spectrum lag ✅ FIXED

---

## 🔧 Technical Debt

### Code Organization
- [ ] Split large files (api.py, capture.py, App.tsx)
- [ ] Add unit tests (pytest + Vitest)
- [ ] Add integration tests
- [ ] JSDoc/docstrings
- [ ] Document complex algorithms

### API Design
- [ ] Add pagination for large lists
- [ ] Add filtering and sorting
- [ ] Bulk operation endpoints
- [ ] Customize OpenAPI docs

### Performance
- [ ] Consider SQLite for persistence (currently in-memory)
- [ ] Binary WebSocket format for FFT data
- [ ] Code splitting with React.lazy
- [ ] Compression for API responses
- [ ] Profile and optimize hot paths

---

## 📝 Documentation Needs

### User Documentation
- [ ] Getting Started tutorial
- [ ] Troubleshooting guide (device detection, permissions, drivers)
- [ ] Keyboard shortcuts reference
- [ ] Frequency/band reference guide
- [ ] Legal disclaimer about monitoring regulations
- [ ] Demodulation mode guide (when to use each)

### Developer Documentation
- [ ] Component usage guidelines
- [ ] Design system documentation
- [ ] API documentation with examples
- [ ] DSP algorithm documentation
- [ ] Contributing guide
- [ ] Architecture documentation

### SDR-Specific Documentation
- [ ] Device capabilities per model
- [ ] Supported sample rates and gains
- [ ] Antenna recommendations per frequency range
- [ ] Signal identification guide
- [ ] Decoder usage guides (P25, DMR, CW, etc.)

---

## Priority Labels

- 🎯 **Phase 1**: Quick wins (1-2 weeks)
- 🔥 **Phase 2**: Core features (4-6 weeks)
- 🟣 **Phase 3**: Digital modes (6-8 weeks)
- 🌟 **Phase 4**: Polish (2-4 weeks)
- 🟡 **Medium**: UX improvements
- 🟢 **Future**: Nice-to-have enhancements
- 🔧 **Technical**: Code quality, architecture
- 📝 **Docs**: Documentation
- 🐛 **Bugs**: Known issues

---

**Last Updated:** 2025-11-11
**Total Estimated Effort:** 13-20 weeks for all phases
**Recommended Start:** Phase 1 (Quick Wins)
