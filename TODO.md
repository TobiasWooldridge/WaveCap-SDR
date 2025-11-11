# WaveCap-SDR TODO List

Based on comprehensive audit conducted 2025-11-10

## ✅ Immediate Wins (Quick & High Impact)

- [x] Add toast notification system (2-3 hours)
- [x] Increase touch targets to 44x44px minimum (1 hour)
- [ ] Add ARIA labels for accessibility (2 hours)
- [ ] Add frequency bookmarks/favorites (3-4 hours)
- [ ] Add click-to-tune on spectrum (2 hours)

## 🔴 Critical Issues (Security/Stability)

- [x] Add error boundaries to prevent full app crashes
- [x] Fix potential memory leaks in WebSocket cleanup
- [x] Add input validation and sanitization
- [ ] Implement rate limiting on API
- [x] Add CORS configuration
- [ ] Validate configuration YAML to prevent crashes

## 🔥 High Priority (Core Functionality)

### UI/UX
- [x] Add success/error toast notifications
- [ ] Implement optimistic updates for better UX
- [x] Add skeleton screens for loading states
- [ ] Fix mobile responsiveness
  - [ ] Touch targets 44x44px minimum
  - [ ] Fix horizontal overflow on NumericSelector
  - [ ] Stack two-column layout on mobile
  - [ ] Convert modals to bottom sheets on mobile
- [ ] Improve error messages with suggested actions
- [ ] Add keyboard shortcuts

### Radio Features
- [ ] Add waterfall display
- [ ] Add audio level meters/VU meters
- [ ] Add frequency bookmarks/favorites
- [ ] Implement click-to-tune on spectrum
- [ ] Add band scanning
- [ ] Show signal strength relative to squelch level

## 🟡 Medium Priority (UX Improvements)

### Navigation
- [ ] Add breadcrumb navigation (Device > Capture > Channel)
- [ ] Implement horizontal scrolling for capture tabs overflow
- [ ] Add search/filter for channels
- [ ] Add capture tab management (close, reorder)
- [ ] Add keyboard navigation for tabs (arrow keys)

### Visual Design
- [ ] Create design system with typography scale
- [ ] Define spacing scale (xs, sm, md, lg, xl)
- [ ] Standardize icon sizes (16, 24, 32)
- [ ] Move inline styles to CSS classes
- [ ] Add CSS custom properties for theming
- [ ] Fix font size inconsistencies

### Component Organization
- [ ] Extract reusable primitives (Card, Modal, FormField, EditableText)
- [ ] Create custom hook for editable text pattern
- [ ] Split RadioTuner into smaller components
- [ ] Extract components from App.tsx
- [ ] Reduce code duplication in name editing

### Accessibility
- [ ] Add ARIA labels for icon-only buttons
- [ ] Add aria-describedby for form field hints
- [ ] Add role="tablist" for capture tabs
- [ ] Add aria-live regions for dynamic updates
- [ ] Ensure all interactive elements are keyboard accessible
- [ ] Add focus-visible styling to all components
- [ ] Run WCAG color contrast checker
- [ ] Add alt text for icons
- [ ] Announce form validation errors to screen readers

### Error Handling
- [ ] Implement error boundaries with fallback UI
- [ ] Add inline validation errors before API submission
- [ ] Add "undo" for destructive operations
- [ ] Create loading state standards
- [ ] Show validation feedback in real-time

## 🟢 Radio Functionality Enhancements

### Device Management
- [ ] Add device hot-plug support (detect new devices)
- [ ] Add device refresh button
- [ ] Show device capabilities in UI
- [ ] Prevent concurrent captures on same device
- [ ] Add driver selection in UI
- [ ] Add device health monitoring (temperature, buffer overruns)

### Channel Tuning
- [ ] Add frequency presets/bookmarks system
- [ ] Add mode descriptions (when to use WBFM vs NBFM)
- [ ] Show squelch threshold visually on signal meter
- [ ] Add frequency history (recent tunings)
- [ ] Add band plan overlay on spectrum
- [ ] Implement memory channels
- [ ] Add auto-tune to strongest signal

### Spectrum Analysis
- [ ] Add waterfall display
- [ ] Implement peak hold mode
- [ ] Add zoom/pan controls
- [ ] Add labeled axes with dB scale
- [ ] Implement click-to-tune
- [ ] Add persistence/averaging modes
- [ ] Add color palette selection
- [ ] Add FFT size control
- [ ] Add marker annotations

### Audio Quality
- [ ] Add audio level meters (VU meter)
- [ ] Show latency measurement
- [ ] Add audio processing (noise reduction, AGC, compressor)
- [ ] Add stereo support
- [ ] Add bitrate selection for compressed formats
- [ ] Add audio quality presets

### Signal Processing
- [ ] Add DSP controls (decimation, AGC, filters)
- [ ] Add constellation diagram viewer
- [ ] Add eye diagram viewer
- [ ] Add I/Q recording
- [ ] Add CTCSS/DCS tone decoder
- [ ] Add RDS decoder for FM
- [ ] Integrate P25/DMR talkgroup info
- [ ] Add signal quality metrics dashboard
- [ ] Add decoding for APRS, ADS-B, etc.

### Configuration
- [ ] Add settings UI for editing configuration
- [ ] Add config validation with helpful errors
- [ ] Add import/export for configurations
- [ ] Add "Save as Preset" button
- [ ] Add configuration profiles system
- [ ] Add inline help for all settings
- [ ] Add config migration for version upgrades

## 🔧 Technical Improvements

### Code Organization
- [ ] Split large files:
  - [ ] api.py → separate routers
  - [ ] capture.py → separate classes
  - [ ] App.tsx → extract components
- [ ] Add unit tests (pytest + Vitest)
- [ ] Add integration tests
- [ ] Add JSDoc/docstrings
- [ ] Use enums for state strings
- [ ] Document complex algorithms

### API Design
- [ ] Add pagination (limit/offset)
- [ ] Add filtering and sorting
- [ ] Add bulk operation endpoints
- [ ] Standardize WebSocket auth
- [ ] Configure CORS properly
- [ ] Customize OpenAPI docs
- [ ] Add custom validation error formatter

### State Management
- [ ] Implement optimistic updates
- [ ] Add global state management (Zustand)
- [ ] Refine cache invalidation
- [ ] Integrate WebSocket with React Query
- [ ] Add localStorage persistence
- [ ] Add state reconciliation logic

### Error Recovery
- [ ] Add React error boundaries
- [ ] Implement auto-recovery for captures
- [ ] Add WebSocket reconnection for spectrum
- [ ] Add circuit breaker pattern
- [ ] Isolate channel failures
- [ ] Add health check monitoring
- [ ] Add alerting for failures

### Performance
- [ ] Add SQLite database for persistence
- [ ] Move DSP to worker threads
- [ ] Use binary WebSocket format for FFT
- [ ] Add backpressure handling
- [ ] Lazy load modals and heavy components
- [ ] Add WebSocket for channel updates (replace polling)
- [ ] Implement code splitting with React.lazy
- [ ] Add compression for API responses
- [ ] Profile and optimize hot paths

## 💡 Nice to Have (Future Enhancements)

- [ ] Add theming support (light/dark mode)
- [ ] Implement band scanning
- [ ] Add RDS/CTCSS decoding
- [ ] Create recipe marketplace
- [ ] Add offline support with service worker
- [ ] Add PWA support
- [ ] Add multi-language support
- [ ] Add user accounts and cloud sync
- [ ] Add remote SDR sharing
- [ ] Add collaborative listening (multiple users)

## 📝 Documentation Needs

- [ ] Add component usage guidelines
- [ ] Document design system
- [ ] Add API documentation examples
- [ ] Document keyboard shortcuts
- [ ] Add troubleshooting guide
- [ ] Document DSP algorithms
- [ ] Add contributing guide
- [ ] Add architecture documentation

## 🐛 Known Issues

- [ ] Capture tabs overflow horizontally without scrolling
- [ ] NumericSelector overflows on narrow screens
- [ ] White border on selected tabs looks bad with shadows ✅ FIXED
- [ ] Frequency selector overflows in channel settings ✅ FIXED
- [ ] Channel cards too narrow on some screens ✅ FIXED
- [ ] Tab alignment in navbar ✅ FIXED
- [ ] Device names not showing shorthand ✅ FIXED
- [ ] Capture names missing device name ✅ FIXED
- [ ] Spectrum analyzer causes lag (idle pause added) ✅ FIXED
- [ ] Play button streaming connection issues ✅ FIXED

## 📊 Metrics to Track

- [ ] Page load time
- [ ] Time to interactive
- [ ] WebSocket connection stability
- [ ] Audio playback latency
- [ ] API response times
- [ ] Error rates
- [ ] User retention
- [ ] Feature usage

---

## Priority Labels

- 🔴 **Critical**: Security, stability, or breaking issues
- 🔥 **High**: Core functionality, major UX issues
- 🟡 **Medium**: Improvements, nice-to-haves that enhance experience
- 🟢 **Enhancement**: New features, future improvements
- 🔧 **Technical**: Code quality, architecture, performance
- 💡 **Future**: Ideas for later consideration
