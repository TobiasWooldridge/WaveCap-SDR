import type { TrunkingSystem } from "./trunking";

export interface Device {
  id: string;
  driver: string;
  label: string;
  freqMinHz: number;
  freqMaxHz: number;
  sampleRates: number[];
  gains: string[];
  gainMin: number | null;
  gainMax: number | null;
  bandwidthMin: number | null;
  bandwidthMax: number | null;
  ppmMin: number | null;
  ppmMax: number | null;
  antennas: string[];
  nickname?: string | null; // User-provided nickname
  shorthand?: string | null; // Auto-generated shorthand name (e.g., "RTL-SDR", "SDRplay RSPdx")
}

export interface ConfigWarning {
  code: string; // Machine-readable code (e.g., "rtl_unstable_sample_rate")
  severity: "warning" | "info";
  message: string; // Human-readable message
}

export interface Capture {
  id: string;
  deviceId: string;
  state:
    | "created"
    | "starting"
    | "running"
    | "stopping"
    | "stopped"
    | "failed"
    | "error";
  centerHz: number;
  sampleRate: number;
  gain: number | null;
  bandwidth: number | null;
  ppm: number | null;
  antenna: string | null;
  deviceSettings?: Record<string, string>;
  elementGains?: Record<string, number>;
  streamFormat?: string | null;
  dcOffsetAuto?: boolean;
  iqBalanceAuto?: boolean;
  errorMessage: string | null;
  name: string | null; // User-provided name
  autoName: string | null; // Auto-generated contextual name
  // FFT/Spectrum settings
  fftFps: number; // Target FFT frames per second
  fftMaxFps: number; // Maximum FFT frames per second (hard cap)
  fftSize: number; // FFT bin count (512, 1024, 2048, 4096)
  // Error indicators
  iqOverflowCount: number;
  iqOverflowRate: number; // Overflows per second
  retryAttempt: number | null; // Current retry attempt (null if not retrying)
  retryMaxAttempts: number | null;
  retryDelay: number | null; // Delay in seconds before next retry
  // Configuration warnings
  configWarnings?: ConfigWarning[]; // Lint warnings about configuration
  // Trunking system ownership
  trunkingSystemId?: string | null; // Set if this capture is managed by a trunking system
}

export interface Channel {
  id: string;
  captureId: string;
  mode: "wbfm" | "nbfm" | "am" | "ssb" | "raw" | "p25" | "dmr";
  state: "created" | "running" | "stopped";
  offsetHz: number;
  audioRate: number;
  squelchDb: number | null;
  name: string | null; // User-provided name
  autoName: string | null; // Auto-generated contextual name (e.g., "Marine Ch 16")
  signalPowerDb: number | null;
  rssiDb: number | null; // Server-side RSSI from IQ
  snrDb: number | null; // Server-side SNR estimate

  // Filter configuration
  // FM filters
  enableDeemphasis: boolean;
  deemphasisTauUs: number;
  enableMpxFilter: boolean;
  mpxCutoffHz: number;
  enableFmHighpass: boolean;
  fmHighpassHz: number;
  enableFmLowpass: boolean;
  fmLowpassHz: number;

  // AM/SSB filters
  enableAmHighpass: boolean;
  amHighpassHz: number;
  enableAmLowpass: boolean;
  amLowpassHz: number;
  enableSsbBandpass: boolean;
  ssbBandpassLowHz: number;
  ssbBandpassHighHz: number;
  ssbMode: string;

  // AGC
  enableAgc: boolean;
  agcTargetDb: number;
  agcAttackMs: number;
  agcReleaseMs: number;

  // Noise blanker
  enableNoiseBlanker: boolean;
  noiseBlankerThresholdDb: number;

  // Notch filters
  notchFrequencies: number[]; // List of frequencies to notch out (Hz)

  // Spectral noise reduction
  enableNoiseReduction: boolean;
  noiseReductionDb: number;

  // RDS data (WBFM only)
  rdsData?: RDSData | null;

  // Audio output level metering
  audioRmsDb?: number | null;
  audioPeakDb?: number | null;
  audioClippingCount?: number;

  // Error indicators
  audioDropCount: number;
  audioDropRate: number; // Drops per second
}

// RDS (Radio Data System) data for FM broadcast
export interface RDSData {
  piCode: string | null; // Program Identification (hex string like "A1B2")
  psName: string | null; // Program Service name (8 chars, station name)
  radioText: string | null; // Radio Text (up to 64 chars)
  pty: number; // Program Type code
  ptyName: string; // Program Type name (e.g., "Rock", "News")
  ta: boolean; // Traffic Announcement flag
  tp: boolean; // Traffic Program flag
  ms: boolean; // Music/Speech switch (true = Music)
}

// POCSAG pager message
export interface POCSAGMessage {
  address: number; // 21-bit capcode
  function: number; // Function code (0-3)
  messageType: "numeric" | "alpha" | "alert_only" | "alpha_2";
  message: string; // Decoded message content
  timestamp: number; // Unix timestamp
  baudRate: number; // 512, 1200, or 2400
}

export interface UpdateCaptureRequest {
  deviceId?: string;
  centerHz?: number;
  sampleRate?: number;
  gain?: number;
  bandwidth?: number;
  ppm?: number;
  antenna?: string;
  deviceSettings?: Record<string, string>;
  elementGains?: Record<string, number>;
  streamFormat?: string;
  dcOffsetAuto?: boolean;
  iqBalanceAuto?: boolean;
  name?: string | null;
  // FFT/Spectrum settings
  fftFps?: number; // Target FPS (1-60)
  fftMaxFps?: number; // Max FPS cap (1-120)
  fftSize?: number; // 512, 1024, 2048, 4096
}

export interface CreateCaptureRequest {
  deviceId?: string;
  centerHz: number;
  sampleRate: number;
  gain?: number;
  bandwidth?: number;
  ppm?: number;
  antenna?: string;
  deviceSettings?: Record<string, string>;
  elementGains?: Record<string, number>;
  streamFormat?: string;
  dcOffsetAuto?: boolean;
  iqBalanceAuto?: boolean;
  createDefaultChannel?: boolean;
  // FFT/Spectrum settings
  fftFps?: number; // Target FPS (1-60)
  fftMaxFps?: number; // Max FPS cap (1-120)
  fftSize?: number; // 512, 1024, 2048, 4096
}

export interface CreateChannelRequest {
  mode: "wbfm" | "nbfm" | "am" | "ssb" | "raw" | "p25" | "dmr";
  offsetHz?: number;
  audioRate?: number;
  squelchDb?: number | null;
  name?: string | null;
  notchFrequencies?: number[];
}

export interface UpdateChannelRequest {
  mode?: "wbfm" | "nbfm" | "am" | "ssb" | "raw" | "p25" | "dmr";
  offsetHz?: number;
  audioRate?: number;
  squelchDb?: number | null;
  name?: string | null;

  // Filter configuration
  // FM filters
  enableDeemphasis?: boolean;
  deemphasisTauUs?: number;
  enableMpxFilter?: boolean;
  mpxCutoffHz?: number;
  enableFmHighpass?: boolean;
  fmHighpassHz?: number;
  enableFmLowpass?: boolean;
  fmLowpassHz?: number;

  // AM/SSB filters
  enableAmHighpass?: boolean;
  amHighpassHz?: number;
  enableAmLowpass?: boolean;
  amLowpassHz?: number;
  enableSsbBandpass?: boolean;
  ssbBandpassLowHz?: number;
  ssbBandpassHighHz?: number;
  ssbMode?: "usb" | "lsb";

  // AGC
  enableAgc?: boolean;
  agcTargetDb?: number;
  agcAttackMs?: number;
  agcReleaseMs?: number;

  // Noise blanker
  enableNoiseBlanker?: boolean;
  noiseBlankerThresholdDb?: number;

  // Notch filters
  notchFrequencies?: number[];

  // Spectral noise reduction
  enableNoiseReduction?: boolean;
  noiseReductionDb?: number;
}

export interface RecipeChannel {
  offsetHz: number;
  name: string;
  mode: Channel["mode"];
  squelchDb: number;
  // POCSAG decoding settings (NBFM only)
  enablePocsag?: boolean;
  pocsagBaud?: number;
}

export interface Recipe {
  id: string;
  name: string;
  description: string;
  category: string;
  centerHz: number;
  sampleRate: number;
  gain?: number | null;
  bandwidth?: number | null;
  channels: RecipeChannel[];
  allowFrequencyInput: boolean;
  frequencyLabel?: string | null;
}

// Scanner types
export interface ScanHit {
  frequencyHz: number;
  timestamp: number;
}

export interface Scanner {
  id: string;
  captureId: string;
  state: "stopped" | "scanning" | "paused" | "locked";
  currentFrequency: number;
  currentIndex: number;
  scanList: number[];
  mode: "sequential" | "priority" | "activity";
  dwellTimeMs: number;
  priorityFrequencies: number[];
  priorityIntervalS: number;
  squelchThresholdDb: number;
  lockoutList: number[];
  pauseDurationMs: number;
  hits: ScanHit[];
}

export interface CreateScannerRequest {
  captureId: string;
  scanList: number[];
  mode?: "sequential" | "priority" | "activity";
  dwellTimeMs?: number;
  priorityFrequencies?: number[];
  priorityIntervalS?: number;
  squelchThresholdDb?: number;
  lockoutFrequencies?: number[];
  pauseDurationMs?: number;
}

export interface UpdateScannerRequest {
  scanList?: number[];
  mode?: "sequential" | "priority" | "activity";
  dwellTimeMs?: number;
  priorityFrequencies?: number[];
  priorityIntervalS?: number;
  squelchThresholdDb?: number;
  lockoutFrequencies?: number[];
  pauseDurationMs?: number;
}

// ==============================================================================
// Error tracking types for real-time health stream
// ==============================================================================

export type ErrorType = "iq_overflow" | "audio_drop" | "device_retry";

export interface ErrorEvent {
  type: ErrorType;
  capture_id: string;
  channel_id: string | null;
  timestamp: number;
  count: number;
  details?: Record<string, unknown>;
}

export interface ErrorStats {
  total: number;
  lastMinute: number;
  rate: number;
}

export interface HealthStatsMessage {
  type: "stats";
  data: Partial<Record<ErrorType, ErrorStats>>;
}

export interface HealthErrorMessage {
  type: "error";
  event: ErrorEvent;
}

export type HealthMessage = HealthStatsMessage | HealthErrorMessage;

// ==============================================================================
// State stream types for real-time capture/channel updates
// ==============================================================================

export type StateChangeAction =
  | "created"
  | "updated"
  | "deleted"
  | "started"
  | "stopped";
export type StateChangeType = "capture" | "channel" | "scanner";

export interface StateChangeMessage {
  type: StateChangeType;
  action: StateChangeAction;
  id: string;
  data: Capture | Channel | Scanner | null;
  timestamp: number;
}

export interface StateSnapshotMessage {
  type: "snapshot";
  captures: Capture[];
  channels: Channel[];
  scanners: Scanner[];
}

export interface StatePingMessage {
  type: "ping";
  timestamp: number;
}

export type StateMessage =
  | StateChangeMessage
  | StateSnapshotMessage
  | StatePingMessage;

// ==============================================================================
// Unified Radio Tab types (captures + trunking systems)
// ==============================================================================

export type RadioTabType = "capture" | "trunking";

export interface RadioTab {
  type: RadioTabType;
  id: string;
  name: string;
  deviceId: string; // Stable device ID for grouping
  deviceName: string;
  state: string;
  frequencyHz: number;
}

// ==============================================================================
// Device-centric tab types (Level 1 navigation)
// ==============================================================================

/** Overall device status derived from capture + trunking states */
export type DeviceStatus = "running" | "starting" | "stopped" | "failed";

/** Control channel state for trunking systems */
export type ControlChannelState = "searching" | "locked" | "lost";

/**
 * A device tab represents one physical SDR device.
 * Used for Level 1 navigation - one tab per device.
 */
export interface DeviceTab {
  /** Stable device ID (used for URL routing and selection) */
  deviceId: string;
  /** Display name for the device (nickname > shorthand > label) */
  deviceName: string;
  /** The capture associated with this device (if any) */
  capture: Capture | null;
  /** The trunking system associated with this device (if any) */
  trunkingSystem: TrunkingSystem | null;
  /** Overall device status - running if either capture or trunking is running */
  status: DeviceStatus;
  /** Whether this device has a capture configured */
  hasRadio: boolean;
  /** Whether this device has a trunking system configured */
  hasTrunking: boolean;
  /** Primary display frequency (from capture or trunking control channel) */
  frequencyHz: number;

  // Trunking-specific status fields for expressive display
  /** Control channel state (for trunking systems) */
  controlChannelState?: ControlChannelState;
  /** Number of active calls (for trunking systems) */
  activeCalls?: number;
  /** Whether manually locked to a frequency */
  isManuallyLocked?: boolean;
}

// ==============================================================================
// System Metrics Types (for System tab)
// ==============================================================================

/**
 * System-wide metrics from psutil.
 */
export interface SystemMetrics {
  timestamp: number;
  cpuPercent: number;
  cpuPerCore: number[];
  memoryUsedMb: number;
  memoryTotalMb: number;
  memoryPercent: number;
  temperatures: Record<string, number>;
}

/**
 * Per-capture metrics for monitoring.
 */
export interface CaptureMetrics {
  captureId: string;
  deviceId: string;
  state: string;
  iqOverflowCount: number;
  iqOverflowRate: number;
  channelCount: number;
  totalSubscribers: number;
  totalDrops: number;
  perfLoopMs: number;
  perfDspMs: number;
  perfFftMs: number;
}

/**
 * Log entry from backend.
 */
export interface LogEntry {
  timestamp: number;
  level: "DEBUG" | "INFO" | "WARNING" | "ERROR" | "CRITICAL";
  loggerName: string;
  message: string;
}

/**
 * WebSocket message types for /stream/system.
 * Note: ErrorEvent is already defined above in the error tracking types section.
 */
export interface SystemMetricsMessage {
  type: "metrics";
  system: SystemMetrics;
  captures: CaptureMetrics[];
}

export interface LogMessage {
  type: "log";
  entry: LogEntry;
}

export interface LogsSnapshotMessage {
  type: "logs_snapshot";
  entries: LogEntry[];
}

export interface ErrorMessage {
  type: "error";
  event: ErrorEvent;
}

export type SystemStreamMessage =
  | SystemMetricsMessage
  | LogMessage
  | LogsSnapshotMessage
  | ErrorMessage;
