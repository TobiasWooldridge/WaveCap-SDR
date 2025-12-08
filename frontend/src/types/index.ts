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
  nickname?: string | null;  // User-provided nickname
  shorthand?: string | null;  // Auto-generated shorthand name (e.g., "RTL-SDR", "SDRplay RSPdx")
}

export interface Capture {
  id: string;
  deviceId: string;
  state: "created" | "starting" | "running" | "stopping" | "stopped" | "failed" | "error";
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
  name: string | null;  // User-provided name
  autoName: string | null;  // Auto-generated contextual name
  // FFT/Spectrum settings
  fftFps: number;  // Target FFT frames per second
  fftSize: number;  // FFT bin count (512, 1024, 2048, 4096)
  // Error indicators
  iqOverflowCount: number;
  iqOverflowRate: number;  // Overflows per second
  retryAttempt: number | null;  // Current retry attempt (null if not retrying)
  retryMaxAttempts: number | null;
  retryDelay: number | null;  // Delay in seconds before next retry
}

export interface Channel {
  id: string;
  captureId: string;
  mode: "wbfm" | "nbfm" | "am" | "ssb" | "raw" | "p25" | "dmr";
  state: "created" | "running" | "stopped";
  offsetHz: number;
  audioRate: number;
  squelchDb: number | null;
  name: string | null;  // User-provided name
  autoName: string | null;  // Auto-generated contextual name (e.g., "Marine Ch 16")
  signalPowerDb: number | null;
  rssiDb: number | null;  // Server-side RSSI from IQ
  snrDb: number | null;   // Server-side SNR estimate

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
  notchFrequencies: number[];  // List of frequencies to notch out (Hz)

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
  audioDropRate: number;  // Drops per second
}

// RDS (Radio Data System) data for FM broadcast
export interface RDSData {
  piCode: string | null;  // Program Identification (hex string like "A1B2")
  psName: string | null;  // Program Service name (8 chars, station name)
  radioText: string | null;  // Radio Text (up to 64 chars)
  pty: number;  // Program Type code
  ptyName: string;  // Program Type name (e.g., "Rock", "News")
  ta: boolean;  // Traffic Announcement flag
  tp: boolean;  // Traffic Program flag
  ms: boolean;  // Music/Speech switch (true = Music)
}

// POCSAG pager message
export interface POCSAGMessage {
  address: number;  // 21-bit capcode
  function: number;  // Function code (0-3)
  messageType: "numeric" | "alpha" | "alert_only" | "alpha_2";
  message: string;  // Decoded message content
  timestamp: number;  // Unix timestamp
  baudRate: number;  // 512, 1200, or 2400
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
  fftFps?: number;  // 1-60 FPS
  fftSize?: number;  // 512, 1024, 2048, 4096
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
  fftFps?: number;  // 1-60 FPS
  fftSize?: number;  // 512, 1024, 2048, 4096
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
  mode: string;
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

export type StateChangeAction = "created" | "updated" | "deleted" | "started" | "stopped";
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

export type StateMessage = StateChangeMessage | StateSnapshotMessage | StatePingMessage;
