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
}

export interface Capture {
  id: string;
  deviceId: string;
  state: "created" | "running" | "stopped" | "failed";
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
}

export interface Channel {
  id: string;
  captureId: string;
  mode: "wbfm";
  state: "created" | "running" | "stopped";
  offsetHz: number;
  audioRate: number;
  squelchDb: number | null;
  signalPowerDb: number | null;
  rssiDb: number | null;  // Server-side RSSI from IQ
  snrDb: number | null;   // Server-side SNR estimate
}

export interface UpdateCaptureRequest {
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
}

export interface CreateChannelRequest {
  mode: "wbfm" | "nfm" | "am";
  offsetHz?: number;
  audioRate?: number;
  squelchDb?: number | null;
}

export interface UpdateChannelRequest {
  mode?: "wbfm" | "nfm" | "am";
  offsetHz?: number;
  audioRate?: number;
  squelchDb?: number | null;
}

export interface RecipeChannel {
  offsetHz: number;
  name: string;
  mode: string;
  squelchDb: number;
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
