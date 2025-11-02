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
}

export interface UpdateCaptureRequest {
  centerHz?: number;
  sampleRate?: number;
  gain?: number;
  bandwidth?: number;
  ppm?: number;
  antenna?: string;
}

export interface CreateCaptureRequest {
  deviceId?: string;
  centerHz: number;
  sampleRate: number;
  gain?: number;
  bandwidth?: number;
  ppm?: number;
  antenna?: string;
}

export interface CreateChannelRequest {
  mode: "wbfm" | "nfm" | "am";
  offsetHz?: number;
  audioRate?: number;
  squelchDb?: number | null;
}
