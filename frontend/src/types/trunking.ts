// P25 Trunking system types

export type TrunkingProtocol = "p25_phase1" | "p25_phase2";

export type TrunkingSystemState =
  | "stopped"
  | "starting"
  | "searching"
  | "syncing"
  | "running"
  | "failed";

export type ControlChannelState = "unlocked" | "searching" | "locked" | "lost";

export type HuntMode = "auto" | "manual" | "scan_once";

export type CallState = "starting" | "recording" | "hold" | "ended";

export interface TrunkingSystem {
  id: string;
  name: string;
  protocol: TrunkingProtocol;
  deviceId: string | null;
  state: TrunkingSystemState;
  controlChannelState: ControlChannelState;
  controlChannelFreqHz: number | null;
  centerHz: number; // SDR center frequency (auto-managed by trunking)
  nac: number | null;
  systemId: number | null;
  rfssId: number | null;
  siteId: number | null;
  decodeRate: number;
  activeCalls: number;
  stats: TrunkingStats;
  // Hunt mode control
  huntMode: HuntMode;
  lockedFrequencyHz: number | null;
  controlChannels: ControlChannel[];
}

export interface ControlChannelMeasurement {
  power_db: number;
  snr_db: number;
  sync_detected: boolean;
}

export interface ControlChannel {
  frequencyHz: number;
  name: string;
  enabled: boolean;
  isCurrent: boolean;
  isLocked: boolean;
  snrDb: number | null;
  powerDb: number | null;
  syncDetected: boolean;
  measurementTime: number | null;
}

export interface ControlChannelScannerStats {
  channels_configured: number;
  channels_measured: number;
  last_scan_time: number;
  current_channel_hz: number | null;
  measurements: Record<string, ControlChannelMeasurement>;
}

export interface TrunkingStats {
  tsbk_count: number;
  grant_count: number;
  calls_total: number;
  recorders_idle: number;
  recorders_active: number;
  initial_scan_complete?: boolean;
  cc_scanner?: ControlChannelScannerStats;
}

export interface Talkgroup {
  tgid: number;
  name: string;
  alphaTag: string;
  category: string;
  priority: number;
  record: boolean;
  monitor: boolean;
}

export interface ActiveCall {
  id: string;
  talkgroupId: number;
  talkgroupName: string;
  sourceId: number | null;
  frequencyHz: number;
  channelId: number;
  state: CallState;
  startTime: number;
  lastActivityTime: number;
  encrypted: boolean;
  audioFrames: number;
  durationSeconds: number;
  recorderId: string | null;
}

export interface VocoderStatus {
  imbe: {
    available: boolean;
    message: string;
  };
  ambe2: {
    available: boolean;
    message: string;
  };
  anyAvailable: boolean;
}

// Request/Response types

export interface CreateSystemRequest {
  id: string;
  name: string;
  protocol: TrunkingProtocol;
  controlChannels: number[];
  centerHz: number;
  sampleRate?: number;
  deviceId?: string | null;
  maxVoiceRecorders?: number;
  recordingPath?: string | null;
  recordUnknown?: boolean;
  talkgroups?: Record<
    string,
    {
      tgid: number;
      name: string;
      alphaTag?: string | null;
      category?: string | null;
      priority?: number;
      record?: boolean;
      monitor?: boolean;
    }
  >;
}

export interface AddTalkgroupRequest {
  tgid: number;
  name: string;
  alphaTag?: string | null;
  category?: string | null;
  priority?: number;
  record?: boolean;
  monitor?: boolean;
}

// Recipe/template for pre-configured trunking systems
export interface TrunkingRecipe {
  id: string;
  name: string;
  description?: string;
  category: string;
  protocol: TrunkingProtocol;
  controlChannels: number[]; // Frequencies in Hz
  centerHz: number;
  sampleRate: number;
  gain?: number;
  talkgroupCount: number;
}

// WebSocket event types

// Decoded P25 message
export interface P25Message {
  timestamp: number;
  opcode: number;
  opcodeName: string;
  nac: number | null;
  summary: string;
}

export type TrunkingEventType =
  | "snapshot"
  | "system_update"
  | "call_start"
  | "call_update"
  | "call_end"
  | "message"
  | "talkgroup_update";

export interface CallHistoryEntry extends ActiveCall {
  endReason?: string;
  endTime?: number;
  systemId?: string;
}

export interface TrunkingSnapshotEvent {
  type: "snapshot";
  systems: TrunkingSystem[];
  activeCalls: ActiveCall[];
  messages?: P25Message[]; // Buffered messages from server
  callHistory?: CallHistoryEntry[]; // Buffered call history from server
}

export interface TrunkingSystemUpdateEvent {
  type: "system_update";
  systemId: string;
  system: TrunkingSystem;
}

export interface TrunkingCallStartEvent {
  type: "call_start";
  systemId: string;
  call: ActiveCall;
}

export interface TrunkingCallUpdateEvent {
  type: "call_update";
  systemId: string;
  call: ActiveCall;
}

export interface TrunkingCallEndEvent {
  type: "call_end";
  systemId: string;
  callId: string;
  call: ActiveCall;
}

export interface TrunkingMessageEvent {
  type: "message";
  systemId: string;
  message: P25Message;
}

export type TrunkingEvent =
  | TrunkingSnapshotEvent
  | TrunkingSystemUpdateEvent
  | TrunkingCallStartEvent
  | TrunkingCallUpdateEvent
  | TrunkingCallEndEvent
  | TrunkingMessageEvent;

// Voice stream types

export type VoiceStreamState =
  | "created"
  | "starting"
  | "active"
  | "silent"
  | "ended";

export interface RadioLocation {
  unitId: number;
  latitude: number;
  longitude: number;
  altitude: number | null;
  speed: number | null;
  heading: number | null;
  accuracy: number | null;
  timestamp: number;
  ageSeconds: number;
  source: "lrrp" | "elc" | "gps_tsbk" | "unknown";
}

export interface VoiceStream {
  id: string;
  systemId: string;
  callId: string;
  recorderId: string;
  state: VoiceStreamState;
  talkgroupId: number;
  talkgroupName: string;
  sourceId: number | null;
  sourceLocation: RadioLocation | null;
  encrypted: boolean;
  startTime: number;
  durationSeconds: number;
  silenceSeconds: number;
  audioFrameCount: number;
  audioBytesSent: number;
  subscriberCount: number;
}

// Voice stream WebSocket message types

export interface VoiceAudioMessage {
  type: "audio";
  streamId: string;
  systemId: string;
  callId: string;
  recorderId: string;
  talkgroupId: number;
  talkgroupName: string;
  sourceId: number | null;
  sourceLocation: RadioLocation | null;
  timestamp: number;
  encrypted: boolean;
  sampleRate: number;
  frameNumber: number;
  format: "pcm16" | "f32";
  audio: string; // Base64 encoded audio
}

export interface VoiceStreamEndedMessage {
  type: "ended";
  streamId: string;
}

export type VoiceStreamMessage = VoiceAudioMessage | VoiceStreamEndedMessage;
