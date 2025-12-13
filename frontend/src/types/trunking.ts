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

export type CallState = "starting" | "recording" | "hold" | "ended";

export interface TrunkingSystem {
  id: string;
  name: string;
  protocol: TrunkingProtocol;
  deviceId: string | null;
  state: TrunkingSystemState;
  controlChannelState: ControlChannelState;
  controlChannelFreqHz: number | null;
  nac: number | null;
  systemId: number | null;
  rfssId: number | null;
  siteId: number | null;
  decodeRate: number;
  activeCalls: number;
  stats: TrunkingStats;
}

export interface TrunkingStats {
  tsbk_count: number;
  grant_count: number;
  calls_total: number;
  recorders_idle: number;
  recorders_active: number;
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

// WebSocket event types

export type TrunkingEventType =
  | "snapshot"
  | "system_update"
  | "call_start"
  | "call_update"
  | "call_end"
  | "talkgroup_update";

export interface TrunkingSnapshotEvent {
  type: "snapshot";
  systems: TrunkingSystem[];
  activeCalls: ActiveCall[];
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

export type TrunkingEvent =
  | TrunkingSnapshotEvent
  | TrunkingSystemUpdateEvent
  | TrunkingCallStartEvent
  | TrunkingCallUpdateEvent
  | TrunkingCallEndEvent;
