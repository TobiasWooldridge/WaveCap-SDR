/**
 * Shared status utilities for trunking systems
 */
import type { TrunkingSystem } from "../types/trunking";
import { formatFrequencyWithUnit } from "./frequency";

export interface UnifiedStatus {
  label: string;
  color: string;
  description: string;
}

export interface StatusBadge {
  text: string;
  color: string;
}

/**
 * Get a single unified status from the system and control channel states.
 * This provides a clear, non-redundant status to the user.
 */
export function getUnifiedSystemStatus(system: TrunkingSystem): UnifiedStatus {
  // Check system-level states first
  if (system.state === "stopped") {
    return { label: "Stopped", color: "secondary", description: "System is not running" };
  }
  if (system.state === "failed") {
    return { label: "Failed", color: "danger", description: "System encountered an error" };
  }
  if (system.state === "starting") {
    return { label: "Starting", color: "warning", description: "System is initializing..." };
  }

  // Check if manually locked (huntMode is manual with a locked frequency)
  if (system.huntMode === "manual" && system.lockedFrequencyHz) {
    return {
      label: "Locked",
      color: "info",
      description: `Locked to ${formatFrequencyWithUnit(system.lockedFrequencyHz)}`,
    };
  }

  // Check control channel state for running system
  switch (system.controlChannelState) {
    case "locked":
      return {
        label: "Synced",
        color: "success",
        description: `Receiving on ${formatFrequencyWithUnit(system.controlChannelFreqHz)}`,
      };
    case "searching":
      return {
        label: "Searching",
        color: "warning",
        description: "Looking for control channel...",
      };
    case "lost":
      return {
        label: "Lost",
        color: "danger",
        description: "Signal lost, hunting for new channel...",
      };
    default:
      return {
        label: "Running",
        color: "primary",
        description: "System is active",
      };
  }
}

/**
 * Get control channel status badge for compact display
 */
export function getControlChannelStatusBadge(system: TrunkingSystem): StatusBadge {
  if (system.lockedFrequencyHz) {
    return { text: "LOCKED", color: "warning" };
  }
  switch (system.controlChannelState) {
    case "locked":
      return { text: "SYNCED", color: "success" };
    case "searching":
      return { text: "HUNTING", color: "warning" };
    case "lost":
      return { text: "LOST", color: "danger" };
    default:
      return { text: "IDLE", color: "secondary" };
  }
}

/**
 * Get SNR measurement for current control channel from scanner stats
 */
export function getChannelSnr(system: TrunkingSystem): number | null {
  const scanner = system.stats.cc_scanner;
  if (!scanner || !scanner.current_channel_hz || !scanner.measurements) {
    return null;
  }

  const currentMHz = (scanner.current_channel_hz / 1e6).toFixed(4);
  const key = `${currentMHz}_MHz`;
  const measurement = scanner.measurements[key];

  return measurement ? measurement.snr_db : null;
}
