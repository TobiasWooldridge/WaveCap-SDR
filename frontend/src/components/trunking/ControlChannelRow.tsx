/**
 * ControlChannelRow - A single control channel row with enable, frequency, SNR, sync status, and lock controls
 */
import { Lock, Unlock } from "lucide-react";
import type { ControlChannel, HuntMode } from "../../types/trunking";
import { useSetHuntMode, useLockToChannel, useSetChannelEnabled } from "../../hooks/useTrunking";
import { formatFrequencyWithUnit } from "../../utils/frequency";
import { SignalBar } from "./SignalBar";

interface ControlChannelRowProps {
  channel: ControlChannel;
  systemId: string;
  isLocking: boolean;
  huntMode: HuntMode;
}

export function ControlChannelRow({
  channel,
  systemId,
  isLocking,
  huntMode,
}: ControlChannelRowProps) {
  const lockMutation = useLockToChannel();
  const enabledMutation = useSetChannelEnabled();
  const setHuntModeMutation = useSetHuntMode();

  const handleLock = () => {
    if (channel.isLocked) {
      // Unlock by switching to auto mode
      setHuntModeMutation.mutate({ systemId, mode: "auto" });
    } else {
      // Lock to this channel
      lockMutation.mutate({ systemId, frequencyHz: channel.frequencyHz });
    }
  };

  const handleToggleEnabled = () => {
    enabledMutation.mutate({
      systemId,
      frequencyHz: channel.frequencyHz,
      enabled: !channel.enabled,
    });
  };

  const isCurrentChannel = channel.isCurrent;
  const isLockedChannel = channel.isLocked;
  const isPending = lockMutation.isPending || setHuntModeMutation.isPending;

  // Determine lock button state and tooltip
  let lockButtonClass = "btn-outline-secondary";
  let lockTitle = "Lock to this channel (manual mode)";
  let LockIcon = Lock;

  if (isLockedChannel) {
    lockButtonClass = "btn-warning";
    lockTitle = "Unlock (switch to auto mode)";
    LockIcon = Lock;
  } else if (isCurrentChannel && huntMode === "auto") {
    lockButtonClass = "btn-outline-success";
    lockTitle = "Currently active (click to lock)";
    LockIcon = Unlock;
  }

  return (
    <div
      className={`d-flex align-items-center gap-2 py-1 px-2 rounded ${
        isCurrentChannel ? "bg-success bg-opacity-25" : ""
      } ${isLockedChannel ? "border border-warning" : ""}`}
      style={{ fontSize: "0.8rem" }}
    >
      <input
        type="checkbox"
        checked={channel.enabled}
        onChange={handleToggleEnabled}
        disabled={enabledMutation.isPending}
        className="form-check-input m-0"
        style={{ cursor: "pointer", width: "14px", height: "14px" }}
        title={channel.enabled ? "Disable channel from scan" : "Enable channel for scan"}
      />
      <span
        className={`font-monospace ${!channel.enabled ? "text-muted" : ""}`}
        style={{ minWidth: "90px", fontSize: "0.75rem" }}
      >
        {formatFrequencyWithUnit(channel.frequencyHz)}
      </span>
      <SignalBar snrDb={channel.snrDb} />
      <span
        className={`badge ${channel.syncDetected ? "bg-success" : "bg-secondary"}`}
        style={{ fontSize: "0.6rem", minWidth: "32px", padding: "2px 4px" }}
        title={channel.syncDetected ? "P25 sync detected" : "No sync"}
      >
        {channel.syncDetected ? "SYNC" : "----"}
      </span>
      <button
        className={`btn btn-sm ${lockButtonClass}`}
        onClick={handleLock}
        disabled={isLocking || isPending || !channel.enabled}
        title={lockTitle}
        style={{ padding: "1px 4px", lineHeight: 1 }}
      >
        <LockIcon size={10} />
      </button>
    </div>
  );
}

/**
 * Column headers for the control channel list
 */
export function ControlChannelHeaders() {
  return (
    <div
      className="d-flex align-items-center gap-2 py-1 px-2 text-muted border-bottom mb-1"
      style={{ fontSize: "0.65rem" }}
    >
      <span style={{ width: "14px" }} title="Enable/disable channel for scanning">
        En
      </span>
      <span style={{ minWidth: "90px" }}>Frequency</span>
      <span style={{ width: "80px" }}>SNR</span>
      <span style={{ minWidth: "32px" }}>Sync</span>
      <span style={{ width: "24px" }} title="Lock to channel">
        Lock
      </span>
    </div>
  );
}
