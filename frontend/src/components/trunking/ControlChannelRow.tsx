/**
 * ControlChannelRow - A single control channel row with enable, frequency, SNR, sync status, and lock controls
 */
import { Lock, Unlock } from "lucide-react";
import type { ControlChannel, HuntMode } from "../../types/trunking";
import {
  useSetHuntMode,
  useLockToChannel,
  useSetChannelEnabled,
} from "../../hooks/useTrunking";
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

  const rowClass = [
    isCurrentChannel ? "table-success" : "",
    isLockedChannel ? "border border-warning" : "",
    !channel.enabled ? "text-muted" : "",
  ].filter(Boolean).join(" ");

  return (
    <tr className={rowClass} style={{ fontSize: "0.75rem" }}>
      <td style={{ width: "40px", padding: "6px 12px" }}>
        <input
          type="checkbox"
          checked={channel.enabled}
          onChange={handleToggleEnabled}
          disabled={enabledMutation.isPending}
          className="form-check-input m-0"
          style={{ cursor: "pointer", width: "14px", height: "14px" }}
          title={
            channel.enabled
              ? "Disable channel from scan"
              : "Enable channel for scan"
          }
        />
      </td>
      <td className="font-monospace" style={{ width: "110px", padding: "6px 12px" }}>
        {formatFrequencyWithUnit(channel.frequencyHz)}
      </td>
      <td style={{ width: "100px", padding: "6px 12px" }}>
        <SignalBar snrDb={channel.snrDb} />
      </td>
      <td style={{ width: "60px", padding: "6px 12px", textAlign: "center" }}>
        <span
          className={`badge ${channel.syncDetected ? "bg-success" : "bg-secondary"}`}
          style={{ fontSize: "0.6rem", padding: "2px 4px" }}
          title={channel.syncDetected ? "P25 sync detected" : "No sync"}
        >
          {channel.syncDetected ? "SYNC" : "----"}
        </span>
      </td>
      <td style={{ width: "50px", padding: "6px 12px", textAlign: "center" }}>
        <button
          className={`btn btn-sm ${lockButtonClass}`}
          onClick={handleLock}
          disabled={isLocking || isPending || !channel.enabled}
          title={lockTitle}
          style={{ padding: "1px 4px", lineHeight: 1 }}
        >
          <LockIcon size={10} />
        </button>
      </td>
      <td
        style={{ padding: "6px 12px" }}
        title={formatFrequencyWithUnit(channel.frequencyHz)}
      >
        {channel.name ? (
          <span style={{ fontWeight: 500 }}>{channel.name}</span>
        ) : (
          <span className="text-muted fst-italic">—</span>
        )}
      </td>
    </tr>
  );
}

/**
 * Column headers for the control channel list
 */
export function ControlChannelHeaders() {
  return (
    <thead>
      <tr className="text-muted" style={{ fontSize: "0.65rem", whiteSpace: "nowrap" }}>
        <th style={{ width: "40px", padding: "6px 12px" }} title="Enable/disable channel">
          En
        </th>
        <th style={{ width: "110px", padding: "6px 12px" }}>Frequency</th>
        <th style={{ width: "100px", padding: "6px 12px" }}>SNR</th>
        <th style={{ width: "60px", padding: "6px 12px", textAlign: "center" }}>Sync</th>
        <th style={{ width: "50px", padding: "6px 12px", textAlign: "center" }} title="Lock to channel">
          Lock
        </th>
        <th style={{ padding: "6px 12px" }}>Site Name</th>
      </tr>
    </thead>
  );
}
