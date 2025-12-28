import { useState } from "react";
import { Lock, Unlock, Radio, RefreshCw, ChevronDown, ChevronRight, HelpCircle } from "lucide-react";
import type { TrunkingSystem, HuntMode, ControlChannel } from "../../types/trunking";
import { useSetHuntMode, useLockToChannel, useSetChannelEnabled, useTriggerScan } from "../../hooks/useTrunking";
import Flex from "../../components/primitives/Flex.react";

interface ControlChannelPanelProps {
  system: TrunkingSystem;
}

function formatFrequency(hz: number): string {
  return (hz / 1_000_000).toFixed(4) + " MHz";
}

function formatFrequencyShort(hz: number): string {
  return (hz / 1_000_000).toFixed(3);
}

function SignalBar({ snrDb, maxDb = 30 }: { snrDb: number | null; maxDb?: number }) {
  if (snrDb === null) return <span className="text-muted">---</span>;

  const percent = Math.min(100, Math.max(0, (snrDb / maxDb) * 100));

  let color = "bg-danger";
  if (percent > 60) color = "bg-success";
  else if (percent > 30) color = "bg-warning";

  return (
    <div className="d-flex align-items-center gap-1" style={{ width: "80px" }}>
      <div
        className="progress"
        style={{ height: "8px", width: "50px", backgroundColor: "#333" }}
      >
        <div
          className={`progress-bar ${color}`}
          style={{ width: `${percent}%` }}
        />
      </div>
      <small className="text-muted" style={{ fontSize: "0.7rem", minWidth: "35px" }}>
        {snrDb.toFixed(0)} dB
      </small>
    </div>
  );
}

function ChannelRow({
  channel,
  systemId,
  isLocking,
  huntMode,
}: {
  channel: ControlChannel;
  systemId: string;
  isLocking: boolean;
  huntMode: HuntMode;
}) {
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
        {formatFrequency(channel.frequencyHz)}
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

function getStatusBadge(system: TrunkingSystem): { text: string; color: string } {
  if (system.lockedFrequencyHz) {
    return { text: "LOCKED", color: "warning" };
  }
  switch (system.controlChannelState) {
    case "locked":
      return { text: "LOCKED", color: "success" };
    case "searching":
      return { text: "HUNTING", color: "warning" };
    case "lost":
      return { text: "LOST", color: "danger" };
    default:
      return { text: "IDLE", color: "secondary" };
  }
}

function HuntModeHelp() {
  return (
    <div className="text-muted small mb-2 px-1" style={{ fontSize: "0.7rem", lineHeight: 1.3 }}>
      <p className="mb-1">
        <strong>Control channels</strong> carry P25 trunking signaling. The system hunts for
        the best channel based on signal quality.
      </p>
      <p className="mb-1">
        <strong>Modes:</strong>{" "}
        <em>Auto</em> switches channels if signal degrades.{" "}
        <em>Manual</em> locks to one channel.{" "}
        <em>Scan Once</em> finds the best channel then stays.
      </p>
      <p className="mb-0">
        Use the <Lock size={10} className="mx-1" /> button to lock/unlock a specific channel.
      </p>
    </div>
  );
}

export function ControlChannelPanel({ system }: ControlChannelPanelProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [showHelp, setShowHelp] = useState(false);
  const setHuntModeMutation = useSetHuntMode();
  const triggerScanMutation = useTriggerScan();

  const handleModeChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    e.stopPropagation();
    const mode = e.target.value as HuntMode;
    setHuntModeMutation.mutate({ systemId: system.id, mode });
  };

  const handleScanNow = (e: React.MouseEvent) => {
    e.stopPropagation();
    triggerScanMutation.mutate(system.id);
  };

  const handleHelpClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    setShowHelp(!showHelp);
  };

  const channels = system.controlChannels || [];
  if (channels.length === 0) {
    return null;
  }

  const status = getStatusBadge(system);
  const currentFreq = system.controlChannelFreqHz;
  const enabledCount = channels.filter(c => c.enabled).length;
  const lockedChannel = channels.find(c => c.isLocked);

  return (
    <div className="card mt-2">
      <div
        className="card-header py-1 px-2 d-flex align-items-center justify-content-between"
        style={{ cursor: "pointer", minHeight: "32px" }}
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <Flex align="center" gap={1}>
          {isExpanded ? (
            <ChevronDown size={14} className="text-muted" />
          ) : (
            <ChevronRight size={14} className="text-muted" />
          )}
          <Radio size={12} className="text-info" />
          <span style={{ fontSize: "0.8rem" }}>Control Channel</span>
          <span
            className={`badge bg-${status.color}`}
            style={{ fontSize: "0.6rem", padding: "2px 4px" }}
            title={
              status.text === "LOCKED" && lockedChannel
                ? `Locked to ${formatFrequency(lockedChannel.frequencyHz)}`
                : status.text === "HUNTING"
                ? "Searching for best control channel"
                : status.text === "LOST"
                ? "Control channel signal lost, hunting..."
                : "Connected to control channel"
            }
          >
            {status.text}
          </span>
          {currentFreq && (
            <span className="font-monospace text-muted" style={{ fontSize: "0.75rem" }}>
              {formatFrequencyShort(currentFreq)}
            </span>
          )}
          <span className="text-muted" style={{ fontSize: "0.7rem" }}>
            ({enabledCount}/{channels.length})
          </span>
        </Flex>

        <Flex align="center" gap={1} onClick={(e) => e.stopPropagation()}>
          <button
            className={`btn btn-sm ${showHelp ? "btn-info" : "btn-outline-secondary"} d-flex align-items-center`}
            onClick={handleHelpClick}
            style={{ padding: "1px 4px", height: "22px" }}
            title="Show help"
          >
            <HelpCircle size={10} />
          </button>
          <select
            className="form-select form-select-sm"
            value={system.huntMode}
            onChange={handleModeChange}
            disabled={setHuntModeMutation.isPending}
            style={{ width: "auto", fontSize: "0.7rem", padding: "1px 20px 1px 4px", height: "22px" }}
            title="Hunt mode: Auto switches channels automatically, Manual locks to current, Scan Once finds best then stays"
          >
            <option value="auto">Auto</option>
            <option value="manual">Manual</option>
            <option value="scan_once">Scan Once</option>
          </select>
          <button
            className="btn btn-sm btn-outline-primary d-flex align-items-center"
            onClick={handleScanNow}
            disabled={triggerScanMutation.isPending}
            style={{ fontSize: "0.7rem", padding: "1px 4px", height: "22px" }}
            title="Scan all enabled channels and measure signal quality"
          >
            <RefreshCw
              size={10}
              className={triggerScanMutation.isPending ? "spin" : ""}
            />
          </button>
        </Flex>
      </div>

      {isExpanded && (
        <div className="card-body py-1 px-2">
          {showHelp && <HuntModeHelp />}

          {/* Column headers */}
          <div className="d-flex align-items-center gap-2 py-1 px-2 text-muted border-bottom mb-1" style={{ fontSize: "0.65rem" }}>
            <span style={{ width: "14px" }} title="Enable/disable channel for scanning">En</span>
            <span style={{ minWidth: "90px" }}>Frequency</span>
            <span style={{ width: "80px" }}>SNR</span>
            <span style={{ minWidth: "32px" }}>Sync</span>
            <span style={{ width: "24px" }} title="Lock to channel">Lock</span>
          </div>

          <div className="d-flex flex-column">
            {channels.map((channel) => (
              <ChannelRow
                key={channel.frequencyHz}
                channel={channel}
                systemId={system.id}
                isLocking={setHuntModeMutation.isPending}
                huntMode={system.huntMode}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
