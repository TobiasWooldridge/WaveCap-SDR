import { Lock, Radio, RefreshCw } from "lucide-react";
import type { TrunkingSystem, HuntMode, ControlChannel } from "../../types/trunking";
import { useSetHuntMode, useLockToChannel, useSetChannelEnabled, useTriggerScan } from "../../hooks/useTrunking";
import Flex from "../../components/primitives/Flex.react";

interface ControlChannelPanelProps {
  system: TrunkingSystem;
}

function formatFrequency(hz: number): string {
  return (hz / 1_000_000).toFixed(4) + " MHz";
}

function SignalBar({ snrDb, maxDb = 30 }: { snrDb: number | null; maxDb?: number }) {
  if (snrDb === null) return <span className="text-muted">---</span>;

  // Normalize SNR to 0-100%
  const percent = Math.min(100, Math.max(0, (snrDb / maxDb) * 100));

  // Color based on signal strength
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
}: {
  channel: ControlChannel;
  systemId: string;
  isLocking: boolean;
}) {
  const lockMutation = useLockToChannel();
  const enabledMutation = useSetChannelEnabled();

  const handleLock = () => {
    lockMutation.mutate({ systemId, frequencyHz: channel.frequencyHz });
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

  return (
    <div
      className={`d-flex align-items-center gap-2 py-1 px-2 rounded ${
        isCurrentChannel ? "bg-success bg-opacity-25" : ""
      }`}
      style={{ fontSize: "0.85rem" }}
    >
      {/* Enable/Disable Checkbox */}
      <input
        type="checkbox"
        checked={channel.enabled}
        onChange={handleToggleEnabled}
        disabled={enabledMutation.isPending}
        className="form-check-input m-0"
        style={{ cursor: "pointer" }}
        title={channel.enabled ? "Disable this channel" : "Enable this channel"}
      />

      {/* Frequency */}
      <span
        className={`font-monospace ${!channel.enabled ? "text-muted" : ""}`}
        style={{ minWidth: "100px" }}
      >
        {formatFrequency(channel.frequencyHz)}
      </span>

      {/* Signal Bar */}
      <SignalBar snrDb={channel.snrDb} />

      {/* Sync Indicator */}
      <span
        className={`badge ${channel.syncDetected ? "bg-success" : "bg-secondary"}`}
        style={{ fontSize: "0.65rem", minWidth: "38px" }}
      >
        {channel.syncDetected ? "SYNC" : "----"}
      </span>

      {/* Lock Button */}
      <button
        className={`btn btn-sm ${isLockedChannel ? "btn-warning" : "btn-outline-secondary"}`}
        onClick={handleLock}
        disabled={isLocking || lockMutation.isPending || !channel.enabled}
        title={isLockedChannel ? "Locked to this channel" : "Lock to this channel"}
        style={{ padding: "0.15rem 0.4rem" }}
      >
        <Lock size={12} />
      </button>
    </div>
  );
}

export function ControlChannelPanel({ system }: ControlChannelPanelProps) {
  const setHuntModeMutation = useSetHuntMode();
  const triggerScanMutation = useTriggerScan();

  const handleModeChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const mode = e.target.value as HuntMode;
    setHuntModeMutation.mutate({ systemId: system.id, mode });
  };

  const handleScanNow = () => {
    triggerScanMutation.mutate(system.id);
  };

  // Get channels - use system.controlChannels if available
  const channels = system.controlChannels || [];

  // Don't render if no channels configured
  if (channels.length === 0) {
    return null;
  }

  return (
    <div className="card mt-2">
      <div className="card-header py-2 d-flex align-items-center justify-content-between">
        <Flex align="center" gap={2}>
          <Radio size={14} className="text-info" />
          <span className="fw-semibold" style={{ fontSize: "0.85rem" }}>
            Control Channels
          </span>
        </Flex>

        <Flex align="center" gap={2}>
          {/* Hunt Mode Selector */}
          <select
            className="form-select form-select-sm"
            value={system.huntMode}
            onChange={handleModeChange}
            disabled={setHuntModeMutation.isPending}
            style={{ width: "auto", fontSize: "0.75rem" }}
          >
            <option value="auto">Auto</option>
            <option value="manual">Manual</option>
            <option value="scan_once">Scan Once</option>
          </select>

          {/* Scan Now Button */}
          <button
            className="btn btn-sm btn-outline-primary d-flex align-items-center gap-1"
            onClick={handleScanNow}
            disabled={triggerScanMutation.isPending}
            style={{ fontSize: "0.75rem" }}
          >
            <RefreshCw
              size={12}
              className={triggerScanMutation.isPending ? "spin" : ""}
            />
            Scan
          </button>
        </Flex>
      </div>

      <div className="card-body py-2">
        {/* Channel List */}
        <div className="d-flex flex-column gap-1">
          {channels.map((channel) => (
            <ChannelRow
              key={channel.frequencyHz}
              channel={channel}
              systemId={system.id}
              isLocking={setHuntModeMutation.isPending}
            />
          ))}
        </div>

        {/* Locked indicator */}
        {system.lockedFrequencyHz && (
          <div className="mt-2 text-center">
            <small className="text-warning">
              <Lock size={10} className="me-1" />
              Locked: {formatFrequency(system.lockedFrequencyHz)}
            </small>
          </div>
        )}
      </div>
    </div>
  );
}
