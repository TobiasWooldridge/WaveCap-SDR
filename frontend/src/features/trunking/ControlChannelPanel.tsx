import { useState } from "react";
import { Radio, RefreshCw, ChevronDown, ChevronRight, HelpCircle } from "lucide-react";
import type { TrunkingSystem, HuntMode } from "../../types/trunking";
import { useSetHuntMode, useTriggerScan } from "../../hooks/useTrunking";
import Flex from "../../components/primitives/Flex.react";
import { formatFrequencyWithUnit } from "../../utils/frequency";
import { getControlChannelStatusBadge } from "../../utils/trunkingStatus";
import { ControlChannelRow, ControlChannelHeaders, HuntModeHelp } from "../../components/trunking";
import { FrequencyDisplay } from "../../components/primitives/FrequencyDisplay.react";

interface ControlChannelPanelProps {
  system: TrunkingSystem;
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

  // Sort channels by frequency
  const channels = [...(system.controlChannels || [])].sort(
    (a, b) => a.frequencyHz - b.frequencyHz
  );
  if (channels.length === 0) {
    return null;
  }

  const status = getControlChannelStatusBadge(system);
  const currentFreq = system.controlChannelFreqHz;
  const enabledCount = channels.filter(c => c.enabled).length;
  const lockedChannel = channels.find(c => c.isLocked);
  const currentChannel =
    channels.find((c) => c.isCurrent) ||
    (currentFreq
      ? channels.find((c) => Math.abs(c.frequencyHz - currentFreq) < 1000)
      : undefined) ||
    lockedChannel;
  const currentName = currentChannel?.name;

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
                ? `Locked to ${formatFrequencyWithUnit(lockedChannel.frequencyHz)}`
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
              <FrequencyDisplay
                frequencyHz={currentFreq}
                decimals={4}
                name={currentName}
                unit="MHz"
              />
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
            style={{ width: "auto", minWidth: "75px", fontSize: "0.7rem", padding: "1px 22px 1px 6px", height: "22px" }}
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

          <table className="table table-sm table-borderless mb-0" style={{ tableLayout: "fixed" }}>
            <ControlChannelHeaders />
            <tbody>
              {channels.map((channel) => (
                <ControlChannelRow
                  key={channel.frequencyHz}
                  channel={channel}
                  systemId={system.id}
                  isLocking={setHuntModeMutation.isPending}
                  huntMode={system.huntMode}
                />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
