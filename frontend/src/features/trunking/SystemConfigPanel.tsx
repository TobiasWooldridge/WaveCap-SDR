import { useState } from "react";
import {
  Radio,
  RefreshCw,
  ChevronDown,
  ChevronRight,
  HelpCircle,
  Network,
  MapPin,
} from "lucide-react";
import type { TrunkingSystem, HuntMode } from "../../types/trunking";
import { useSetHuntMode, useTriggerScan } from "../../hooks/useTrunking";
import Flex from "../../components/primitives/Flex.react";
import { formatFrequencyWithUnit } from "../../utils/frequency";
import { formatHex } from "../../utils/formatting";
import { getControlChannelStatusBadge } from "../../utils/trunkingStatus";
import {
  ControlChannelRow,
  ControlChannelHeaders,
  HuntModeHelp,
} from "../../components/trunking";
import { FrequencyDisplay } from "../../components/primitives/FrequencyDisplay.react";

interface SystemConfigPanelProps {
  system: TrunkingSystem;
}

export function SystemConfigPanel({ system }: SystemConfigPanelProps) {
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
  const status = getControlChannelStatusBadge(system);
  const currentFreq = system.controlChannelFreqHz;
  const enabledCount = channels.filter((c) => c.enabled).length;
  const lockedChannel = channels.find((c) => c.isLocked);
  const currentChannel =
    channels.find((c) => c.isCurrent) ||
    (currentFreq
      ? channels.find((c) => Math.abs(c.frequencyHz - currentFreq) < 1000)
      : undefined) ||
    lockedChannel;
  const currentName = currentChannel?.name;

  return (
    <div className="card">
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
          <Network size={12} className="text-info" />
          <span style={{ fontSize: "0.8rem" }}>System Configuration</span>
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
          {/* Quick info when collapsed */}
          {!isExpanded && (
            <span
              className="d-none d-md-inline-flex align-items-center gap-2"
              style={{ fontSize: "0.7rem" }}
            >
              {/* Current frequency - most important */}
              {currentFreq && (
                <span className="font-monospace fw-semibold">
                  <FrequencyDisplay
                    frequencyHz={currentFreq}
                    decimals={4}
                    name={currentName}
                    unit="MHz"
                  />
                </span>
              )}
              {/* SNR if available */}
              {channels.find((c) => c.isCurrent)?.snrDb != null && (
                <span className="text-muted">
                  {channels.find((c) => c.isCurrent)?.snrDb?.toFixed(0)} dB
                </span>
              )}
              {/* Decode rate */}
              {system.decodeRate > 0 && (
                <span className="text-muted">
                  {system.decodeRate.toFixed(0)}/s
                </span>
              )}
              {/* NAC only if known */}
              {system.nac != null && (
                <span
                  className="badge bg-dark"
                  style={{ fontSize: "0.6rem", padding: "1px 3px" }}
                >
                  NAC {formatHex(system.nac, 3)}
                </span>
              )}
              {/* Site only if known */}
              {system.rfssId != null && system.siteId != null && (
                <span
                  className="badge bg-secondary"
                  style={{ fontSize: "0.6rem", padding: "1px 3px" }}
                >
                  Site {system.rfssId}-{system.siteId}
                </span>
              )}
            </span>
          )}
        </Flex>

        {/* Hunt mode controls always visible */}
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
            style={{
              width: "auto",
              fontSize: "0.7rem",
              padding: "1px 20px 1px 4px",
              height: "22px",
            }}
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
        <div className="card-body py-2 px-2">
          {showHelp && <HuntModeHelp />}

          {/* Network & Site Info */}
          <div className="row g-2 mb-2" style={{ fontSize: "0.7rem" }}>
            {/* Network Info - only show if we have data */}
            {(system.nac != null || system.systemId != null) && (
              <div className="col-6 col-md-4">
                <div className="d-flex align-items-center gap-1 mb-1 text-primary">
                  <Network size={10} />
                  <span className="fw-semibold">Network</span>
                </div>
                <div className="d-flex flex-wrap gap-2">
                  {system.nac != null && (
                    <span>
                      <span className="text-muted">NAC:</span>{" "}
                      <span className="font-monospace">
                        {formatHex(system.nac, 3)}
                      </span>
                    </span>
                  )}
                  {system.systemId != null && (
                    <span>
                      <span className="text-muted">SYS:</span>{" "}
                      <span className="font-monospace">
                        {formatHex(system.systemId, 3)}
                      </span>
                    </span>
                  )}
                </div>
              </div>
            )}

            {/* Site Info - only show if we have data */}
            {(system.rfssId != null || system.siteId != null) && (
              <div className="col-6 col-md-4">
                <div className="d-flex align-items-center gap-1 mb-1 text-success">
                  <MapPin size={10} />
                  <span className="fw-semibold">Site</span>
                </div>
                <div className="d-flex flex-wrap gap-2">
                  {system.rfssId != null && (
                    <span>
                      <span className="text-muted">RFSS:</span>{" "}
                      <span className="font-monospace">{system.rfssId}</span>
                    </span>
                  )}
                  {system.siteId != null && (
                    <span>
                      <span className="text-muted">Site:</span>{" "}
                      <span className="font-monospace">{system.siteId}</span>
                    </span>
                  )}
                </div>
              </div>
            )}

            {/* Current CC */}
            <div className="col-12 col-md-4">
              <div className="d-flex align-items-center gap-1 mb-1 text-warning">
                <Radio size={10} />
                <span className="fw-semibold">Control Channel</span>
              </div>
              <div className="d-flex flex-wrap gap-2 align-items-center">
                {currentFreq ? (
                  <>
                    <span className="font-monospace fw-semibold">
                      <FrequencyDisplay
                        frequencyHz={currentFreq}
                        decimals={4}
                        name={currentName}
                        unit="MHz"
                      />
                    </span>
                    {channels.find((c) => c.isCurrent)?.snrDb != null && (
                      <span className="text-muted">
                        {channels.find((c) => c.isCurrent)?.snrDb?.toFixed(1)}{" "}
                        dB SNR
                      </span>
                    )}
                    {system.decodeRate > 0 && (
                      <span
                        className="badge bg-info"
                        style={{ fontSize: "0.6rem" }}
                      >
                        {system.decodeRate.toFixed(0)} msg/s
                      </span>
                    )}
                  </>
                ) : (
                  <span className="text-muted fst-italic">Searching...</span>
                )}
              </div>
            </div>
          </div>

          {/* Control Channels List */}
          {channels.length > 0 && (
            <div className="border-top pt-2 mt-1">
              <div className="d-flex align-items-center justify-content-between mb-1">
                <span className="text-muted" style={{ fontSize: "0.7rem" }}>
                  Control Channels ({enabledCount}/{channels.length} enabled)
                </span>
              </div>

              <table
                className="table table-sm table-borderless mb-0"
                style={{ tableLayout: "fixed" }}
              >
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
      )}
    </div>
  );
}
