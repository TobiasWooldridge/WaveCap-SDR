import { useState } from "react";
import { ChevronDown, ChevronRight, Network, Radio, MapPin, Server } from "lucide-react";
import type { TrunkingSystem } from "../../types/trunking";

interface NetworkConfigPanelProps {
  system: TrunkingSystem;
}

function formatHex(value: number | null | undefined, digits: number = 3): string {
  if (value === null || value === undefined) return "---";
  return `0x${value.toString(16).toUpperCase().padStart(digits, "0")}`;
}

function formatFrequency(hz: number): string {
  return (hz / 1_000_000).toFixed(4);
}

export function NetworkConfigPanel({ system }: NetworkConfigPanelProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  // Group control channels by status
  const currentCC = system.controlChannels.find(cc => cc.isCurrent);
  const backupCCs = system.controlChannels.filter(cc => !cc.isCurrent && cc.enabled);

  return (
    <div className="card">
      {/* Collapsible header */}
      <div
        className="card-header py-2 d-flex align-items-center gap-2"
        style={{ cursor: "pointer" }}
        onClick={() => setIsExpanded(!isExpanded)}
      >
        {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        <Network size={14} />
        <span className="fw-semibold" style={{ fontSize: "0.85rem" }}>Network Configuration</span>

        {/* Quick summary when collapsed */}
        {!isExpanded && (
          <span className="text-muted ms-auto" style={{ fontSize: "0.75rem" }}>
            NAC: {formatHex(system.nac)} |
            SYS: {formatHex(system.systemId)} |
            Site: {system.rfssId ?? "---"}-{system.siteId ?? "---"}
          </span>
        )}
      </div>

      {/* Expanded content */}
      {isExpanded && (
        <div className="card-body py-2" style={{ fontSize: "0.75rem" }}>
          <div className="row g-3">
            {/* Network Info */}
            <div className="col-md-4">
              <div className="d-flex align-items-center gap-1 mb-2 text-primary">
                <Network size={12} />
                <span className="fw-semibold">Network</span>
              </div>
              <table className="table table-sm table-borderless mb-0" style={{ fontSize: "0.7rem" }}>
                <tbody>
                  <tr>
                    <td className="text-muted py-0" style={{ width: "80px" }}>NAC</td>
                    <td className="py-0 font-monospace">{formatHex(system.nac, 3)}</td>
                  </tr>
                  <tr>
                    <td className="text-muted py-0">System ID</td>
                    <td className="py-0 font-monospace">{formatHex(system.systemId, 3)}</td>
                  </tr>
                  <tr>
                    <td className="text-muted py-0">Protocol</td>
                    <td className="py-0">
                      {system.protocol === "p25_phase1" ? "P25 Phase I" : "P25 Phase II"}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>

            {/* Site Info */}
            <div className="col-md-4">
              <div className="d-flex align-items-center gap-1 mb-2 text-success">
                <MapPin size={12} />
                <span className="fw-semibold">Site</span>
              </div>
              <table className="table table-sm table-borderless mb-0" style={{ fontSize: "0.7rem" }}>
                <tbody>
                  <tr>
                    <td className="text-muted py-0" style={{ width: "80px" }}>RFSS ID</td>
                    <td className="py-0 font-monospace">{system.rfssId ?? "---"}</td>
                  </tr>
                  <tr>
                    <td className="text-muted py-0">Site ID</td>
                    <td className="py-0 font-monospace">{system.siteId ?? "---"}</td>
                  </tr>
                  <tr>
                    <td className="text-muted py-0">Site Key</td>
                    <td className="py-0 font-monospace">
                      {system.systemId !== null && system.rfssId !== null && system.siteId !== null
                        ? `${formatHex(system.systemId, 3)}-${system.rfssId}-${system.siteId}`
                        : "---"}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>

            {/* Current Control Channel */}
            <div className="col-md-4">
              <div className="d-flex align-items-center gap-1 mb-2 text-warning">
                <Radio size={12} />
                <span className="fw-semibold">Control Channel</span>
              </div>
              <table className="table table-sm table-borderless mb-0" style={{ fontSize: "0.7rem" }}>
                <tbody>
                  <tr>
                    <td className="text-muted py-0" style={{ width: "80px" }}>Current</td>
                    <td className="py-0 font-monospace">
                      {currentCC ? `${formatFrequency(currentCC.frequencyHz)} MHz` : "---"}
                    </td>
                  </tr>
                  <tr>
                    <td className="text-muted py-0">SNR</td>
                    <td className="py-0">
                      {currentCC?.snrDb !== null && currentCC?.snrDb !== undefined
                        ? `${currentCC.snrDb.toFixed(1)} dB`
                        : "---"}
                    </td>
                  </tr>
                  <tr>
                    <td className="text-muted py-0">Status</td>
                    <td className="py-0">
                      <ControlChannelStateBadge state={system.controlChannelState} />
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* Control Channels List */}
          {backupCCs.length > 0 && (
            <div className="mt-3">
              <div className="d-flex align-items-center gap-1 mb-2 text-info">
                <Server size={12} />
                <span className="fw-semibold">Control Channels ({system.controlChannels.length})</span>
              </div>
              <div className="table-responsive">
                <table className="table table-sm table-striped mb-0" style={{ fontSize: "0.7rem" }}>
                  <thead>
                    <tr>
                      <th className="py-1">Frequency</th>
                      <th className="py-1">SNR</th>
                      <th className="py-1">Power</th>
                      <th className="py-1">Sync</th>
                      <th className="py-1">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {system.controlChannels
                      .filter(cc => cc.enabled)
                      .sort((a, b) => (b.snrDb ?? -99) - (a.snrDb ?? -99))
                      .map((cc) => (
                        <tr
                          key={cc.frequencyHz}
                          className={cc.isCurrent ? "table-success" : cc.isLocked ? "table-warning" : ""}
                        >
                          <td className="py-1 font-monospace">
                            {formatFrequency(cc.frequencyHz)} MHz
                          </td>
                          <td className="py-1">
                            {cc.snrDb !== null ? `${cc.snrDb.toFixed(1)} dB` : "---"}
                          </td>
                          <td className="py-1">
                            {cc.powerDb !== null ? `${cc.powerDb.toFixed(1)} dB` : "---"}
                          </td>
                          <td className="py-1">
                            {cc.syncDetected ? (
                              <span className="badge bg-success" style={{ fontSize: "0.55rem" }}>YES</span>
                            ) : (
                              <span className="badge bg-secondary" style={{ fontSize: "0.55rem" }}>NO</span>
                            )}
                          </td>
                          <td className="py-1">
                            {cc.isCurrent && (
                              <span className="badge bg-primary" style={{ fontSize: "0.55rem" }}>CURRENT</span>
                            )}
                            {cc.isLocked && !cc.isCurrent && (
                              <span className="badge bg-warning text-dark" style={{ fontSize: "0.55rem" }}>LOCKED</span>
                            )}
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function ControlChannelStateBadge({ state }: { state: string }) {
  switch (state) {
    case "locked":
      return <span className="badge bg-success" style={{ fontSize: "0.6rem" }}>Locked</span>;
    case "searching":
      return <span className="badge bg-warning text-dark" style={{ fontSize: "0.6rem" }}>Searching</span>;
    case "lost":
      return <span className="badge bg-danger" style={{ fontSize: "0.6rem" }}>Lost</span>;
    case "unlocked":
    default:
      return <span className="badge bg-secondary" style={{ fontSize: "0.6rem" }}>Unlocked</span>;
  }
}
