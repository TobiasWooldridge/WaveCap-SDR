import { useMemo, useCallback } from "react";
import { Copy } from "lucide-react";
import type { TrunkingSystem, ActiveCall, P25Message } from "../../types/trunking";

interface ActivitySummaryProps {
  system: TrunkingSystem;
  activeCalls: ActiveCall[];
  messages: P25Message[];
}

function formatFrequency(hz: number): string {
  return (hz / 1_000_000).toFixed(4);
}

function formatHex(value: number | null | undefined, digits: number = 3): string {
  if (value === null || value === undefined) return "(pending)";
  return `0x${value.toString(16).toUpperCase().padStart(digits, "0")}`;
}

function formatNumber(value: number): string {
  return value.toLocaleString();
}

export function ActivitySummary({ system, activeCalls, messages }: ActivitySummaryProps) {
  // Build the summary text
  const summaryText = useMemo(() => {
    const lines: string[] = [];

    // Header
    lines.push(`=== ${system.name} ===`);
    lines.push(`Protocol: ${system.protocol === "p25_phase1" ? "P25 Phase I" : "P25 Phase II"}`);
    lines.push(`State: ${formatState(system.state)}`);
    lines.push(`Control Channel: ${formatControlState(system.controlChannelState)}`);
    lines.push("");

    // Network section
    lines.push("NETWORK");
    lines.push(`  NAC: ${formatHex(system.nac, 3)}`);
    lines.push(`  System ID: ${formatHex(system.systemId, 3)}`);
    lines.push("");

    // Site section
    lines.push("SITE");
    lines.push(`  RFSS: ${system.rfssId !== null ? system.rfssId : "(pending)"}`);
    lines.push(`  Site ID: ${system.siteId !== null ? system.siteId : "(pending)"}`);
    if (system.controlChannelFreqHz) {
      lines.push(`  Control Channel: ${formatFrequency(system.controlChannelFreqHz)} MHz`);
    }
    lines.push("");

    // Control channels
    lines.push(`CONTROL CHANNELS (${system.controlChannels.length} configured)`);
    for (const cc of system.controlChannels) {
      const marker = cc.isCurrent ? " [CURRENT]" : cc.isLocked ? " [LOCKED]" : "";
      const snr = cc.snrDb !== null ? `SNR: ${cc.snrDb.toFixed(1)} dB` : "SNR: ---";
      const sync = cc.syncDetected ? " SYNC" : "";
      const enabled = cc.enabled ? "" : " (disabled)";
      lines.push(`  ${formatFrequency(cc.frequencyHz)} MHz - ${snr}${sync}${marker}${enabled}`);
    }
    lines.push("");

    // Statistics
    lines.push("STATISTICS");
    lines.push(`  TSBKs: ${formatNumber(system.stats.tsbk_count)}`);
    lines.push(`  Grants: ${formatNumber(system.stats.grant_count)}`);
    lines.push(`  Total Calls: ${formatNumber(system.stats.calls_total)}`);
    lines.push(`  Decode Rate: ${system.decodeRate.toFixed(1)} fps`);
    lines.push(`  Recorders: ${system.stats.recorders_active} active / ${system.stats.recorders_idle} idle`);
    lines.push("");

    // Active calls
    lines.push(`ACTIVE CALLS (${activeCalls.length})`);
    if (activeCalls.length === 0) {
      lines.push("  (none)");
    } else {
      for (const call of activeCalls) {
        const enc = call.encrypted ? " [ENC]" : "";
        const source = call.sourceId ? ` RU:${call.sourceId}` : "";
        lines.push(`  TG ${call.talkgroupId} (${call.talkgroupName}) @ ${formatFrequency(call.frequencyHz)}${source}${enc}`);
      }
    }
    lines.push("");

    // Recent messages summary
    lines.push(`RECENT MESSAGES (${messages.length} buffered)`);
    const msgCounts: Record<string, number> = {};
    for (const msg of messages.slice(-100)) {
      const type = getMessageCategory(msg.opcodeName);
      msgCounts[type] = (msgCounts[type] || 0) + 1;
    }
    if (Object.keys(msgCounts).length === 0) {
      lines.push("  (none)");
    } else {
      for (const [type, count] of Object.entries(msgCounts).sort((a, b) => b[1] - a[1])) {
        lines.push(`  ${type}: ${count}`);
      }
    }

    return lines.join("\n");
  }, [system, activeCalls, messages]);

  // Copy to clipboard
  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(summaryText);
  }, [summaryText]);

  return (
    <div className="d-flex flex-column">
      {/* Toolbar */}
      <div className="d-flex align-items-center gap-2 mb-2" style={{ fontSize: "0.75rem" }}>
        <span className="text-muted">System activity summary</span>
        <div className="ms-auto d-flex gap-1">
          <button
            className="btn btn-sm btn-outline-secondary"
            onClick={handleCopy}
            title="Copy to clipboard"
            style={{ padding: "2px 6px" }}
          >
            <Copy size={12} />
          </button>
        </div>
      </div>

      {/* Summary text */}
      <pre
        className="bg-dark text-light rounded p-2 mb-0 overflow-auto font-monospace flex-grow-1"
        style={{ fontSize: "0.7rem", maxHeight: 400, whiteSpace: "pre-wrap" }}
      >
        {summaryText}
      </pre>
    </div>
  );
}

function formatState(state: string): string {
  switch (state) {
    case "stopped": return "Stopped";
    case "starting": return "Starting";
    case "searching": return "Searching for control channel";
    case "syncing": return "Syncing";
    case "running": return "Running";
    case "failed": return "Failed";
    default: return state;
  }
}

function formatControlState(state: string): string {
  switch (state) {
    case "unlocked": return "Unlocked (searching)";
    case "searching": return "Searching";
    case "locked": return "Locked (receiving)";
    case "lost": return "Lost (recovering)";
    default: return state;
  }
}

function getMessageCategory(opcodeName: string): string {
  if (!opcodeName) return "Unknown";
  if (opcodeName.includes("GRANT")) return "Grants";
  if (opcodeName.includes("STS") || opcodeName.includes("BCAST")) return "Status";
  if (opcodeName.includes("REG") || opcodeName.includes("AFF")) return "Registration";
  if (opcodeName.includes("IDEN")) return "Identifiers";
  return "Other";
}
