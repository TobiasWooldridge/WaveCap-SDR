import { useState, useEffect } from "react";
import { Phone, PhoneOff, Lock, Volume2, Link, Copy, CheckCircle } from "lucide-react";
import type { ActiveCall } from "../../types/trunking";
import { TRUNKING_VOICE_STREAM_FORMATS } from "../../components/StreamLinks";

interface ActiveCallsTableProps {
  calls: ActiveCall[];
  systemId: string;
  onPlayAudio?: (callId: string, streamId: string) => void;
  playingCallId?: string | null;
  onCopyUrl?: (url: string) => void;
}

function formatFrequency(hz: number): string {
  return (hz / 1_000_000).toFixed(4);
}

function formatDuration(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

/**
 * Calculate call duration dynamically.
 * For active calls (not ended), compute from startTime to now.
 * For ended calls, use the server-provided durationSeconds.
 */
function getCallDuration(call: ActiveCall): number {
  if (call.state === "ended") {
    // Use server-provided duration for ended calls
    return call.durationSeconds;
  }
  // Calculate live duration for active calls
  const now = Date.now() / 1000; // Current time in seconds
  return Math.max(0, now - call.startTime);
}

function getCallStateIcon(state: string, encrypted: boolean) {
  if (encrypted) {
    return <Lock size={14} className="text-danger" />;
  }
  switch (state) {
    case "recording":
      return <Phone size={14} className="text-success animate-pulse" />;
    case "hold":
      return <Phone size={14} className="text-warning" />;
    case "ended":
      return <PhoneOff size={14} className="text-muted" />;
    default:
      return <Phone size={14} className="text-info" />;
  }
}

function getCallStateClass(state: string): string {
  switch (state) {
    case "recording":
      return "table-success";
    case "hold":
      return "table-warning";
    default:
      return "";
  }
}

export function ActiveCallsTable({
  calls,
  systemId,
  onPlayAudio,
  playingCallId,
  onCopyUrl,
}: ActiveCallsTableProps) {
  const [expandedCallId, setExpandedCallId] = useState<string | null>(null);
  const [copiedFormat, setCopiedFormat] = useState<string | null>(null);
  const [, setTick] = useState(0);

  // Refresh every second to update live durations for active calls
  useEffect(() => {
    const hasActiveCalls = calls.some(c => c.state !== "ended");
    if (!hasActiveCalls) return;

    const interval = setInterval(() => {
      setTick(t => t + 1);
    }, 1000);

    return () => clearInterval(interval);
  }, [calls]);

  if (calls.length === 0) {
    return (
      <div className="text-center text-muted py-4">
        <Phone size={32} className="mb-2 opacity-50" />
        <p className="small mb-0">No active calls</p>
      </div>
    );
  }

  // Sort by start time (most recent first) and state (recording first)
  const sortedCalls = [...calls].sort((a, b) => {
    // Recording calls first
    if (a.state === "recording" && b.state !== "recording") return -1;
    if (a.state !== "recording" && b.state === "recording") return 1;
    // Then by start time
    return b.startTime - a.startTime;
  });

  const handleCopyStreamUrl = (call: ActiveCall, formatKey: string) => {
    if (!call.recorderId) return;
    const baseUrl = `/api/v1/trunking/stream/${systemId}/voice/${call.recorderId}`;
    const format = TRUNKING_VOICE_STREAM_FORMATS.find(f => f.key === formatKey);
    if (format && onCopyUrl) {
      const url = format.buildUrl(baseUrl);
      onCopyUrl(url);
      setCopiedFormat(`${call.id}-${formatKey}`);
      setTimeout(() => setCopiedFormat(null), 2000);
    }
  };

  const toggleExpanded = (callId: string) => {
    setExpandedCallId(expandedCallId === callId ? null : callId);
  };

  return (
    <div className="table-responsive">
      <table className="table table-sm table-hover mb-0">
        <thead>
          <tr>
            <th style={{ width: "30px" }}></th>
            <th>Talkgroup</th>
            <th>Source</th>
            <th className="text-end">Frequency</th>
            <th className="text-end">Duration</th>
            <th style={{ width: "80px" }}></th>
          </tr>
        </thead>
        <tbody>
          {sortedCalls.map((call) => (
            <>
              <tr key={call.id} className={getCallStateClass(call.state)}>
                <td className="text-center">
                  {getCallStateIcon(call.state, call.encrypted)}
                </td>
                <td>
                  <div className="fw-semibold">{call.talkgroupName}</div>
                  <small className="text-muted">TG {call.talkgroupId}</small>
                </td>
                <td>
                  {call.sourceId !== null ? (
                    <span className="badge bg-secondary">{call.sourceId}</span>
                  ) : (
                    <span className="text-muted">---</span>
                  )}
                </td>
                <td className="text-end font-monospace">
                  {formatFrequency(call.frequencyHz)}
                </td>
                <td className="text-end font-monospace">
                  {formatDuration(getCallDuration(call))}
                </td>
                <td className="text-center">
                  <div className="btn-group btn-group-sm">
                    {onPlayAudio && !call.encrypted && call.state === "recording" && call.recorderId && (
                      <button
                        className={`btn ${
                          playingCallId === call.id
                            ? "btn-warning"
                            : "btn-outline-success"
                        }`}
                        onClick={() => onPlayAudio(call.id, call.recorderId!)}
                        title={
                          playingCallId === call.id
                            ? "Stop playing"
                            : "Play audio"
                        }
                      >
                        <Volume2 size={14} />
                      </button>
                    )}
                    {onCopyUrl && call.state === "recording" && call.recorderId && (
                      <button
                        className={`btn ${expandedCallId === call.id ? "btn-secondary" : "btn-outline-secondary"}`}
                        onClick={() => toggleExpanded(call.id)}
                        title="Stream URLs"
                      >
                        <Link size={14} />
                      </button>
                    )}
                  </div>
                </td>
              </tr>
              {expandedCallId === call.id && onCopyUrl && (
                <tr key={`${call.id}-urls`} className="bg-body-secondary">
                  <td colSpan={6}>
                    <div className="d-flex gap-2 py-1 px-2">
                      <small className="text-muted me-2">Stream URLs:</small>
                      {TRUNKING_VOICE_STREAM_FORMATS.map((format) => {
                        const isCopied = copiedFormat === `${call.id}-${format.key}`;
                        return (
                          <button
                            key={format.key}
                            className="btn btn-sm btn-outline-secondary d-flex align-items-center gap-1"
                            onClick={() => handleCopyStreamUrl(call, format.key)}
                          >
                            {isCopied ? (
                              <CheckCircle size={12} className="text-success" />
                            ) : (
                              <Copy size={12} />
                            )}
                            <span style={{ fontSize: "11px" }}>{format.label}</span>
                          </button>
                        );
                      })}
                    </div>
                  </td>
                </tr>
              )}
            </>
          ))}
        </tbody>
      </table>
    </div>
  );
}
