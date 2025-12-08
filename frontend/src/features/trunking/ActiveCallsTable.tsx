import { Phone, PhoneOff, Lock, Volume2 } from "lucide-react";
import type { ActiveCall } from "../../types/trunking";

interface ActiveCallsTableProps {
  calls: ActiveCall[];
  onPlayAudio?: (callId: string) => void;
  playingCallId?: string | null;
}

function formatFrequency(hz: number): string {
  return (hz / 1_000_000).toFixed(4);
}

function formatDuration(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
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
  onPlayAudio,
  playingCallId,
}: ActiveCallsTableProps) {
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
            {onPlayAudio && <th style={{ width: "40px" }}></th>}
          </tr>
        </thead>
        <tbody>
          {sortedCalls.map((call) => (
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
                {formatDuration(call.durationSeconds)}
              </td>
              {onPlayAudio && (
                <td className="text-center">
                  {!call.encrypted && call.state === "recording" && (
                    <button
                      className={`btn btn-sm ${
                        playingCallId === call.id
                          ? "btn-danger"
                          : "btn-outline-primary"
                      }`}
                      onClick={() => onPlayAudio(call.id)}
                      title={
                        playingCallId === call.id
                          ? "Stop playing"
                          : "Play audio"
                      }
                    >
                      <Volume2 size={14} />
                    </button>
                  )}
                </td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
