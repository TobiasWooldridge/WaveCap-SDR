import { useState, useEffect } from "react";
import {
  Phone,
  PhoneOff,
  Lock,
  Volume2,
  Link,
  Copy,
  CheckCircle,
  Info,
} from "lucide-react";
import type { ActiveCall } from "../../types/trunking";
import { TRUNKING_VOICE_STREAM_FORMATS } from "../../components/StreamLinks";
import { FrequencyDisplay } from "../../components/primitives/FrequencyDisplay.react";

interface ActiveCallsTableProps {
  calls: ActiveCall[];
  systemId: string;
  onPlayAudio?: (callId: string, streamId: string) => void;
  playingCallId?: string | null;
  onCopyUrl?: (url: string) => void;
}

function formatDuration(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

function formatLocation(location: ActiveCall["sourceLocation"]): string {
  if (!location) return "No location";
  const lat = location.latitude.toFixed(6);
  const lon = location.longitude.toFixed(6);
  const parts = [`${lat}, ${lon}`];
  if (location.altitude !== null && location.altitude !== undefined) {
    parts.push(`alt ${location.altitude.toFixed(1)}m`);
  }
  if (location.speed !== null && location.speed !== undefined) {
    parts.push(`spd ${location.speed.toFixed(1)} km/h`);
  }
  if (location.heading !== null && location.heading !== undefined) {
    parts.push(`hdg ${location.heading.toFixed(0)}°`);
  }
  if (location.accuracy !== null && location.accuracy !== undefined) {
    parts.push(`±${location.accuracy.toFixed(1)}m`);
  }
  const age = location.ageSeconds.toFixed(1);
  parts.push(`${age}s ago`);
  return parts.join(" • ");
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

/**
 * Check if a call is actively transmitting (received audio recently).
 * Returns true if audio was received within the threshold.
 */
function isActivelyTransmitting(
  call: ActiveCall,
  thresholdSeconds = 2,
): boolean {
  if (call.state !== "recording") return false;
  const now = Date.now() / 1000;
  return now - call.lastActivityTime < thresholdSeconds;
}

/**
 * Audio activity bars component - animated VU meter style indicator
 */
function AudioBars() {
  return (
    <span className="audio-bars" title="Receiving audio">
      <span className="audio-bar" />
      <span className="audio-bar" />
      <span className="audio-bar" />
    </span>
  );
}

/**
 * Transmission indicator component - shows TX badge and audio bars
 */
function TransmissionIndicator({ isActive }: { isActive: boolean }) {
  if (!isActive) return null;
  return (
    <span className="tx-indicator">
      <span className="tx-badge">TX</span>
      <AudioBars />
    </span>
  );
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

function getCallStateClass(state: string, isTransmitting: boolean): string {
  if (isTransmitting) {
    return "call-row-active";
  }
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
    const hasActiveCalls = calls.some((c) => c.state !== "ended");
    if (!hasActiveCalls) return;

    const interval = setInterval(() => {
      setTick((t) => t + 1);
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
    const format = TRUNKING_VOICE_STREAM_FORMATS.find(
      (f) => f.key === formatKey,
    );
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
            <th style={{ minWidth: "180px" }}>Talkgroup</th>
            <th>Category</th>
            <th style={{ width: "70px" }}>Source</th>
            <th className="text-end" style={{ width: "90px" }}>
              Frequency
            </th>
            <th className="text-end" style={{ width: "70px" }}>
              Duration
            </th>
            <th style={{ width: "80px" }}></th>
          </tr>
        </thead>
        <tbody>
          {sortedCalls.map((call) => {
            const transmitting = isActivelyTransmitting(call);
            return (
              <>
                <tr
                  key={call.id}
                  className={getCallStateClass(call.state, transmitting)}
                >
                  <td className="text-center">
                    <div className="d-flex flex-column align-items-center gap-1">
                      {getCallStateIcon(call.state, call.encrypted)}
                      <TransmissionIndicator isActive={transmitting} />
                    </div>
                  </td>
                  <td style={{ wordBreak: "break-word" }}>
                    <div className="fw-semibold" title={call.talkgroupName}>
                      {call.talkgroupName}
                    </div>
                    <div className="d-flex align-items-center gap-2">
                      <small className="text-muted">
                        TG {call.talkgroupId}
                      </small>
                      {call.encrypted && (
                        <span
                          className="badge bg-danger"
                          style={{ fontSize: "0.65rem" }}
                        >
                          Encrypted
                        </span>
                      )}
                    </div>
                  </td>
                  <td>
                    {call.talkgroupCategory && (
                      <span className="badge bg-secondary">
                        {call.talkgroupCategory}
                      </span>
                    )}
                  </td>
                  <td>
                    {call.sourceId !== null ? (
                      <span className="badge bg-secondary">
                        {call.sourceId}
                      </span>
                    ) : (
                      <span className="text-muted">---</span>
                    )}
                  </td>
                  <td className="text-end font-monospace">
                    <FrequencyDisplay
                      frequencyHz={call.frequencyHz}
                      decimals={4}
                      unit="MHz"
                    />
                  </td>
                  <td className="text-end font-monospace">
                    {formatDuration(getCallDuration(call))}
                  </td>
                  <td className="text-center">
                    <div className="btn-group btn-group-sm">
                      <button
                        className={`btn ${
                          expandedCallId === call.id
                            ? "btn-secondary"
                            : "btn-outline-secondary"
                        }`}
                        onClick={() => toggleExpanded(call.id)}
                        title="Call metadata"
                      >
                        <Info size={14} />
                      </button>
                      {onPlayAudio &&
                        !call.encrypted &&
                        call.state === "recording" &&
                        call.recorderId && (
                          <button
                            className={`btn ${
                              playingCallId === call.id
                                ? "btn-warning"
                                : "btn-outline-success"
                            }`}
                            onClick={() =>
                              onPlayAudio(call.id, call.recorderId!)
                            }
                            title={
                              playingCallId === call.id
                                ? "Stop playing"
                                : "Play audio"
                            }
                          >
                            <Volume2 size={14} />
                          </button>
                        )}
                      {onCopyUrl &&
                        call.state === "recording" &&
                        call.recorderId && (
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
                {expandedCallId === call.id && (
                  <tr key={`${call.id}-urls`} className="bg-body-secondary">
                    <td colSpan={7}>
                      <div className="d-flex flex-column gap-2 py-2 px-2">
                        <div>
                          <small className="text-muted me-2">Metadata:</small>
                          <div className="d-flex flex-wrap gap-2 mt-1">
                            <span className="badge bg-dark">
                              Channel: 0x
                              {call.channelId.toString(16).toUpperCase()}
                            </span>
                            {call.talkgroupAlphaTag && (
                              <span className="badge bg-secondary">
                                Alpha: {call.talkgroupAlphaTag}
                              </span>
                            )}
                            {call.talkgroupPriority !== null &&
                              call.talkgroupPriority !== undefined && (
                                <span className="badge bg-secondary">
                                  Priority: {call.talkgroupPriority}
                                </span>
                              )}
                            {call.talkgroupRecord !== null &&
                              call.talkgroupRecord !== undefined && (
                                <span className="badge bg-secondary">
                                  Record: {call.talkgroupRecord ? "yes" : "no"}
                                </span>
                              )}
                            {call.talkgroupMonitor !== null &&
                              call.talkgroupMonitor !== undefined && (
                                <span className="badge bg-secondary">
                                  Monitor:{" "}
                                  {call.talkgroupMonitor ? "yes" : "no"}
                                </span>
                              )}
                            <span className="badge bg-secondary">
                              Last activity:{" "}
                              {Math.max(
                                0,
                                Date.now() / 1000 - call.lastActivityTime,
                              ).toFixed(1)}
                              s
                            </span>
                          </div>
                          <div
                            className="text-muted mt-2"
                            style={{ fontSize: "0.75rem" }}
                          >
                            Location: {formatLocation(call.sourceLocation)}
                          </div>
                          <pre
                            className="bg-body-tertiary rounded p-2 mt-2 mb-0"
                            style={{
                              fontSize: "0.65rem",
                              whiteSpace: "pre-wrap",
                            }}
                          >
                            {JSON.stringify(call, null, 2)}
                          </pre>
                        </div>
                        {onCopyUrl &&
                          call.state === "recording" &&
                          call.recorderId && (
                            <div className="d-flex gap-2">
                              <small className="text-muted me-2">
                                Stream URLs:
                              </small>
                              {TRUNKING_VOICE_STREAM_FORMATS.map((format) => {
                                const isCopied =
                                  copiedFormat === `${call.id}-${format.key}`;
                                return (
                                  <button
                                    key={format.key}
                                    className="btn btn-sm btn-outline-secondary d-flex align-items-center gap-1"
                                    onClick={() =>
                                      handleCopyStreamUrl(call, format.key)
                                    }
                                  >
                                    {isCopied ? (
                                      <CheckCircle
                                        size={12}
                                        className="text-success"
                                      />
                                    ) : (
                                      <Copy size={12} />
                                    )}
                                    <span style={{ fontSize: "11px" }}>
                                      {format.label}
                                    </span>
                                  </button>
                                );
                              })}
                            </div>
                          )}
                      </div>
                    </td>
                  </tr>
                )}
              </>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
