import { useEffect, useRef, useState, useMemo, useCallback } from "react";
import {
  Phone,
  PhoneOff,
  PhoneIncoming,
  Lock,
  Filter,
  Copy,
  Download,
  X,
} from "lucide-react";

export interface CallEvent {
  id: string;
  timestamp: number;
  type: "start" | "end" | "update";
  talkgroupId: number;
  talkgroupName: string;
  sourceId: number | null;
  frequencyHz: number;
  durationSeconds?: number;
  encrypted?: boolean;
}

interface CallEventLogProps {
  events: CallEvent[];
  maxHeight?: number;
}

// Filter types
type EventTypeFilter = "all" | "start" | "end";
type EncryptedFilter = "all" | "encrypted" | "unencrypted";

function formatTime(timestamp: number): string {
  const date = new Date(timestamp * 1000);
  return date.toLocaleTimeString("en-US", {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function formatMillis(timestamp: number): string {
  const millis = Math.floor((timestamp % 1) * 1000);
  return millis.toString().padStart(3, "0");
}

function formatFrequency(hz: number): string {
  return (hz / 1_000_000).toFixed(4);
}

function formatDuration(seconds: number | undefined): string {
  if (seconds === undefined || seconds <= 0) return "";

  if (seconds < 60) {
    // Show with decimal for sub-minute durations
    return `${seconds.toFixed(1)}s`;
  }

  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}m ${secs}s`;
}

function formatFullTimestamp(timestamp: number): string {
  const date = new Date(timestamp * 1000);
  return date.toISOString();
}

function getEventIcon(type: string, encrypted?: boolean) {
  if (encrypted) {
    return <Lock size={12} className="text-danger" />;
  }
  switch (type) {
    case "start":
      return <PhoneIncoming size={12} className="text-success" />;
    case "end":
      return <PhoneOff size={12} className="text-body-secondary" />;
    default:
      return <Phone size={12} className="text-info" />;
  }
}

function getEventClass(type: string): string {
  switch (type) {
    case "start":
      return "border-start border-success border-2";
    case "end":
      return "border-start border-secondary border-2";
    default:
      return "";
  }
}

export function CallEventLog({ events, maxHeight = 300 }: CallEventLogProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const isAtBottomRef = useRef(true);

  // Filter state
  const [eventTypeFilter, setEventTypeFilter] =
    useState<EventTypeFilter>("all");
  const [encryptedFilter, setEncryptedFilter] =
    useState<EncryptedFilter>("all");
  const [showFilters, setShowFilters] = useState(false);

  // Filter events
  const filteredEvents = useMemo(() => {
    return events.filter((event) => {
      // Event type filter
      if (eventTypeFilter !== "all" && event.type !== eventTypeFilter) {
        return false;
      }
      // Encrypted filter
      if (encryptedFilter === "encrypted" && !event.encrypted) {
        return false;
      }
      if (encryptedFilter === "unencrypted" && event.encrypted) {
        return false;
      }
      return true;
    });
  }, [events, eventTypeFilter, encryptedFilter]);

  // Track if user is at bottom
  const handleScroll = () => {
    if (scrollRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = scrollRef.current;
      isAtBottomRef.current = scrollHeight - scrollTop - clientHeight < 10;
    }
  };

  // Auto-scroll to bottom when new events arrive (if already at bottom)
  useEffect(() => {
    if (scrollRef.current && isAtBottomRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [filteredEvents.length]);

  // Copy to clipboard
  const handleCopy = useCallback(() => {
    const text = filteredEvents
      .map((event) => {
        const duration =
          event.type === "end" ? formatDuration(event.durationSeconds) : "";
        return `${formatFullTimestamp(event.timestamp)}\t${event.type.toUpperCase()}\t${event.talkgroupName}\t${event.talkgroupId}\t${event.sourceId ?? ""}\t${formatFrequency(event.frequencyHz)}\t${duration}\t${event.encrypted ? "ENC" : ""}`;
      })
      .join("\n");
    navigator.clipboard.writeText(text);
  }, [filteredEvents]);

  // Export to CSV
  const handleExport = useCallback(() => {
    const headers =
      "timestamp,type,talkgroupName,talkgroupId,sourceId,frequencyMHz,durationSeconds,encrypted";
    const rows = filteredEvents.map((event) => {
      const ts = formatFullTimestamp(event.timestamp);
      const freq = formatFrequency(event.frequencyHz);
      const duration = event.durationSeconds?.toFixed(1) ?? "";
      const enc = event.encrypted ? "true" : "false";
      const name = `"${event.talkgroupName.replace(/"/g, '""')}"`;
      return `${ts},${event.type},${name},${event.talkgroupId},${event.sourceId ?? ""},${freq},${duration},${enc}`;
    });
    const csv = [headers, ...rows].join("\n");

    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `call-events-${new Date().toISOString().slice(0, 19).replace(/[:-]/g, "")}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, [filteredEvents]);

  // Clear filters
  const clearFilters = () => {
    setEventTypeFilter("all");
    setEncryptedFilter("all");
  };

  const hasActiveFilters =
    eventTypeFilter !== "all" || encryptedFilter !== "all";

  if (events.length === 0) {
    return (
      <div
        className="text-center text-muted py-3 bg-body-secondary rounded"
        style={{ fontSize: "0.8rem" }}
      >
        <Phone size={20} className="mb-1 opacity-50" />
        <div>Waiting for call activity...</div>
        <div className="small mt-1">
          Events will appear when calls start or end
        </div>
      </div>
    );
  }

  return (
    <div className="d-flex flex-column" style={{ maxHeight: maxHeight + 40 }}>
      {/* Toolbar */}
      <div
        className="d-flex align-items-center gap-2 mb-2 flex-wrap"
        style={{ fontSize: "0.75rem" }}
      >
        {/* Filter toggle */}
        <button
          className={`btn btn-sm ${showFilters ? "btn-primary" : "btn-outline-secondary"}`}
          onClick={() => setShowFilters(!showFilters)}
          title="Toggle filters"
          style={{ padding: "2px 6px" }}
        >
          <Filter size={12} />
        </button>

        {/* Filter status */}
        {hasActiveFilters && (
          <span className="badge bg-info text-dark">
            {filteredEvents.length} / {events.length}
            <button
              className="btn-close btn-close-white ms-1"
              style={{ fontSize: "0.5rem" }}
              onClick={clearFilters}
              title="Clear filters"
            />
          </span>
        )}

        <div className="ms-auto d-flex gap-1">
          {/* Copy button */}
          <button
            className="btn btn-sm btn-outline-secondary"
            onClick={handleCopy}
            title="Copy to clipboard"
            style={{ padding: "2px 6px" }}
          >
            <Copy size={12} />
          </button>

          {/* Export button */}
          <button
            className="btn btn-sm btn-outline-secondary"
            onClick={handleExport}
            title="Export CSV"
            style={{ padding: "2px 6px" }}
          >
            <Download size={12} />
          </button>
        </div>
      </div>

      {/* Filter chips */}
      {showFilters && (
        <div
          className="d-flex gap-2 mb-2 flex-wrap align-items-center"
          style={{ fontSize: "0.7rem" }}
        >
          {/* Event type filter */}
          <div className="d-flex gap-1 align-items-center">
            <span className="text-muted me-1">Type:</span>
            {(["all", "start", "end"] as EventTypeFilter[]).map((type) => (
              <button
                key={type}
                className={`btn btn-sm ${eventTypeFilter === type ? "btn-primary" : "btn-outline-secondary"}`}
                onClick={() => setEventTypeFilter(type)}
                style={{ fontSize: "0.65rem", padding: "1px 6px" }}
              >
                {type === "all" ? "All" : type === "start" ? "Starts" : "Ends"}
              </button>
            ))}
          </div>

          {/* Encrypted filter */}
          <div className="d-flex gap-1 align-items-center">
            <span className="text-muted me-1">Encrypted:</span>
            {(["all", "encrypted", "unencrypted"] as EncryptedFilter[]).map(
              (filter) => (
                <button
                  key={filter}
                  className={`btn btn-sm ${encryptedFilter === filter ? "btn-primary" : "btn-outline-secondary"}`}
                  onClick={() => setEncryptedFilter(filter)}
                  style={{ fontSize: "0.65rem", padding: "1px 6px" }}
                >
                  {filter === "all" ? (
                    "All"
                  ) : filter === "encrypted" ? (
                    <>
                      <Lock size={10} className="me-1" />
                      Yes
                    </>
                  ) : (
                    "No"
                  )}
                </button>
              ),
            )}
          </div>

          {/* Clear all */}
          {hasActiveFilters && (
            <button
              className="btn btn-sm btn-outline-danger"
              onClick={clearFilters}
              style={{ fontSize: "0.65rem", padding: "1px 6px" }}
            >
              <X size={10} className="me-1" />
              Clear
            </button>
          )}
        </div>
      )}

      {/* Event list */}
      <div
        ref={scrollRef}
        className="bg-body-tertiary rounded overflow-auto flex-grow-1"
        style={{ maxHeight, fontSize: "0.75rem" }}
        onScroll={handleScroll}
      >
        {filteredEvents.length === 0 ? (
          <div className="text-center text-body-secondary py-3">
            No events match the current filter
          </div>
        ) : (
          <div className="d-flex flex-column">
            {filteredEvents.map((event) => (
              <div
                key={`${event.id}-${event.type}-${event.timestamp}`}
                className={`d-flex align-items-center gap-2 px-2 py-1 ${getEventClass(event.type)}`}
                style={{ borderBottom: "1px solid var(--bs-border-color)" }}
              >
                {/* Time with milliseconds */}
                <span
                  className="text-body-secondary font-monospace"
                  style={{ width: "75px" }}
                >
                  {formatTime(event.timestamp)}
                  <span className="opacity-75">
                    .{formatMillis(event.timestamp)}
                  </span>
                </span>

                {/* Icon */}
                <span style={{ width: "16px" }}>
                  {getEventIcon(event.type, event.encrypted)}
                </span>

                {/* Event type badge */}
                <span
                  className={`badge ${
                    event.type === "start"
                      ? "bg-success"
                      : event.type === "end"
                        ? "bg-secondary"
                        : "bg-info"
                  }`}
                  style={{ width: "42px", fontSize: "0.6rem" }}
                >
                  {event.type === "start"
                    ? "START"
                    : event.type === "end"
                      ? "END"
                      : "UPDATE"}
                </span>

                {/* Talkgroup name + ID */}
                <span
                  className="d-flex align-items-center gap-1"
                  style={{ minWidth: "160px", flex: "1 1 auto" }}
                >
                  <span
                    className="fw-semibold"
                    style={{ wordBreak: "break-word" }}
                    title={event.talkgroupName}
                  >
                    {event.talkgroupName || `TG ${event.talkgroupId}`}
                  </span>
                  {event.talkgroupName && (
                    <span
                      className="badge bg-secondary"
                      style={{ fontSize: "0.55rem" }}
                    >
                      {event.talkgroupId}
                    </span>
                  )}
                </span>

                {/* Source ID */}
                <span style={{ width: "55px" }}>
                  {event.sourceId !== null ? (
                    <span
                      className="badge bg-secondary"
                      style={{ fontSize: "0.6rem" }}
                      title="Radio Unit ID"
                    >
                      RU {event.sourceId}
                    </span>
                  ) : (
                    <span className="text-body-tertiary">---</span>
                  )}
                </span>

                {/* Frequency */}
                <span
                  className="font-monospace text-body-secondary"
                  style={{ width: "65px" }}
                  title="Voice Channel Frequency"
                >
                  {formatFrequency(event.frequencyHz)}
                </span>

                {/* Duration (for end events) */}
                <span style={{ width: "50px" }}>
                  {event.type === "end" &&
                    event.durationSeconds !== undefined && (
                      <span className="text-info fw-semibold">
                        {formatDuration(event.durationSeconds)}
                      </span>
                    )}
                </span>

                {/* Encrypted indicator */}
                <span style={{ width: "20px" }}>
                  {event.encrypted && (
                    <span
                      className="badge bg-danger"
                      style={{ fontSize: "0.55rem" }}
                      title="Encrypted call"
                    >
                      <Lock size={8} />
                    </span>
                  )}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
