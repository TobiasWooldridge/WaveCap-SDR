import { useEffect, useRef } from "react";
import { Phone, PhoneOff, PhoneIncoming } from "lucide-react";

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

function formatTime(timestamp: number): string {
  const date = new Date(timestamp * 1000);
  return date.toLocaleTimeString("en-US", {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function formatFrequency(hz: number): string {
  return (hz / 1_000_000).toFixed(4);
}

function formatDuration(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  if (mins > 0) {
    return `${mins}m ${secs}s`;
  }
  return `${secs}s`;
}

function getEventIcon(type: string, encrypted?: boolean) {
  if (encrypted) {
    return <Phone size={12} className="text-danger" />;
  }
  switch (type) {
    case "start":
      return <PhoneIncoming size={12} className="text-success" />;
    case "end":
      return <PhoneOff size={12} className="text-muted" />;
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
  }, [events.length]);

  if (events.length === 0) {
    return (
      <div
        className="text-center text-muted py-3 bg-body-secondary rounded"
        style={{ fontSize: "0.8rem" }}
      >
        <Phone size={20} className="mb-1 opacity-50" />
        <div>Waiting for call activity...</div>
      </div>
    );
  }

  return (
    <div
      ref={scrollRef}
      className="bg-body-secondary rounded overflow-auto"
      style={{ maxHeight, fontSize: "0.75rem" }}
      onScroll={handleScroll}
    >
      <div className="d-flex flex-column">
        {events.map((event) => (
          <div
            key={`${event.id}-${event.type}-${event.timestamp}`}
            className={`d-flex align-items-center gap-2 px-2 py-1 ${getEventClass(event.type)}`}
            style={{ borderBottom: "1px solid var(--bs-border-color)" }}
          >
            {/* Time */}
            <span className="text-muted font-monospace" style={{ width: "60px" }}>
              {formatTime(event.timestamp)}
            </span>

            {/* Icon */}
            <span style={{ width: "16px" }}>
              {getEventIcon(event.type, event.encrypted)}
            </span>

            {/* Event type */}
            <span
              className={`badge ${
                event.type === "start"
                  ? "bg-success"
                  : event.type === "end"
                  ? "bg-secondary"
                  : "bg-info"
              }`}
              style={{ width: "40px", fontSize: "0.65rem" }}
            >
              {event.type.toUpperCase()}
            </span>

            {/* Talkgroup */}
            <span className="fw-semibold text-truncate" style={{ minWidth: "80px", maxWidth: "150px" }}>
              {event.talkgroupName}
            </span>

            {/* TG ID */}
            <span className="text-muted" style={{ width: "50px" }}>
              TG {event.talkgroupId}
            </span>

            {/* Source */}
            <span style={{ width: "50px" }}>
              {event.sourceId !== null ? (
                <span className="badge bg-secondary" style={{ fontSize: "0.65rem" }}>
                  {event.sourceId}
                </span>
              ) : (
                <span className="text-muted">---</span>
              )}
            </span>

            {/* Frequency */}
            <span className="font-monospace text-muted" style={{ width: "70px" }}>
              {formatFrequency(event.frequencyHz)}
            </span>

            {/* Duration (for end events) */}
            <span style={{ width: "45px" }}>
              {event.type === "end" && event.durationSeconds !== undefined && (
                <span className="text-info">{formatDuration(event.durationSeconds)}</span>
              )}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
