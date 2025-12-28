import { useEffect, useRef } from "react";
import { MessageSquare } from "lucide-react";

export interface P25Message {
  timestamp: number;
  opcode: number;
  opcodeName: string;
  nac: number | null;
  summary: string;
}

interface MessageLogProps {
  messages: P25Message[];
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

function formatMillis(timestamp: number): string {
  const millis = Math.floor((timestamp % 1) * 1000);
  return millis.toString().padStart(3, "0");
}

function getOpcodeColor(opcodeName: string | undefined): string {
  if (!opcodeName) return "text-secondary";
  // Voice grants - green
  if (opcodeName.includes("GRANT")) {
    return "text-success";
  }
  // Channel identifiers - blue
  if (opcodeName.includes("IDEN")) {
    return "text-info";
  }
  // Status broadcasts - yellow
  if (opcodeName.includes("STS") || opcodeName.includes("BCAST")) {
    return "text-warning";
  }
  // Registration/affiliation - purple
  if (opcodeName.includes("REG") || opcodeName.includes("AFF")) {
    return "text-primary";
  }
  // Default - gray
  return "text-secondary";
}

function getOpcodeBadgeClass(opcodeName: string | undefined): string {
  if (!opcodeName) return "bg-secondary";
  if (opcodeName.includes("GRANT")) {
    return "bg-success";
  }
  if (opcodeName.includes("IDEN")) {
    return "bg-info";
  }
  if (opcodeName.includes("STS") || opcodeName.includes("BCAST")) {
    return "bg-warning text-dark";
  }
  if (opcodeName.includes("REG") || opcodeName.includes("AFF")) {
    return "bg-primary";
  }
  return "bg-secondary";
}

export function MessageLog({ messages, maxHeight = 400 }: MessageLogProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const isAtBottomRef = useRef(true);

  // Track if user is at bottom
  const handleScroll = () => {
    if (scrollRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = scrollRef.current;
      isAtBottomRef.current = scrollHeight - scrollTop - clientHeight < 10;
    }
  };

  // Auto-scroll to bottom when new messages arrive (if already at bottom)
  useEffect(() => {
    if (scrollRef.current && isAtBottomRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages.length]);

  if (messages.length === 0) {
    return (
      <div
        className="text-center text-muted py-4 bg-body-secondary rounded"
        style={{ fontSize: "0.8rem" }}
      >
        <MessageSquare size={24} className="mb-2 opacity-50" />
        <div>Waiting for decoded messages...</div>
        <div className="small mt-1">Messages will appear when the control channel is locked</div>
      </div>
    );
  }

  return (
    <div
      ref={scrollRef}
      className="bg-dark text-light rounded overflow-auto font-monospace"
      style={{ maxHeight, fontSize: "0.7rem" }}
      onScroll={handleScroll}
    >
      <div className="d-flex flex-column">
        {messages.map((msg, idx) => (
          <div
            key={`${msg.timestamp}-${idx}`}
            className="d-flex align-items-start gap-2 px-2 py-1 border-bottom border-secondary"
          >
            {/* Timestamp with milliseconds */}
            <span className="text-muted" style={{ minWidth: "85px" }}>
              {formatTime(msg.timestamp)}
              <span className="text-secondary">.{formatMillis(msg.timestamp)}</span>
            </span>

            {/* NAC if available */}
            {msg.nac !== null && (
              <span className="text-info" style={{ minWidth: "45px" }}>
                NAC:{msg.nac.toString(16).toUpperCase().padStart(3, "0")}
              </span>
            )}

            {/* Opcode badge */}
            <span
              className={`badge ${getOpcodeBadgeClass(msg.opcodeName)}`}
              style={{ fontSize: "0.6rem", minWidth: "100px" }}
            >
              {msg.opcodeName}
            </span>

            {/* Summary - wraps if needed */}
            <span className={`flex-grow-1 ${getOpcodeColor(msg.opcodeName)}`}>
              {msg.summary}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
