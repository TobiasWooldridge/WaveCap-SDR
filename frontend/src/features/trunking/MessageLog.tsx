import { useEffect, useRef, useState, useMemo, useCallback } from "react";
import { MessageSquare, Search, Filter, Copy, Download, X } from "lucide-react";

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

// Filter categories matching opcode patterns
type FilterCategory = "all" | "grants" | "status" | "registration" | "identifiers" | "other";

const FILTER_OPTIONS: { value: FilterCategory; label: string; match: (name: string) => boolean }[] = [
  { value: "all", label: "All", match: () => true },
  { value: "grants", label: "Grants", match: (n) => n.includes("GRANT") },
  { value: "status", label: "Status", match: (n) => n.includes("STS") || n.includes("BCAST") },
  { value: "registration", label: "Registration", match: (n) => n.includes("REG") || n.includes("AFF") },
  { value: "identifiers", label: "Identifiers", match: (n) => n.includes("IDEN") },
  { value: "other", label: "Other", match: (n) => {
    return !n.includes("GRANT") && !n.includes("STS") && !n.includes("BCAST") &&
           !n.includes("REG") && !n.includes("AFF") && !n.includes("IDEN");
  }},
];

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

function formatFullTimestamp(timestamp: number): string {
  const date = new Date(timestamp * 1000);
  return date.toISOString();
}

function getOpcodeColor(opcodeName: string | undefined): string {
  if (!opcodeName) return "text-body-secondary";
  if (opcodeName.includes("GRANT")) return "text-success";
  if (opcodeName.includes("IDEN")) return "text-info";
  if (opcodeName.includes("STS") || opcodeName.includes("BCAST")) return "text-warning";
  if (opcodeName.includes("REG") || opcodeName.includes("AFF")) return "text-primary";
  return "text-body-secondary";
}

function getOpcodeBadgeClass(opcodeName: string | undefined): string {
  if (!opcodeName) return "bg-secondary";
  if (opcodeName.includes("GRANT")) return "bg-success";
  if (opcodeName.includes("IDEN")) return "bg-info";
  if (opcodeName.includes("STS") || opcodeName.includes("BCAST")) return "bg-warning text-dark";
  if (opcodeName.includes("REG") || opcodeName.includes("AFF")) return "bg-primary";
  return "bg-secondary";
}

export function MessageLog({ messages, maxHeight = 400 }: MessageLogProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const isAtBottomRef = useRef(true);

  // Filter and search state
  const [filter, setFilter] = useState<FilterCategory>("all");
  const [search, setSearch] = useState("");
  const [showFilters, setShowFilters] = useState(false);

  // Filter messages
  const filteredMessages = useMemo(() => {
    const filterOption = FILTER_OPTIONS.find(f => f.value === filter);
    return messages.filter(msg => {
      // Apply category filter
      if (filter !== "all" && filterOption) {
        if (!filterOption.match(msg.opcodeName || "")) return false;
      }
      // Apply text search
      if (search) {
        const searchLower = search.toLowerCase();
        const matchesOpcode = (msg.opcodeName || "").toLowerCase().includes(searchLower);
        const matchesSummary = (msg.summary || "").toLowerCase().includes(searchLower);
        if (!matchesOpcode && !matchesSummary) return false;
      }
      return true;
    });
  }, [messages, filter, search]);

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
  }, [filteredMessages.length]);

  // Copy to clipboard
  const handleCopy = useCallback(() => {
    const text = filteredMessages
      .map(msg => `${formatFullTimestamp(msg.timestamp)}\t${msg.opcodeName || ""}\t${msg.summary || ""}`)
      .join("\n");
    navigator.clipboard.writeText(text);
  }, [filteredMessages]);

  // Export to CSV
  const handleExport = useCallback(() => {
    const headers = "timestamp,opcode,opcodeName,nac,summary";
    const rows = filteredMessages.map(msg => {
      const ts = formatFullTimestamp(msg.timestamp);
      const opcode = msg.opcode?.toString(16).toUpperCase() || "";
      const opcodeName = msg.opcodeName || "";
      const nac = msg.nac !== null ? msg.nac.toString(16).toUpperCase() : "";
      const summary = `"${(msg.summary || "").replace(/"/g, '""')}"`;
      return `${ts},0x${opcode},${opcodeName},${nac},${summary}`;
    });
    const csv = [headers, ...rows].join("\n");

    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `p25-messages-${new Date().toISOString().slice(0, 19).replace(/[:-]/g, "")}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, [filteredMessages]);

  // Clear filters
  const clearFilters = () => {
    setFilter("all");
    setSearch("");
  };

  const hasActiveFilters = filter !== "all" || search !== "";

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
    <div className="d-flex flex-column" style={{ maxHeight: maxHeight + 40 }}>
      {/* Toolbar */}
      <div className="d-flex align-items-center gap-2 mb-2 flex-wrap" style={{ fontSize: "0.75rem" }}>
        {/* Filter toggle */}
        <button
          className={`btn btn-sm ${showFilters ? "btn-primary" : "btn-outline-secondary"}`}
          onClick={() => setShowFilters(!showFilters)}
          title="Toggle filters"
          style={{ padding: "2px 6px" }}
        >
          <Filter size={12} />
        </button>

        {/* Search input */}
        <div className="input-group input-group-sm" style={{ maxWidth: "200px" }}>
          <span className="input-group-text" style={{ padding: "2px 6px" }}>
            <Search size={12} />
          </span>
          <input
            type="text"
            className="form-control form-control-sm"
            placeholder="Search..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            style={{ fontSize: "0.7rem", padding: "2px 6px" }}
          />
          {search && (
            <button
              className="btn btn-outline-secondary btn-sm"
              onClick={() => setSearch("")}
              style={{ padding: "2px 6px" }}
            >
              <X size={12} />
            </button>
          )}
        </div>

        {/* Filter status */}
        {hasActiveFilters && (
          <span className="badge bg-info text-dark">
            {filteredMessages.length} / {messages.length}
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
        <div className="d-flex gap-1 mb-2 flex-wrap">
          {FILTER_OPTIONS.map(opt => (
            <button
              key={opt.value}
              className={`btn btn-sm ${filter === opt.value ? "btn-primary" : "btn-outline-secondary"}`}
              onClick={() => setFilter(opt.value)}
              style={{ fontSize: "0.65rem", padding: "1px 6px" }}
            >
              {opt.label}
            </button>
          ))}
        </div>
      )}

      {/* Message list */}
      <div
        ref={scrollRef}
        className="bg-body-tertiary rounded overflow-auto font-monospace flex-grow-1"
        style={{ maxHeight, fontSize: "0.7rem" }}
        onScroll={handleScroll}
      >
        {filteredMessages.length === 0 ? (
          <div className="text-center text-body-secondary py-3">
            No messages match the current filter
          </div>
        ) : (
          <div className="d-flex flex-column">
            {filteredMessages.map((msg, idx) => (
              <div
                key={`${msg.timestamp}-${idx}`}
                className="d-flex align-items-start gap-2 px-2 py-1 border-bottom"
              >
                {/* Timestamp with milliseconds */}
                <span className="text-body-secondary" style={{ minWidth: "85px" }}>
                  {formatTime(msg.timestamp)}
                  <span className="opacity-75">.{formatMillis(msg.timestamp)}</span>
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
                  {msg.opcodeName || "UNKNOWN"}
                </span>

                {/* Summary - wraps if needed */}
                <span className={`flex-grow-1 ${getOpcodeColor(msg.opcodeName)}`}>
                  {msg.summary}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
