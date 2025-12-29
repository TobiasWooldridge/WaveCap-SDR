import { useState, useRef, useEffect, useCallback, useMemo } from "react";
import { Copy, Pause, Play, Search, Filter } from "lucide-react";
import type { AggregatedPOCSAGMessage } from "../../hooks/useDigitalMessages";
import { copyToClipboard } from "../../utils/clipboard";
import { useToast } from "../../hooks/useToast";

interface POCSAGMessageLogProps {
  messages: AggregatedPOCSAGMessage[];
  channels: { id: string; name: string | null; autoName: string | null }[];
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

function formatFullTimestamp(timestamp: number): string {
  const date = new Date(timestamp * 1000);
  return date.toLocaleString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

function getMessageTypeBadge(messageType: string): {
  label: string;
  className: string;
} {
  switch (messageType) {
    case "alpha":
      return { label: "TXT", className: "bg-primary" };
    case "numeric":
      return { label: "NUM", className: "bg-success" };
    case "alert_only":
      return { label: "ALT", className: "bg-warning text-dark" };
    case "alpha_2":
      return { label: "TX2", className: "bg-info" };
    default:
      return {
        label: messageType.toUpperCase().slice(0, 3),
        className: "bg-secondary",
      };
  }
}

export function POCSAGMessageLog({
  messages,
  channels,
  maxHeight = 400,
}: POCSAGMessageLogProps) {
  const [isPaused, setIsPaused] = useState(false);
  const [searchText, setSearchText] = useState("");
  const [addressFilter, setAddressFilter] = useState("");
  const [channelFilter, setChannelFilter] = useState<string>("all");
  const scrollRef = useRef<HTMLDivElement>(null);
  const toast = useToast();

  // Auto-scroll when not paused
  useEffect(() => {
    if (!isPaused && scrollRef.current) {
      scrollRef.current.scrollTop = 0;
    }
  }, [messages, isPaused]);

  // Pause on scroll
  const handleScroll = useCallback(() => {
    if (scrollRef.current && scrollRef.current.scrollTop > 50) {
      setIsPaused(true);
    }
  }, []);

  // Filter messages
  const filteredMessages = useMemo(() => {
    return messages.filter((msg) => {
      // Channel filter
      if (channelFilter !== "all" && msg.channelId !== channelFilter) {
        return false;
      }

      // Address filter
      if (addressFilter && !msg.address.toString().includes(addressFilter)) {
        return false;
      }

      // Text search
      if (searchText) {
        const searchLower = searchText.toLowerCase();
        const matchesMessage = msg.message.toLowerCase().includes(searchLower);
        const matchesAddress = msg.address.toString().includes(searchText);
        if (!matchesMessage && !matchesAddress) {
          return false;
        }
      }

      return true;
    });
  }, [messages, channelFilter, addressFilter, searchText]);

  const handleCopyMessage = async (msg: AggregatedPOCSAGMessage) => {
    const text = `[${formatFullTimestamp(msg.timestamp)}] ${msg.address}: ${msg.message}`;
    const success = await copyToClipboard(text);
    if (success) {
      toast.success("Message copied to clipboard");
    }
  };

  const handleAddressClick = (address: number) => {
    setAddressFilter(address.toString());
  };

  return (
    <div className="d-flex flex-column h-100" style={{ minHeight: 0 }}>
      {/* Controls */}
      <div className="d-flex gap-2 align-items-center mb-2 flex-wrap">
        {/* Channel filter */}
        <div className="d-flex align-items-center gap-1">
          <Filter size={12} className="text-body-secondary" />
          <select
            className="form-select form-select-sm"
            style={{ width: "auto", fontSize: "0.75rem" }}
            value={channelFilter}
            onChange={(e) => setChannelFilter(e.target.value)}
          >
            <option value="all">All Channels</option>
            {channels.map((ch) => (
              <option key={ch.id} value={ch.id}>
                {ch.name || ch.autoName || ch.id.slice(0, 8)}
              </option>
            ))}
          </select>
        </div>

        {/* Address filter */}
        <div className="d-flex align-items-center gap-1">
          <input
            type="text"
            className="form-control form-control-sm"
            placeholder="Address..."
            style={{ width: "100px", fontSize: "0.75rem" }}
            value={addressFilter}
            onChange={(e) => setAddressFilter(e.target.value)}
          />
          {addressFilter && (
            <button
              className="btn btn-sm btn-outline-secondary"
              onClick={() => setAddressFilter("")}
              title="Clear address filter"
              style={{ padding: "0.1rem 0.3rem", fontSize: "0.7rem" }}
            >
              &times;
            </button>
          )}
        </div>

        {/* Search */}
        <div className="d-flex align-items-center gap-1 flex-grow-1">
          <Search size={12} className="text-body-secondary" />
          <input
            type="text"
            className="form-control form-control-sm"
            placeholder="Search messages..."
            style={{ maxWidth: "200px", fontSize: "0.75rem" }}
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
          />
        </div>

        {/* Pause/Resume */}
        <button
          className="btn btn-sm btn-outline-secondary d-flex align-items-center gap-1"
          onClick={() => setIsPaused(!isPaused)}
          title={isPaused ? "Resume auto-scroll" : "Pause auto-scroll"}
        >
          {isPaused ? <Play size={12} /> : <Pause size={12} />}
        </button>

        {/* Stats */}
        <span
          className="badge bg-body-secondary text-body-secondary"
          style={{ fontSize: "0.7rem" }}
        >
          {filteredMessages.length} / {messages.length} msgs
        </span>
      </div>

      {/* Message list */}
      <div
        ref={scrollRef}
        className="bg-body-tertiary rounded overflow-auto font-monospace flex-grow-1"
        style={{ maxHeight, fontSize: "0.75rem" }}
        onScroll={handleScroll}
      >
        {filteredMessages.length === 0 ? (
          <div className="text-center text-body-secondary py-4">
            {messages.length === 0
              ? "No POCSAG messages received yet"
              : "No messages match the current filters"}
          </div>
        ) : (
          <table className="table table-sm table-hover mb-0">
            <thead className="sticky-top bg-body-tertiary">
              <tr>
                <th style={{ width: "70px" }}>Time</th>
                <th style={{ width: "100px" }}>Channel</th>
                <th style={{ width: "90px" }}>Address</th>
                <th style={{ width: "40px" }}>Type</th>
                <th>Message</th>
                <th style={{ width: "30px" }}></th>
              </tr>
            </thead>
            <tbody>
              {filteredMessages.map((msg, idx) => {
                const typeBadge = getMessageTypeBadge(msg.messageType);
                return (
                  <tr key={`${msg.timestamp}-${msg.address}-${idx}`}>
                    <td
                      className="text-body-secondary"
                      title={formatFullTimestamp(msg.timestamp)}
                    >
                      {formatTime(msg.timestamp)}
                    </td>
                    <td className="text-truncate" style={{ maxWidth: "100px" }}>
                      <span
                        className="badge bg-body-secondary text-body-secondary"
                        style={{ fontSize: "0.65rem" }}
                      >
                        {msg.channelName}
                      </span>
                    </td>
                    <td>
                      <button
                        className="btn btn-link btn-sm p-0 text-decoration-none text-start"
                        style={{ fontSize: "0.75rem" }}
                        onClick={() => handleAddressClick(msg.address)}
                        title="Click to filter by this address"
                      >
                        {msg.alias ? (
                          <div className="d-flex flex-column lh-sm">
                            <span className="text-primary">{msg.alias}</span>
                            <span
                              className="font-monospace text-body-secondary"
                              style={{ fontSize: "0.65rem" }}
                            >
                              {msg.address}
                            </span>
                          </div>
                        ) : (
                          <span className="font-monospace">{msg.address}</span>
                        )}
                      </button>
                    </td>
                    <td>
                      <span
                        className={`badge ${typeBadge.className}`}
                        style={{ fontSize: "0.6rem" }}
                      >
                        {typeBadge.label}
                      </span>
                    </td>
                    <td
                      className="text-break"
                      style={{
                        whiteSpace: "pre-wrap",
                        wordBreak: "break-word",
                      }}
                    >
                      {msg.message || (
                        <span className="text-body-secondary fst-italic">
                          (alert tone)
                        </span>
                      )}
                    </td>
                    <td>
                      <button
                        className="btn btn-link btn-sm p-0 text-body-secondary"
                        onClick={() => handleCopyMessage(msg)}
                        title="Copy message"
                      >
                        <Copy size={12} />
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
