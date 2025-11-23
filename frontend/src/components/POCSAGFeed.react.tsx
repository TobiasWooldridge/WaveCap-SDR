import { useState } from "react";
import { ChevronUp, ChevronDown, MessageSquare } from "lucide-react";
import type { POCSAGMessage } from "../types";
import { usePOCSAGMessages } from "../hooks/useDecodedMessages";
import Flex from "./primitives/Flex.react";

interface POCSAGFeedProps {
  channelId: string;
  enabled?: boolean;
}

function formatTimestamp(ts: number): string {
  const date = new Date(ts * 1000);
  return date.toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function formatAddress(address: number): string {
  return address.toString().padStart(7, "0");
}

function getMessageTypeLabel(msgType: POCSAGMessage["messageType"]): string {
  switch (msgType) {
    case "numeric":
      return "NUM";
    case "alpha":
      return "TXT";
    case "alert_only":
      return "ALT";
    case "alpha_2":
      return "TXT2";
  }
}

function getMessageTypeBadgeClass(
  msgType: POCSAGMessage["messageType"],
): string {
  switch (msgType) {
    case "numeric":
      return "bg-info text-dark";
    case "alpha":
    case "alpha_2":
      return "bg-success";
    case "alert_only":
      return "bg-warning text-dark";
  }
}

export const POCSAGFeed = ({ channelId, enabled = true }: POCSAGFeedProps) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const { data: messages = [], isLoading } = usePOCSAGMessages(channelId, {
    enabled,
    limit: 20,
    refetchInterval: 2000,
  });

  if (!enabled) {
    return null;
  }

  const hasMessages = messages.length > 0;
  const latestMessage = messages[0];

  return (
    <div
      className="border rounded bg-dark text-light"
      style={{ fontSize: "10px" }}
    >
      {/* Header - always visible */}
      <button
        className="btn btn-sm w-100 text-start d-flex justify-content-between align-items-center p-2"
        onClick={() => setIsExpanded(!isExpanded)}
        style={{ background: "transparent", border: "none", color: "inherit" }}
      >
        <Flex align="center" gap={1}>
          <span
            className="badge bg-warning text-dark"
            style={{ fontSize: "8px" }}
          >
            POCSAG
          </span>
          <MessageSquare size={12} />
          <span className="fw-semibold">Pager Feed</span>
          {hasMessages && (
            <span className="badge bg-secondary" style={{ fontSize: "8px" }}>
              {messages.length}
            </span>
          )}
        </Flex>
        {isExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </button>

      {/* Latest message preview (when collapsed) */}
      {!isExpanded && latestMessage && (
        <div className="px-2 pb-2">
          <Flex direction="column" gap={1}>
            <Flex align="center" gap={1}>
              <span className="text-muted" style={{ fontSize: "8px" }}>
                {formatTimestamp(latestMessage.timestamp)}
              </span>
              <span
                className={`badge ${getMessageTypeBadgeClass(latestMessage.messageType)}`}
                style={{ fontSize: "7px" }}
              >
                {getMessageTypeLabel(latestMessage.messageType)}
              </span>
              <span className="font-monospace" style={{ fontSize: "9px" }}>
                {formatAddress(latestMessage.address)}
              </span>
            </Flex>
            <div
              className="text-truncate font-monospace"
              style={{ fontSize: "10px" }}
              title={latestMessage.message || "(alert only)"}
            >
              {latestMessage.message || "(alert only)"}
            </div>
          </Flex>
        </div>
      )}

      {/* Expanded message list */}
      {isExpanded && (
        <div className="border-top border-secondary">
          {isLoading && messages.length === 0 && (
            <div className="p-2 text-center text-muted">Loading...</div>
          )}
          {!isLoading && messages.length === 0 && (
            <div className="p-2 text-center text-muted">
              No messages received yet
            </div>
          )}
          {messages.length > 0 && (
            <div
              style={{
                maxHeight: "200px",
                overflowY: "auto",
              }}
            >
              {messages.map((msg, idx) => (
                <div
                  key={`${msg.timestamp}-${msg.address}-${idx}`}
                  className={`p-2 ${idx !== messages.length - 1 ? "border-bottom border-secondary" : ""}`}
                >
                  <Flex direction="column" gap={1}>
                    <Flex align="center" gap={1} justify="between">
                      <Flex align="center" gap={1}>
                        <span className="text-muted" style={{ fontSize: "8px" }}>
                          {formatTimestamp(msg.timestamp)}
                        </span>
                        <span
                          className={`badge ${getMessageTypeBadgeClass(msg.messageType)}`}
                          style={{ fontSize: "7px" }}
                        >
                          {getMessageTypeLabel(msg.messageType)}
                        </span>
                      </Flex>
                      <span
                        className="font-monospace text-info"
                        style={{ fontSize: "9px" }}
                        title={`Address: ${msg.address}, Function: ${msg.function}`}
                      >
                        {formatAddress(msg.address)}
                      </span>
                    </Flex>
                    <div
                      className="font-monospace"
                      style={{
                        fontSize: "10px",
                        wordBreak: "break-word",
                        whiteSpace: "pre-wrap",
                      }}
                    >
                      {msg.message || "(alert only)"}
                    </div>
                  </Flex>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
