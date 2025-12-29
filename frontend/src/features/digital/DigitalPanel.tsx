import { Radio, AlertCircle, Wifi, WifiOff, RefreshCw } from "lucide-react";
import { useDigitalMessages } from "../../hooks/useDigitalMessages";
import { POCSAGMessageLog } from "./POCSAGMessageLog";
import Spinner from "../../components/primitives/Spinner.react";

interface DigitalPanelProps {
  captureId: string;
  captureName?: string;
}

export function DigitalPanel({ captureId, captureName }: DigitalPanelProps) {
  const { messages, channels, isLoading, isError, error } = useDigitalMessages(
    captureId,
    {
      enabled: !!captureId,
      limit: 100,
      refetchInterval: 2000,
    },
  );

  // Calculate stats
  const lastMessageTime = messages.length > 0 ? messages[0].timestamp : null;
  const timeSinceLastMessage = lastMessageTime
    ? Math.floor(Date.now() / 1000 - lastMessageTime)
    : null;

  return (
    <div className="d-flex flex-column h-100 p-3">
      {/* Header */}
      <div className="d-flex justify-content-between align-items-center mb-3">
        <div className="d-flex align-items-center gap-2">
          <Radio size={20} className="text-primary" />
          <h5 className="mb-0">POCSAG Pager Messages</h5>
          {captureName && (
            <span className="badge bg-body-secondary text-body-secondary">
              {captureName}
            </span>
          )}
        </div>

        {/* Connection status */}
        <div className="d-flex align-items-center gap-3">
          {channels.length > 0 ? (
            <span className="d-flex align-items-center gap-1 text-success small">
              <Wifi size={14} />
              {channels.length} channel{channels.length !== 1 ? "s" : ""} active
            </span>
          ) : (
            <span className="d-flex align-items-center gap-1 text-body-secondary small">
              <WifiOff size={14} />
              No POCSAG channels
            </span>
          )}

          {isLoading && (
            <RefreshCw size={14} className="text-body-secondary spin" />
          )}
        </div>
      </div>

      {/* Error state */}
      {isError && (
        <div className="alert alert-danger d-flex align-items-center gap-2 py-2">
          <AlertCircle size={16} />
          <span>Error loading messages: {String(error)}</span>
        </div>
      )}

      {/* No channels configured */}
      {!isLoading && channels.length === 0 && (
        <div className="flex-grow-1 d-flex flex-column justify-content-center align-items-center text-body-secondary">
          <Radio size={48} className="mb-3 opacity-50" />
          <h6>No POCSAG Channels Configured</h6>
          <p className="text-center small mb-0" style={{ maxWidth: "400px" }}>
            To receive pager messages, create an NBFM channel and enable POCSAG
            decoding in the channel settings. Common paging frequencies include
            148.0625 MHz (SA-GRN) and 929 MHz band.
          </p>
        </div>
      )}

      {/* Loading state */}
      {isLoading && channels.length === 0 && (
        <div className="flex-grow-1 d-flex justify-content-center align-items-center">
          <Spinner size="md" />
        </div>
      )}

      {/* Message log */}
      {channels.length > 0 && (
        <>
          <div className="flex-grow-1" style={{ minHeight: 0 }}>
            <POCSAGMessageLog
              messages={messages}
              channels={channels.map((ch) => ({
                id: ch.id,
                name: ch.name,
                autoName: ch.autoName,
              }))}
              maxHeight={600}
            />
          </div>

          {/* Footer stats */}
          <div className="d-flex justify-content-between align-items-center mt-2 pt-2 border-top small text-body-secondary">
            <span>{messages.length} messages received</span>
            {timeSinceLastMessage !== null && (
              <span>
                Last message:{" "}
                {timeSinceLastMessage < 60
                  ? `${timeSinceLastMessage}s ago`
                  : timeSinceLastMessage < 3600
                    ? `${Math.floor(timeSinceLastMessage / 60)}m ago`
                    : `${Math.floor(timeSinceLastMessage / 3600)}h ago`}
              </span>
            )}
          </div>
        </>
      )}
    </div>
  );
}
