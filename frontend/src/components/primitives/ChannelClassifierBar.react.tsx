import { useEffect, useState, useCallback, useMemo } from "react";
import { Radio, RotateCcw, ChevronDown, ChevronUp } from "lucide-react";
import type { Capture } from "../../types";
import { formatFrequencyWithUnit } from "../../utils/frequency";
import { FrequencyDisplay } from "./FrequencyDisplay.react";

interface ClassifiedChannel {
  freqHz: number;
  powerDb: number;
  stdDevDb: number;
  channelType: "control" | "voice" | "variable" | "unknown";
}

interface ClassifierStatus {
  elapsed_seconds: number;
  sample_count: number;
  is_ready: boolean;
  remaining_seconds: number;
}

interface ClassifiedChannelsResponse {
  channels: ClassifiedChannel[];
  status: ClassifierStatus;
}

interface ChannelClassifierBarProps {
  capture: Capture;
}

export default function ChannelClassifierBar({
  capture,
}: ChannelClassifierBarProps) {
  const [channels, setChannels] = useState<ClassifiedChannel[]>([]);
  const [status, setStatus] = useState<ClassifierStatus | null>(null);
  const [isListExpanded, setIsListExpanded] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch classified channels from API
  const fetchChannels = useCallback(async (reset: boolean = false) => {
    if (capture.state !== "running") return;

    try {
      const url = `/api/v1/captures/${capture.id}/classified-channels${reset ? "?reset=true" : ""}`;
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data: ClassifiedChannelsResponse = await response.json();
      setChannels(data.channels);
      setStatus(data.status);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch");
    }
  }, [capture.id, capture.state]);

  // Poll for updates every 2 seconds
  useEffect(() => {
    if (capture.state !== "running") {
      setChannels([]);
      setStatus(null);
      return;
    }

    fetchChannels();
    const interval = setInterval(() => fetchChannels(), 2000);
    return () => clearInterval(interval);
  }, [capture.id, capture.state, fetchChannels]);

  // Reset when capture parameters change
  useEffect(() => {
    setChannels([]);
    setStatus(null);
  }, [capture.centerHz, capture.sampleRate]);

  const handleReset = useCallback(() => {
    fetchChannels(true);
  }, [fetchChannels]);

  // Memoized counts
  const { controlCount, voiceCount } = useMemo(() => ({
    controlCount: channels.filter(c => c.channelType === "control").length,
    voiceCount: channels.filter(c => c.channelType === "voice").length,
  }), [channels]);

  const formatFreq = (hz: number) => formatFrequencyWithUnit(hz, 4);

  const getTypeLabel = (type: string) => {
    switch (type) {
      case "control": return "Control";
      case "voice": return "Voice";
      case "variable": return "Variable";
      default: return type;
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case "control": return "#00ff00";
      case "voice": return "#ff6600";
      case "variable": return "#0088ff";
      default: return "#888888";
    }
  };

  const isReady = status?.is_ready ?? false;
  const remainingSeconds = status?.remaining_seconds ?? 60;
  const elapsedSeconds = status?.elapsed_seconds ?? 0;

  return (
    <div className="card shadow-sm mt-2">
      <div className="card-header bg-body-tertiary py-1 px-2">
        <div className="d-flex justify-content-between align-items-center">
          <small className="fw-semibold mb-0 d-flex align-items-center gap-1">
            <Radio size={12} />
            Channel Classifier
            {isReady && channels.length > 0 && (
              <span className="badge bg-success text-white ms-2" style={{ fontSize: "8px" }}>
                {controlCount} CC / {voiceCount} VC
              </span>
            )}
          </small>
          <div className="d-flex align-items-center gap-1">
            {isReady && channels.length > 0 && (
              <button
                className="btn btn-sm btn-outline-secondary p-0 d-flex align-items-center justify-content-center"
                style={{ width: "20px", height: "20px" }}
                onClick={() => setIsListExpanded(!isListExpanded)}
                title={isListExpanded ? "Hide channel list" : "Show channel list"}
              >
                {isListExpanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
              </button>
            )}
            <button
              className="btn btn-sm btn-outline-secondary p-0 d-flex align-items-center justify-content-center"
              style={{ width: "20px", height: "20px" }}
              onClick={handleReset}
              title="Reset classification"
            >
              <RotateCcw size={12} />
            </button>
          </div>
        </div>
      </div>
      <div className="card-body p-2">
        {/* Status bar */}
        <div
          className="d-flex align-items-center justify-content-between"
          style={{ fontSize: "10px", fontFamily: "monospace" }}
        >
          <div className="d-flex align-items-center gap-3">
            <span className="d-flex align-items-center gap-1">
              <span style={{ width: "8px", height: "8px", backgroundColor: "#00ff00", display: "inline-block" }} />
              Control
            </span>
            <span className="d-flex align-items-center gap-1">
              <span style={{ width: "8px", height: "8px", backgroundColor: "#ff6600", display: "inline-block" }} />
              Voice
            </span>
            <span className="d-flex align-items-center gap-1">
              <span style={{ width: "8px", height: "8px", backgroundColor: "#0088ff", display: "inline-block" }} />
              Variable
            </span>
          </div>
          <span className="text-muted">
            {error ? (
              <span className="text-danger">{error}</span>
            ) : !isReady ? (
              `Collecting... ${Math.round(remainingSeconds)}s remaining`
            ) : (
              `${channels.length} signals (${Math.round(elapsedSeconds)}s)`
            )}
          </span>
        </div>

        {/* Channel list */}
        {isListExpanded && channels.length > 0 && (
          <div
            className="mt-2"
            style={{
              maxHeight: "200px",
              overflowY: "auto",
              fontSize: "10px",
              fontFamily: "monospace",
            }}
          >
            <table className="table table-sm table-dark mb-0" style={{ fontSize: "10px" }}>
              <thead>
                <tr>
                  <th style={{ width: "60px" }}>Type</th>
                  <th>Frequency</th>
                  <th style={{ width: "60px" }}>Power</th>
                  <th style={{ width: "60px" }}>Std Dev</th>
                </tr>
              </thead>
              <tbody>
                {channels.map((ch, idx) => (
                  <tr key={idx}>
                    <td>
                      <span
                        className="badge"
                        style={{
                          backgroundColor: getTypeColor(ch.channelType),
                          color: ch.channelType === "control" ? "#000" : "#fff",
                          fontSize: "9px",
                        }}
                      >
                        {getTypeLabel(ch.channelType)}
                      </span>
                    </td>
                    <td>
                      <FrequencyDisplay frequencyHz={ch.freqHz} decimals={4}  unit="MHz"/>
                    </td>
                    <td>{ch.powerDb.toFixed(1)} dB</td>
                    <td>{ch.stdDevDb.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Simple channel summary when not expanded */}
        {!isListExpanded && isReady && channels.length > 0 && (
          <div className="mt-1" style={{ fontSize: "10px", fontFamily: "monospace" }}>
            {channels.slice(0, 5).map((ch, idx) => (
              <span
                key={idx}
                className="badge me-1"
                style={{
                  backgroundColor: getTypeColor(ch.channelType),
                  color: ch.channelType === "control" ? "#000" : "#fff",
                  fontSize: "9px",
                }}
                title={`${formatFreq(ch.freqHz)} - ${ch.powerDb.toFixed(1)} dB, std ${ch.stdDevDb.toFixed(2)}`}
              >
                <FrequencyDisplay frequencyHz={ch.freqHz} decimals={3} unit="MHz" />
              </span>
            ))}
            {channels.length > 5 && (
              <span className="text-muted">+{channels.length - 5} more</span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
