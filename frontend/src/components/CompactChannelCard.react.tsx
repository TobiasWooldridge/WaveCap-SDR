import { useState, useRef, useEffect } from "react";
import { Volume2, VolumeX, Trash2, Copy, CheckCircle, Settings, ChevronUp, ChevronDown, Edit2 } from "lucide-react";
import type { Capture, Channel } from "../types";
import { useUpdateChannel, useDeleteChannel } from "../hooks/useChannels";
import { useToast } from "../hooks/useToast";
import { formatFrequencyMHz } from "../utils/frequency";
import Button from "./primitives/Button.react";
import Flex from "./primitives/Flex.react";
import Slider from "./primitives/Slider.react";
import FrequencySelector from "./primitives/FrequencySelector.react";
import SignalMeter from "./primitives/SignalMeter.react";
import { FrequencyLabel } from "./FrequencyLabel.react";

interface CompactChannelCardProps {
  channel: Channel;
  capture: Capture;
  isPlaying: boolean;
  onTogglePlay: () => void;
  onCopyUrl: (url: string) => void;
  copiedUrl: string | null;
}

// Format channel ID for display (e.g., "ch1" -> "Ch 1")
function formatChannelId(id: string): string {
  const match = id.match(/^ch(\d+)$/);
  return match ? `Ch ${match[1]}` : id;
}

export const CompactChannelCard = ({
  channel,
  capture,
  isPlaying,
  onTogglePlay,
  onCopyUrl,
  copiedUrl,
}: CompactChannelCardProps) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [showStreamDropdown, setShowStreamDropdown] = useState(false);
  const [isEditingName, setIsEditingName] = useState(false);
  const [editNameValue, setEditNameValue] = useState("");
  const nameInputRef = useRef<HTMLInputElement>(null);
  const updateChannel = useUpdateChannel(capture.id);
  const deleteChannel = useDeleteChannel();
  const toast = useToast();

  const getChannelFrequency = () => capture.centerHz + channel.offsetHz;
  const displayName = channel.name || channel.autoName || formatChannelId(channel.id);

  useEffect(() => {
    if (isEditingName && nameInputRef.current) {
      nameInputRef.current.focus();
      nameInputRef.current.select();
    }
  }, [isEditingName]);

  const handleStartEditName = () => {
    setEditNameValue(channel.name || channel.autoName || "");
    setIsEditingName(true);
  };

  const handleSaveEditName = () => {
    const trimmedValue = editNameValue.trim();
    updateChannel.mutate({
      channelId: channel.id,
      request: { name: trimmedValue || null },
    }, {
      onSuccess: () => {
        toast.success("Channel name updated");
      },
      onError: (error: any) => {
        toast.error(error?.message || "Failed to update channel name");
      },
    });
    setIsEditingName(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleSaveEditName();
    } else if (e.key === "Escape") {
      setIsEditingName(false);
    }
  };

  const handleDelete = () => {
    if (confirm("Delete this channel?")) {
      deleteChannel.mutate(channel.id, {
        onSuccess: () => {
          toast.success("Channel deleted successfully");
        },
        onError: (error: any) => {
          toast.error(error?.message || "Failed to delete channel");
        },
      });
    }
  };

  // Helper function to update channel with toast notifications
  const updateChannelWithToast = (request: any) => {
    updateChannel.mutate({
      channelId: channel.id,
      request,
    }, {
      onError: (error: any) => {
        toast.error(error?.message || "Failed to update channel");
      },
    });
  };

  const streamFormats = [
    { format: 'PCM', ext: '.pcm', label: 'Raw PCM' },
    { format: 'MP3', ext: '.mp3', label: 'MP3 (128k)' },
    { format: 'Opus', ext: '.opus', label: 'Opus' },
    { format: 'AAC', ext: '.aac', label: 'AAC' },
  ];

  return (
    <div className="card shadow-sm h-100">
      {/* Compact Header */}
      <div className="card-header bg-body-tertiary p-2">
        <Flex justify="between" align="center">
          <div className="small fw-semibold text-truncate" style={{ flex: 1, minWidth: 0 }}>
            {formatChannelId(channel.id)} • {formatFrequencyMHz(getChannelFrequency())} MHz
          </div>
          <Flex gap={1}>
            <Button
              use={isPlaying ? "warning" : "success"}
              size="sm"
              onClick={onTogglePlay}
              title={isPlaying ? "Stop Listening" : "Listen Now"}
              className="px-2 py-1"
            >
              <Flex align="center" gap={1}>
                {isPlaying ? <VolumeX size={14} /> : <Volume2 size={14} />}
                <span style={{ fontSize: "11px" }}>{isPlaying ? "Stop" : "Listen"}</span>
              </Flex>
            </Button>
            <Button
              use="secondary"
              size="sm"
              onClick={() => setIsExpanded(!isExpanded)}
              title={isExpanded ? "Collapse" : "Settings"}
              className="p-1"
            >
              {isExpanded ? <ChevronUp size={14} /> : <Settings size={14} />}
            </Button>
            <Button
              use="danger"
              size="sm"
              appearance="outline"
              onClick={handleDelete}
              title="Delete"
              className="p-1"
            >
              <Trash2 size={14} />
            </Button>
          </Flex>
        </Flex>
      </div>

      {/* Compact Body - Always Visible */}
      <div className="card-body p-2">
        <Flex direction="column" gap={2}>
          {/* Channel Info */}
          <div>
            <Flex justify="between" align="center">
              <div className="small text-muted">{channel.mode.toUpperCase()}</div>
              <button
                className="btn btn-sm p-0"
                style={{ width: "16px", height: "16px" }}
                onClick={handleStartEditName}
                title="Edit name"
              >
                <Edit2 size={12} />
              </button>
            </Flex>
            {isEditingName ? (
              <input
                ref={nameInputRef}
                type="text"
                className="form-control form-control-sm mt-1"
                value={editNameValue}
                onChange={(e) => setEditNameValue(e.target.value)}
                onBlur={handleSaveEditName}
                onKeyDown={handleKeyDown}
                placeholder="Channel name"
              />
            ) : (
              <div>
                <div className="fw-semibold" title={channel.name && channel.autoName ? `Auto: ${channel.autoName}` : undefined}>
                  {displayName}
                </div>
                <FrequencyLabel frequencyHz={getChannelFrequency()} autoName={channel.name ? channel.autoName : null} />
              </div>
            )}
          </div>

          {/* Prominent Signal Meters */}
          <div className="border rounded p-2 bg-light">
            <Flex direction="column" gap={1}>
              <Flex align="center" gap={1}>
                <span className="badge bg-success text-white" style={{fontSize: "8px", width: "32px"}}>LIVE</span>
                <div style={{ flex: 1 }}>
                  <SignalMeter signalPowerDb={channel.rssiDb} width={200} height={18} />
                </div>
              </Flex>
              <Flex direction="row" gap={1} align="center" style={{fontSize: "9px"}}>
                <span className="text-muted" style={{ width: "40px" }}>RSSI:</span>
                <span className="fw-semibold">{channel.rssiDb?.toFixed(1) ?? 'N/A'} dB</span>
                <span className="text-muted ms-2" style={{ width: "35px" }}>SNR:</span>
                <span className="fw-semibold">{channel.snrDb?.toFixed(1) ?? 'N/A'} dB</span>
              </Flex>
            </Flex>
          </div>

          {/* Stream URL Dropdown */}
          <div className="dropdown" style={{ position: 'relative' }}>
            <Button
              use="secondary"
              size="sm"
              onClick={() => setShowStreamDropdown(!showStreamDropdown)}
              className="w-100 d-flex justify-content-between align-items-center"
            >
              <span className="small">Copy Stream URL</span>
              <ChevronDown size={12} />
            </Button>
            {showStreamDropdown && (
              <div
                className="dropdown-menu show w-100"
                style={{ position: 'absolute', top: '100%', zIndex: 1000 }}
              >
                {streamFormats.map(({ format, ext, label }) => {
                  const url = `${window.location.origin}/api/v1/stream/channels/${channel.id}${ext}`;
                  const isCopied = copiedUrl === url;
                  return (
                    <button
                      key={format}
                      className="dropdown-item d-flex justify-content-between align-items-center"
                      onClick={() => {
                        onCopyUrl(url);
                        setShowStreamDropdown(false);
                      }}
                    >
                      <span className="small">{label}</span>
                      {isCopied ? <CheckCircle size={12} className="text-success" /> : <Copy size={12} />}
                    </button>
                  );
                })}
              </div>
            )}
          </div>

          {/* Expanded Settings */}
          {isExpanded && (
            <div className="border-top pt-2 mt-1">
              <Flex direction="column" gap={2}>
                {/* Mode Selector */}
                <Flex direction="column" gap={1}>
                  <label className="form-label small mb-0">Mode</label>
                  <select
                    className="form-select form-select-sm"
                    value={channel.mode}
                    onChange={(e) =>
                      updateChannelWithToast({ mode: e.target.value as any })
                    }
                  >
                    <option value="wbfm">WBFM</option>
                    <option value="nbfm">NBFM</option>
                    <option value="am">AM</option>
                    <option value="ssb">SSB</option>
                    <option value="raw">Raw IQ</option>
                    <option value="p25">P25</option>
                    <option value="dmr">DMR</option>
                  </select>
                </Flex>

                {/* Squelch Slider */}
                <Slider
                  label="Squelch"
                  value={channel.squelchDb ?? -60}
                  min={-80}
                  max={0}
                  step={1}
                  coarseStep={10}
                  unit="dB"
                  formatValue={(val) => `${val.toFixed(0)} dB`}
                  onChange={(val) =>
                    updateChannelWithToast({ squelchDb: val })
                  }
                />

                {/* Audio Rate */}
                <Flex direction="column" gap={1}>
                  <label className="form-label small mb-0">Audio Rate</label>
                  <select
                    className="form-select form-select-sm"
                    value={channel.audioRate}
                    onChange={(e) =>
                      updateChannelWithToast({ audioRate: parseInt(e.target.value) })
                    }
                  >
                    <option value={8000}>8 kHz</option>
                    <option value={16000}>16 kHz</option>
                    <option value={24000}>24 kHz</option>
                    <option value={48000}>48 kHz</option>
                  </select>
                </Flex>

                {/* Frequency */}
                <Flex direction="column" gap={1}>
                  <FrequencySelector
                    label="Frequency"
                    value={getChannelFrequency()}
                    min={capture.centerHz - (capture.sampleRate / 2)}
                    max={capture.centerHz + (capture.sampleRate / 2)}
                    step={1000}
                    onChange={(hz) =>
                      updateChannelWithToast({ offsetHz: hz - capture.centerHz })
                    }
                  />
                  <small className="text-muted">
                    Offset: {(channel.offsetHz / 1000).toFixed(0)} kHz
                  </small>
                </Flex>
              </Flex>
            </div>
          )}
        </Flex>
      </div>

      {/* Status Badge Footer */}
      <div className="card-footer p-1 bg-body-tertiary">
        <Flex justify="between" align="center">
          <span className={`badge bg-${channel.state === "running" ? "success" : "secondary"}`} style={{fontSize: "9px"}}>
            {channel.state}
          </span>
          <small className="text-muted" style={{fontSize: "9px"}}>
            {(channel.offsetHz / 1000).toFixed(0)} kHz • {channel.audioRate / 1000} kHz
          </small>
        </Flex>
      </div>
    </div>
  );
};
