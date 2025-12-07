import { useState, useRef, useEffect } from "react";
import { Volume2, VolumeX, Trash2, Settings, ChevronUp, Edit2 } from "lucide-react";
import type { Capture, Channel } from "../../types";
import { useUpdateChannel, useDeleteChannel } from "../../hooks/useChannels";
import { useChannelAudio } from "../../hooks/useAudio";
import { useToast } from "../../hooks/useToast";
import { formatFrequencyMHz } from "../../utils/frequency";
import { copyToClipboard } from "../../utils/clipboard";
import Button from "../../components/primitives/Button.react";
import Flex from "../../components/primitives/Flex.react";
import SMeter from "../../components/primitives/SMeter.react";
import AudioWaveform from "../../components/primitives/AudioWaveform.react";
import { FrequencyLabel } from "../../components/FrequencyLabel.react";
import { ChannelSettings } from "./ChannelSettings";
import { RdsDisplay } from "./RdsDisplay";
import { StreamUrlDropdown } from "./StreamUrlDropdown";

interface ChannelCardProps {
  channel: Channel;
  capture: Capture;
}

export function ChannelCard({ channel, capture }: ChannelCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [isEditingName, setIsEditingName] = useState(false);
  const [editNameValue, setEditNameValue] = useState("");
  const nameInputRef = useRef<HTMLInputElement>(null);

  const { isPlaying, toggle: togglePlay } = useChannelAudio(channel.id);
  const updateChannel = useUpdateChannel(capture.id);
  const deleteChannel = useDeleteChannel();
  const toast = useToast();

  const channelFrequency = capture.centerHz + channel.offsetHz;
  const displayName = channel.name || channel.autoName || formatChannelId(channel.id);

  // Focus input when editing
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
    updateChannel.mutate(
      { channelId: channel.id, request: { name: trimmedValue || null } },
      {
        onSuccess: () => toast.success("Channel name updated"),
        onError: (error: Error) => toast.error(error.message),
      }
    );
    setIsEditingName(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") handleSaveEditName();
    if (e.key === "Escape") setIsEditingName(false);
  };

  const handleDelete = () => {
    if (confirm("Delete this channel?")) {
      deleteChannel.mutate(channel.id, {
        onSuccess: () => toast.success("Channel deleted"),
        onError: (error: Error) => toast.error(error.message),
      });
    }
  };

  const handleCopyUrl = async (url: string) => {
    const success = await copyToClipboard(url);
    if (success) {
      toast.success("URL copied to clipboard");
    } else {
      toast.error("Failed to copy URL");
    }
  };

  // Check if channel is outside spectrum range
  const spectrumMin = capture.centerHz - capture.sampleRate / 2;
  const spectrumMax = capture.centerHz + capture.sampleRate / 2;
  const isOutOfRange = channelFrequency < spectrumMin || channelFrequency > spectrumMax;

  return (
    <div className="card shadow-sm">
      {/* Header */}
      <div className="card-header bg-body-tertiary p-2">
        <Flex justify="between" align="center">
          <div className="small fw-semibold text-truncate" style={{ flex: 1, minWidth: 0 }}>
            {formatChannelId(channel.id)} • {formatFrequencyMHz(channelFrequency)} MHz
          </div>
          <Flex gap={1}>
            <Button
              use={isPlaying ? "warning" : "success"}
              size="sm"
              onClick={togglePlay}
              title={isPlaying ? "Stop" : "Listen"}
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

      {/* Body */}
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
              <>
                <div className="fw-semibold">{displayName}</div>
                <FrequencyLabel
                  frequencyHz={channelFrequency}
                  autoName={channel.name ? channel.autoName : null}
                />
              </>
            )}

            {isOutOfRange && (
              <div
                className="alert alert-warning py-1 px-2 mt-1 mb-0"
                style={{ fontSize: "0.7rem" }}
              >
                <strong>Warning:</strong> Channel outside observable spectrum
              </div>
            )}
          </div>

          {/* S-Meter */}
          <div className="border rounded p-2 bg-light">
            <Flex direction="column" gap={1}>
              <Flex align="center" gap={1}>
                <span
                  className="badge bg-success text-white"
                  style={{ fontSize: "8px", width: "32px" }}
                >
                  LIVE
                </span>
                <div style={{ flex: 1 }}>
                  <SMeter rssiDbFs={channel.rssiDb} frequencyHz={channelFrequency} width={200} height={24} />
                </div>
              </Flex>
              <Flex direction="row" gap={1} align="center" style={{ fontSize: "9px" }}>
                <span className="text-muted" style={{ width: "40px" }}>RSSI:</span>
                <span className="fw-semibold">{channel.rssiDb?.toFixed(1) ?? "N/A"} dBFS</span>
                <span className="text-muted ms-2" style={{ width: "35px" }}>SNR:</span>
                <span className="fw-semibold">{channel.snrDb?.toFixed(1) ?? "N/A"} dB</span>
              </Flex>
            </Flex>
          </div>

          {/* RDS Display (WBFM only) */}
          {channel.mode === "wbfm" && channel.rdsData && <RdsDisplay rdsData={channel.rdsData} />}

          {/* Audio Waveform */}
          {isPlaying && (
            <div className="border rounded p-2 bg-dark">
              <AudioWaveform channelId={channel.id} isPlaying={isPlaying} width={200} height={40} />
            </div>
          )}

          {/* Stream URL */}
          <StreamUrlDropdown channelId={channel.id} onCopyUrl={handleCopyUrl} />

          {/* Expanded Settings */}
          {isExpanded && (
            <ChannelSettings channel={channel} capture={capture} />
          )}
        </Flex>
      </div>
    </div>
  );
}

function formatChannelId(id: string): string {
  const match = id.match(/^ch(\d+)$/);
  return match ? `Ch ${match[1]}` : id;
}
