import { useState, useRef, useEffect } from "react";
import { Radio, Plus, Trash2, Copy, CheckCircle, Play, Pause } from "lucide-react";
import type { Capture, Channel } from "../types";
import {
  useChannels,
  useCreateChannel,
  useDeleteChannel,
  useStartChannel,
  useStopChannel,
} from "../hooks/useChannels";
import { formatFrequencyMHz } from "../utils/frequency";
import Button from "./primitives/Button.react";
import Flex from "./primitives/Flex.react";
import Slider from "./primitives/Slider.react";
import Spinner from "./primitives/Spinner.react";

interface ChannelManagerProps {
  capture: Capture;
}

// Format channel ID for display (e.g., "ch1" -> "Channel 1")
function formatChannelId(id: string): string {
  const match = id.match(/^ch(\d+)$/);
  return match ? `Channel ${match[1]}` : id;
}

export const ChannelManager = ({ capture }: ChannelManagerProps) => {
  const { data: channels, isLoading } = useChannels(capture.id);
  const createChannel = useCreateChannel();
  const deleteChannel = useDeleteChannel();
  const startChannel = useStartChannel(capture.id);
  const stopChannel = useStopChannel(capture.id);

  const [showNewChannel, setShowNewChannel] = useState(false);
  const [newChannelFrequency, setNewChannelFrequency] = useState<number>(capture.centerHz);
  const [newChannelMode, setNewChannelMode] = useState<"wbfm" | "nfm" | "am">("wbfm");
  const [copiedUrl, setCopiedUrl] = useState<string | null>(null);
  const [playingChannel, setPlayingChannel] = useState<string | null>(null);

  const audioContextRef = useRef<AudioContext | null>(null);
  const audioSourceRef = useRef<AudioBufferSourceNode | null>(null);
  const audioWorkletRef = useRef<AudioWorkletNode | null>(null);
  const streamReaderRef = useRef<ReadableStreamDefaultReader<Uint8Array> | null>(null);
  const shouldPlayRef = useRef<boolean>(false);
  const nextStartTimeRef = useRef<number>(0);

  const stopAudio = () => {
    // Signal to stop playback
    shouldPlayRef.current = false;

    // Stop any ongoing stream
    if (streamReaderRef.current) {
      streamReaderRef.current.cancel();
      streamReaderRef.current = null;
    }

    // Disconnect audio nodes
    if (audioWorkletRef.current) {
      audioWorkletRef.current.disconnect();
      audioWorkletRef.current = null;
    }

    if (audioSourceRef.current) {
      audioSourceRef.current.stop();
      audioSourceRef.current = null;
    }
  };

  // Stop audio and reset frequency when capture changes
  useEffect(() => {
    setNewChannelFrequency(capture.centerHz);
    return () => {
      stopAudio();
    };
  }, [capture.id]);

  const handleCreateChannel = () => {
    // Calculate offset from absolute frequency
    const offsetHz = newChannelFrequency - capture.centerHz;

    createChannel.mutate({
      captureId: capture.id,
      request: {
        mode: newChannelMode,
        offsetHz,
        audioRate: 48000,
      },
    }, {
      onSuccess: () => {
        setShowNewChannel(false);
        setNewChannelFrequency(capture.centerHz);
      },
    });
  };

  const handleDeleteChannel = (channelId: string) => {
    if (confirm("Delete this channel?")) {
      deleteChannel.mutate(channelId);
    }
  };

  const copyToClipboard = (text: string, url: string) => {
    navigator.clipboard.writeText(text).then(() => {
      setCopiedUrl(url);
      setTimeout(() => setCopiedUrl(null), 2000);
    });
  };

  const playPCMAudio = async (channelId: string) => {
    try {
      // Initialize AudioContext if needed
      if (!audioContextRef.current) {
        audioContextRef.current = new AudioContext({ sampleRate: 48000 });
      }

      const audioContext = audioContextRef.current;

      // Initialize playback timing
      shouldPlayRef.current = true;
      nextStartTimeRef.current = audioContext.currentTime;

      const streamUrl = getStreamUrl(channelId);
      const response = await fetch(streamUrl);

      if (!response.ok || !response.body) {
        throw new Error('Failed to fetch audio stream');
      }

      const reader = response.body.getReader();
      streamReaderRef.current = reader;

      const bufferSize = 4096;
      let pcmBuffer: number[] = [];

      const processChunk = async () => {
        while (shouldPlayRef.current) {
          const { done, value } = await reader.read();

          if (done) break;

          // Convert PCM16 to Float32
          const dataView = new DataView(value.buffer, value.byteOffset, value.byteLength);
          // Only read complete 16-bit samples (need 2 bytes per sample)
          const sampleCount = Math.floor(value.length / 2);
          for (let i = 0; i < sampleCount; i++) {
            const sample = dataView.getInt16(i * 2, true) / 32768.0;
            pcmBuffer.push(sample);
          }

          // When we have enough samples, play them
          while (pcmBuffer.length >= bufferSize && shouldPlayRef.current) {
            const chunk = pcmBuffer.splice(0, bufferSize);
            const audioBuffer = audioContext.createBuffer(1, chunk.length, 48000);
            const channelData = audioBuffer.getChannelData(0);

            for (let i = 0; i < chunk.length; i++) {
              channelData[i] = chunk[i];
            }

            const source = audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(audioContext.destination);

            // Schedule this buffer to play at the right time
            const startTime = Math.max(nextStartTimeRef.current, audioContext.currentTime);
            source.start(startTime);
            nextStartTimeRef.current = startTime + audioBuffer.duration;
          }
        }
      };

      processChunk().catch((error) => {
        console.error('Audio playback error:', error);
        setPlayingChannel(null);
      });

    } catch (error) {
      console.error('Failed to play audio:', error);
      setPlayingChannel(null);
    }
  };

  const togglePlay = async (channel: Channel) => {
    if (playingChannel === channel.id) {
      stopAudio();
      setPlayingChannel(null);
      stopChannel.mutate(channel.id);
      return;
    }

    if (playingChannel) {
      stopAudio();
      setPlayingChannel(null);
    }

    try {
      if (channel.state !== "running") {
        await startChannel.mutateAsync(channel.id);
      }
      setPlayingChannel(channel.id);
      playPCMAudio(channel.id);
    } catch (error) {
      console.error("Unable to start channel for playback:", error);
      setPlayingChannel(null);
    }
  };

  const getStreamUrl = (channelId: string) => {
    return `${window.location.origin}/api/v1/stream/channels/${channelId}.pcm`;
  };

  const getChannelFrequency = (channel: Channel) => {
    return capture.centerHz + channel.offsetHz;
  };

  return (
    <div className="card shadow-sm">
      <div className="card-header bg-body-tertiary">
        <Flex justify="between" align="center">
          <Flex align="center" gap={2}>
            <Radio size={20} />
            <h3 className="h6 mb-0">Channels</h3>
          </Flex>
          <Button
            use="primary"
            size="sm"
            onClick={() => setShowNewChannel(!showNewChannel)}
            disabled={capture.state !== "running"}
          >
            <Plus size={16} />
          </Button>
        </Flex>
      </div>

      {/* New Channel Form */}
      {showNewChannel && (
        <div className="card-body border-bottom">
          <Flex direction="column" gap={3}>
            <h6 className="mb-0">New Channel</h6>

            <Flex direction="column" gap={1}>
              <label className="form-label small mb-1">Mode</label>
              <select
                className="form-select form-select-sm"
                value={newChannelMode}
                onChange={(e) => setNewChannelMode(e.target.value as any)}
              >
                <option value="wbfm">WBFM (Wideband FM)</option>
                <option value="nfm">NFM (Narrowband FM)</option>
                <option value="am">AM</option>
              </select>
            </Flex>

            <Slider
              label="Frequency"
              value={newChannelFrequency}
              min={capture.centerHz - (capture.sampleRate / 2)}
              max={capture.centerHz + (capture.sampleRate / 2)}
              step={1000}
              coarseStep={100000}
              unit="MHz"
              formatValue={(hz) => formatFrequencyMHz(hz)}
              onChange={setNewChannelFrequency}
            />
            <small className="text-muted">
              Offset: {((newChannelFrequency - capture.centerHz) / 1000).toFixed(0)} kHz
            </small>

            <Flex gap={2}>
              <Button
                use="success"
                size="sm"
                onClick={handleCreateChannel}
                disabled={createChannel.isPending}
              >
                Create
              </Button>
              <Button
                use="secondary"
                size="sm"
                onClick={() => setShowNewChannel(false)}
              >
                Cancel
              </Button>
            </Flex>
          </Flex>
        </div>
      )}

      <div className="card-body">
        {isLoading && (
          <Flex justify="center" className="py-3">
            <Spinner size="sm" />
          </Flex>
        )}

        {!isLoading && (!channels || channels.length === 0) && (
          <div className="text-muted small text-center py-3">
            No channels. Click + to create one.
          </div>
        )}

        {channels && channels.length > 0 && (
          <Flex direction="column" gap={3}>
            {channels.map((channel) => {
              const isPlaying = playingChannel === channel.id;

              return (
                <div key={channel.id} className="border rounded p-3">
                  <Flex direction="column" gap={2}>
                    {/* Channel Header */}
                    <Flex justify="between" align="center">
                      <div>
                        <div className="fw-semibold">{formatChannelId(channel.id)}</div>
                        <div className="small text-muted">
                          {formatFrequencyMHz(getChannelFrequency(channel))} MHz • {channel.mode.toUpperCase()}
                        </div>
                      </div>
                      <Flex gap={1}>
                        <Button
                          use={isPlaying ? "warning" : "success"}
                          size="sm"
                          appearance="outline"
                          onClick={() => togglePlay(channel)}
                          title={isPlaying ? "Pause" : "Play"}
                          aria-label={isPlaying ? "Pause" : "Play"}
                          className="px-2"
                        >
                          {isPlaying ? <Pause size={14} /> : <Play size={14} />}
                        </Button>
                        <span className={`badge bg-${channel.state === "running" ? "success" : "secondary"}`}>
                          {channel.state}
                        </span>
                        <Button
                          use="danger"
                          size="sm"
                          appearance="outline"
                          onClick={() => handleDeleteChannel(channel.id)}
                          title="Delete"
                          aria-label="Delete"
                          className="px-2"
                        >
                          <Trash2 size={14} />
                        </Button>
                      </Flex>
                    </Flex>

                    {/* Stream URLs */}
                    <Flex direction="column" gap={1}>
                      <label className="small text-muted mb-0">Stream URLs</label>
                      {[
                        { format: 'PCM', ext: '.pcm', label: 'Raw PCM' },
                        { format: 'MP3', ext: '.mp3', label: 'MP3 (128k)' },
                        { format: 'Opus', ext: '.opus', label: 'Opus' },
                        { format: 'AAC', ext: '.aac', label: 'AAC' },
                      ].map(({ format, ext, label }) => {
                        const formatUrl = `${window.location.origin}/api/v1/stream/channels/${channel.id}${ext}`;
                        const isFormatCopied = copiedUrl === formatUrl;
                        return (
                          <Flex key={format} gap={1} align="center">
                            <span className="badge bg-secondary text-nowrap" style={{ width: '80px', fontSize: '10px' }}>
                              {label}
                            </span>
                            <input
                              type="text"
                              className="form-control form-control-sm font-monospace small"
                              value={formatUrl}
                              readOnly
                              onClick={(e) => (e.target as HTMLInputElement).select()}
                            />
                            <Button
                              use={isFormatCopied ? "success" : "secondary"}
                              size="sm"
                              appearance="outline"
                              onClick={() => copyToClipboard(formatUrl, formatUrl)}
                              title={isFormatCopied ? "Copied!" : "Copy URL"}
                              aria-label={isFormatCopied ? "Copied!" : "Copy URL"}
                              className="px-2"
                            >
                              {isFormatCopied ? <CheckCircle size={14} /> : <Copy size={14} />}
                            </Button>
                          </Flex>
                        );
                      })}
                    </Flex>

                    {/* Channel Details */}
                    <div className="small text-muted">
                      <strong>Offset:</strong> {channel.offsetHz.toLocaleString()} Hz •{" "}
                      <strong>Audio Rate:</strong> {channel.audioRate} Hz
                      {channel.squelchDb !== null && (
                        <> • <strong>Squelch:</strong> {channel.squelchDb} dB</>
                      )}
                    </div>
                  </Flex>
                </div>
              );
            })}
          </Flex>
        )}
      </div>

      {capture.state !== "running" && (
        <div className="card-footer bg-body-tertiary">
          <small className="text-muted">
            Start the capture to create channels and stream audio.
          </small>
        </div>
      )}
    </div>
  );
};
