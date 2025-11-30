import { useState, useRef, useEffect, useCallback } from "react";
import { Radio, Plus, Volume2, VolumeX } from "lucide-react";
import type { Capture } from "../types";
import {
  useChannels,
  useCreateChannel,
  useStartChannel,
  useStopChannel,
} from "../hooks/useChannels";
import { useToast } from "../hooks/useToast";
import Button from "./primitives/Button.react";
import Flex from "./primitives/Flex.react";
import Slider from "./primitives/Slider.react";
import FrequencySelector from "./primitives/FrequencySelector.react";
import Spinner from "./primitives/Spinner.react";
import { CompactChannelCard } from "./CompactChannelCard.react";
import { SkeletonChannelCard } from "./primitives/Skeleton.react";

// Type for tracking individual channel streams
interface ChannelStream {
  ws: WebSocket;
  gainNode: GainNode;
  shouldPlay: boolean;
  nextStartTime: number;
}

interface ChannelManagerProps {
  capture: Capture;
}

export const ChannelManager = ({ capture }: ChannelManagerProps) => {
  const { data: channels, isLoading } = useChannels(capture.id);
  const createChannel = useCreateChannel();
  const startChannel = useStartChannel(capture.id);
  const stopChannel = useStopChannel(capture.id);
  const toast = useToast();

  const [showNewChannel, setShowNewChannel] = useState(false);
  const [newChannelFrequency, setNewChannelFrequency] = useState<number>(capture.centerHz);
  const [newChannelMode, setNewChannelMode] = useState<"wbfm" | "nbfm" | "am" | "ssb" | "raw" | "p25" | "dmr">("wbfm");
  const [newChannelSquelch, setNewChannelSquelch] = useState<number>(-60);
  const [newChannelAudioRate, setNewChannelAudioRate] = useState<number>(48000);
  const [copiedUrl, setCopiedUrl] = useState<string | null>(null);
  const [playingChannels, setPlayingChannels] = useState<Set<string>>(new Set());

  const audioContextRef = useRef<AudioContext | null>(null);
  const masterGainRef = useRef<GainNode | null>(null);
  const channelStreamsRef = useRef<Map<string, ChannelStream>>(new Map());

  // Stop audio for a specific channel
  const stopChannelAudio = useCallback((channelId: string) => {
    const stream = channelStreamsRef.current.get(channelId);
    if (stream) {
      stream.shouldPlay = false;
      stream.ws.close();
      stream.gainNode.disconnect();
      channelStreamsRef.current.delete(channelId);
    }
  }, []);

  // Stop all audio
  const stopAllAudio = useCallback(() => {
    channelStreamsRef.current.forEach((stream) => {
      stream.shouldPlay = false;
      stream.ws.close();
      stream.gainNode.disconnect();
    });
    channelStreamsRef.current.clear();
    setPlayingChannels(new Set());
  }, []);

  // Stop audio when capture changes
  useEffect(() => {
    setNewChannelFrequency(capture.centerHz);
    return () => {
      stopAllAudio();
    };
  }, [capture.id, stopAllAudio]);

  const handleCreateChannel = () => {
    const offsetHz = newChannelFrequency - capture.centerHz;

    createChannel.mutate({
      captureId: capture.id,
      request: {
        mode: newChannelMode,
        offsetHz,
        audioRate: newChannelAudioRate,
        squelchDb: newChannelSquelch,
      },
    }, {
      onSuccess: () => {
        setShowNewChannel(false);
        setNewChannelFrequency(capture.centerHz);
        setNewChannelSquelch(-60);
        setNewChannelAudioRate(48000);
        toast.success("Channel created successfully");
      },
      onError: (error: any) => {
        toast.error(error?.message || "Failed to create channel");
      },
    });
  };

  const copyToClipboard = (url: string) => {
    // Try modern Clipboard API first
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(url).then(() => {
        setCopiedUrl(url);
        setTimeout(() => setCopiedUrl(null), 2000);
        toast.success("URL copied to clipboard");
      }).catch(() => {
        // Fallback to legacy method
        fallbackCopyToClipboard(url);
      });
    } else {
      // Use fallback method directly
      fallbackCopyToClipboard(url);
    }
  };

  const fallbackCopyToClipboard = (url: string) => {
    try {
      // Create a temporary textarea element
      const textarea = document.createElement('textarea');
      textarea.value = url;
      textarea.style.position = 'fixed';
      textarea.style.left = '-999999px';
      textarea.style.top = '-999999px';
      document.body.appendChild(textarea);
      textarea.focus();
      textarea.select();

      // Try to copy using execCommand
      const successful = document.execCommand('copy');
      document.body.removeChild(textarea);

      if (successful) {
        setCopiedUrl(url);
        setTimeout(() => setCopiedUrl(null), 2000);
        toast.success("URL copied to clipboard");
      } else {
        toast.error("Failed to copy URL");
      }
    } catch (err) {
      console.error('Fallback copy failed:', err);
      toast.error("Failed to copy URL");
    }
  };

  // Initialize audio context and master gain
  // Safari/iOS requires AudioContext to be resumed after user interaction
  const initAudioContext = useCallback(() => {
    if (!audioContextRef.current) {
      // Use webkitAudioContext for older Safari versions
      const AudioContextClass = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      audioContextRef.current = new AudioContextClass({ sampleRate: 48000 });
      masterGainRef.current = audioContextRef.current.createGain();
      masterGainRef.current.gain.value = 1.0;
      masterGainRef.current.connect(audioContextRef.current.destination);
    }
    // Resume if suspended (required for Safari/iOS after user gesture)
    if (audioContextRef.current.state === "suspended") {
      audioContextRef.current.resume().catch((err) => {
        console.error("Failed to resume AudioContext:", err);
      });
    }
    return audioContextRef.current;
  }, []);

  // Play PCM audio for a single channel using WebSocket (Safari-compatible)
  const playPCMAudio = useCallback(async (channelId: string) => {
    try {
      const audioContext = initAudioContext();
      const masterGain = masterGainRef.current!;

      // Create a gain node for this channel (allows individual volume control)
      const channelGain = audioContext.createGain();
      // Reduce individual channel volume when mixing multiple channels
      const numChannels = channelStreamsRef.current.size + 1;
      channelGain.gain.value = 1.0 / Math.sqrt(numChannels);
      channelGain.connect(masterGain);

      // Use WebSocket for audio streaming - more reliable than fetch for Safari/iOS
      const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const wsUrl = `${wsProtocol}//${window.location.host}/api/v1/stream/channels/${channelId}?format=pcm16`;
      console.log("[ChannelManager] Connecting to WebSocket:", wsUrl);

      const ws = new WebSocket(wsUrl);
      ws.binaryType = "arraybuffer";

      const bufferSize = 4096;
      let pcmBuffer: number[] = [];
      let nextStartTime = audioContext.currentTime;

      // Store stream info before WebSocket opens
      const streamInfo: ChannelStream = {
        ws,
        gainNode: channelGain,
        shouldPlay: true,
        nextStartTime: audioContext.currentTime,
      };
      channelStreamsRef.current.set(channelId, streamInfo);

      // Update all channel gains for mixing
      const totalChannels = channelStreamsRef.current.size;
      const mixGain = 1.0 / Math.sqrt(totalChannels);
      channelStreamsRef.current.forEach(stream => {
        stream.gainNode.gain.value = mixGain;
      });

      ws.onopen = () => {
        console.log("[ChannelManager] WebSocket connected for channel:", channelId);
      };

      ws.onmessage = (event) => {
        if (!streamInfo.shouldPlay) return;

        const data = event.data;
        if (!(data instanceof ArrayBuffer)) {
          return; // Skip non-binary messages
        }

        const dataView = new DataView(data);
        const sampleCount = Math.floor(data.byteLength / 2);
        for (let i = 0; i < sampleCount; i++) {
          const sample = dataView.getInt16(i * 2, true) / 32768.0;
          pcmBuffer.push(sample);
        }

        // Process buffered audio
        while (pcmBuffer.length >= bufferSize && streamInfo.shouldPlay) {
          const chunk = pcmBuffer.splice(0, bufferSize);
          const audioBuffer = audioContext.createBuffer(1, chunk.length, 48000);
          const channelData = audioBuffer.getChannelData(0);

          for (let i = 0; i < chunk.length; i++) {
            channelData[i] = chunk[i];
          }

          const source = audioContext.createBufferSource();
          source.buffer = audioBuffer;
          source.connect(channelGain);

          const startTime = Math.max(nextStartTime, audioContext.currentTime);
          source.start(startTime);
          nextStartTime = startTime + audioBuffer.duration;
          streamInfo.nextStartTime = nextStartTime;
        }
      };

      ws.onerror = (error) => {
        console.error("[ChannelManager] WebSocket error:", error);
        stopChannelAudio(channelId);
        setPlayingChannels(prev => {
          const next = new Set(prev);
          next.delete(channelId);
          return next;
        });
      };

      ws.onclose = (event) => {
        console.log("[ChannelManager] WebSocket closed:", event.code, event.reason);
        if (streamInfo.shouldPlay) {
          // Unexpected close
          stopChannelAudio(channelId);
          setPlayingChannels(prev => {
            const next = new Set(prev);
            next.delete(channelId);
            return next;
          });
        }
      };

    } catch (error) {
      console.error("[ChannelManager] Failed to play audio:", error);
      setPlayingChannels(prev => {
        const next = new Set(prev);
        next.delete(channelId);
        return next;
      });
    }
  }, [initAudioContext, stopChannelAudio]);

  // Toggle playback for a single channel
  const togglePlay = async (channelId: string) => {
    if (playingChannels.has(channelId)) {
      // Stop this channel
      stopChannelAudio(channelId);
      stopChannel.mutate(channelId);
      setPlayingChannels(prev => {
        const next = new Set(prev);
        next.delete(channelId);
        return next;
      });
      return;
    }

    try {
      const channel = channels?.find(ch => ch.id === channelId);
      if (channel && channel.state !== "running") {
        await startChannel.mutateAsync(channelId);
      }
      setPlayingChannels(prev => new Set(prev).add(channelId));
      playPCMAudio(channelId);
    } catch (error: any) {
      console.error("Unable to start channel for playback:", error);
      toast.error(error?.message || "Failed to start channel");
    }
  };

  // Play all channels simultaneously
  const playAllChannels = async () => {
    if (!channels || channels.length === 0) return;

    try {
      // Start all channels that aren't running
      for (const channel of channels) {
        if (channel.state !== "running") {
          await startChannel.mutateAsync(channel.id);
        }
      }

      // Start audio for all channels
      const allChannelIds = new Set(channels.map(ch => ch.id));
      setPlayingChannels(allChannelIds);

      for (const channel of channels) {
        if (!channelStreamsRef.current.has(channel.id)) {
          playPCMAudio(channel.id);
        }
      }
    } catch (error: any) {
      console.error("Unable to start all channels:", error);
      toast.error(error?.message || "Failed to start all channels");
    }
  };

  // Stop all channels
  const stopAllChannels = async () => {
    stopAllAudio();
    if (channels) {
      for (const channel of channels) {
        stopChannel.mutate(channel.id);
      }
    }
  };

  return (
    <div className="card shadow-sm">
      <div className="card-header bg-body-tertiary">
        <Flex justify="between" align="center">
          <Flex align="center" gap={2}>
            <Radio size={20} />
            <h3 className="h6 mb-0">Channels</h3>
            {channels && channels.length > 0 && (
              <span className="badge bg-secondary">{channels.length}</span>
            )}
          </Flex>
          <Flex align="center" gap={2}>
            {channels && channels.length > 0 && (
              <Button
                use={playingChannels.size > 0 ? "warning" : "success"}
                size="sm"
                onClick={playingChannels.size > 0 ? stopAllChannels : playAllChannels}
                disabled={capture.state !== "running"}
                title={playingChannels.size > 0 ? "Stop all channels" : "Listen to all channels"}
              >
                <Flex align="center" gap={1}>
                  {playingChannels.size > 0 ? <VolumeX size={16} /> : <Volume2 size={16} />}
                  <span className="d-none d-sm-inline">
                    {playingChannels.size > 0 ? "Stop All" : "Listen All"}
                  </span>
                </Flex>
              </Button>
            )}
            <Button
              use="primary"
              size="sm"
              onClick={() => setShowNewChannel(!showNewChannel)}
              disabled={capture.state !== "running"}
            >
              <Plus size={16} />
            </Button>
          </Flex>
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
                <option value="nbfm">NBFM (Narrowband FM)</option>
                <option value="am">AM</option>
                <option value="ssb">SSB</option>
                <option value="raw">Raw IQ</option>
                <option value="p25">P25 (Trunked)</option>
                <option value="dmr">DMR (Trunked)</option>
              </select>
            </Flex>

            <Flex direction="column" gap={1}>
              <FrequencySelector
                label="Frequency"
                value={newChannelFrequency}
                min={capture.centerHz - (capture.sampleRate / 2)}
                max={capture.centerHz + (capture.sampleRate / 2)}
                step={1000}
                onChange={setNewChannelFrequency}
              />
              <small className="text-muted">
                Offset: {((newChannelFrequency - capture.centerHz) / 1000).toFixed(0)} kHz
              </small>
            </Flex>

            <Slider
              label="Squelch"
              value={newChannelSquelch}
              min={-80}
              max={0}
              step={1}
              coarseStep={10}
              unit="dB"
              formatValue={(val) => `${val.toFixed(0)} dB`}
              onChange={setNewChannelSquelch}
              info="Signal strength threshold. Lower values (more negative) allow weaker signals."
            />

            <Flex direction="column" gap={1}>
              <label className="form-label small mb-1">Audio Rate</label>
              <select
                className="form-select form-select-sm"
                value={newChannelAudioRate}
                onChange={(e) => setNewChannelAudioRate(parseInt(e.target.value))}
              >
                <option value={8000}>8 kHz</option>
                <option value={16000}>16 kHz</option>
                <option value={24000}>24 kHz</option>
                <option value={48000}>48 kHz (CD quality)</option>
              </select>
            </Flex>

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

        {/* Loading Skeleton */}
        {isLoading && (
          <div className="row g-3">
            {[1, 2].map((i) => (
              <div key={i} className="col-12 col-xl-6">
                <SkeletonChannelCard />
              </div>
            ))}
          </div>
        )}

        {!isLoading && (!channels || channels.length === 0) && (
          <div className="text-muted small text-center py-3">
            No channels. Click + to create one.
          </div>
        )}

        {/* Channel List */}
        {!isLoading && channels && channels.length > 0 && (
          <div className="d-flex flex-column gap-2">
            {channels.map((channel) => (
              <CompactChannelCard
                key={channel.id}
                channel={channel}
                capture={capture}
                isPlaying={playingChannels.has(channel.id)}
                onTogglePlay={() => togglePlay(channel.id)}
                onCopyUrl={copyToClipboard}
                copiedUrl={copiedUrl}
              />
            ))}
          </div>
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
