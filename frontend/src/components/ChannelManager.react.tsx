import { useState, useRef, useEffect } from "react";
import { Radio, Plus } from "lucide-react";
import type { Capture } from "../types";
import {
  useChannels,
  useCreateChannel,
  useStartChannel,
  useStopChannel,
} from "../hooks/useChannels";
import Button from "./primitives/Button.react";
import Flex from "./primitives/Flex.react";
import Slider from "./primitives/Slider.react";
import FrequencySelector from "./primitives/FrequencySelector.react";
import Spinner from "./primitives/Spinner.react";
import { CompactChannelCard } from "./CompactChannelCard.react";

interface ChannelManagerProps {
  capture: Capture;
}

export const ChannelManager = ({ capture }: ChannelManagerProps) => {
  const { data: channels, isLoading } = useChannels(capture.id);
  const createChannel = useCreateChannel();
  const startChannel = useStartChannel(capture.id);
  const stopChannel = useStopChannel(capture.id);

  const [showNewChannel, setShowNewChannel] = useState(false);
  const [newChannelFrequency, setNewChannelFrequency] = useState<number>(capture.centerHz);
  const [newChannelMode, setNewChannelMode] = useState<"wbfm" | "nbfm" | "am" | "ssb" | "raw" | "p25" | "dmr">("wbfm");
  const [newChannelSquelch, setNewChannelSquelch] = useState<number>(-60);
  const [newChannelAudioRate, setNewChannelAudioRate] = useState<number>(48000);
  const [copiedUrl, setCopiedUrl] = useState<string | null>(null);
  const [playingChannel, setPlayingChannel] = useState<string | null>(null);

  const audioContextRef = useRef<AudioContext | null>(null);
  const streamReaderRef = useRef<ReadableStreamDefaultReader<Uint8Array> | null>(null);
  const shouldPlayRef = useRef<boolean>(false);
  const nextStartTimeRef = useRef<number>(0);

  const stopAudio = () => {
    shouldPlayRef.current = false;

    if (streamReaderRef.current) {
      streamReaderRef.current.cancel();
      streamReaderRef.current = null;
    }
  };

  // Stop audio when capture changes
  useEffect(() => {
    setNewChannelFrequency(capture.centerHz);
    return () => {
      stopAudio();
    };
  }, [capture.id]);

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
      },
    });
  };

  const copyToClipboard = (url: string) => {
    navigator.clipboard.writeText(url).then(() => {
      setCopiedUrl(url);
      setTimeout(() => setCopiedUrl(null), 2000);
    });
  };

  const playPCMAudio = async (channelId: string) => {
    try {
      if (!audioContextRef.current) {
        audioContextRef.current = new AudioContext({ sampleRate: 48000 });
      }

      const audioContext = audioContextRef.current;
      shouldPlayRef.current = true;
      nextStartTimeRef.current = audioContext.currentTime;

      const streamUrl = `${window.location.origin}/api/v1/stream/channels/${channelId}.pcm`;
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

          const dataView = new DataView(value.buffer, value.byteOffset, value.byteLength);
          const sampleCount = Math.floor(value.length / 2);
          for (let i = 0; i < sampleCount; i++) {
            const sample = dataView.getInt16(i * 2, true) / 32768.0;
            pcmBuffer.push(sample);
          }

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

  const togglePlay = async (channelId: string) => {
    if (playingChannel === channelId) {
      stopAudio();
      setPlayingChannel(null);
      stopChannel.mutate(channelId);
      return;
    }

    if (playingChannel) {
      stopAudio();
      setPlayingChannel(null);
    }

    try {
      const channel = channels?.find(ch => ch.id === channelId);
      if (channel && channel.state !== "running") {
        await startChannel.mutateAsync(channelId);
      }
      setPlayingChannel(channelId);
      playPCMAudio(channelId);
    } catch (error) {
      console.error("Unable to start channel for playback:", error);
      setPlayingChannel(null);
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

        {!isLoading && (!channels || channels.length === 0) && (
          <div className="text-muted small text-center py-3">
            No channels. Click + to create one.
          </div>
        )}

        {/* Grid Layout for Channels */}
        {channels && channels.length > 0 && (
          <div className="row g-3">
            {channels.map((channel) => (
              <div key={channel.id} className="col-12 col-sm-6 col-xl-4 col-xxl-3">
                <CompactChannelCard
                  channel={channel}
                  capture={capture}
                  isPlaying={playingChannel === channel.id}
                  onTogglePlay={() => togglePlay(channel.id)}
                  onCopyUrl={copyToClipboard}
                  copiedUrl={copiedUrl}
                />
              </div>
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
