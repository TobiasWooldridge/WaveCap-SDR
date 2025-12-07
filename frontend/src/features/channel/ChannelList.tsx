import { useState } from "react";
import { Plus, VolumeX, Volume2 } from "lucide-react";
import type { Capture } from "../../types";
import { useChannels, useCreateChannel, useStartChannel } from "../../hooks/useChannels";
import { useAudio } from "../../hooks/useAudio";
import { useToast } from "../../hooks/useToast";
import { ChannelCard } from "./ChannelCard";
import Button from "../../components/primitives/Button.react";
import Flex from "../../components/primitives/Flex.react";
import Slider from "../../components/primitives/Slider.react";
import FrequencySelector from "../../components/primitives/FrequencySelector.react";
import Spinner from "../../components/primitives/Spinner.react";
import { SkeletonChannelCard } from "../../components/primitives/Skeleton.react";

interface ChannelListProps {
  capture: Capture;
}

export function ChannelList({ capture }: ChannelListProps) {
  const { data: channels, isLoading } = useChannels(capture.id);
  const createChannel = useCreateChannel();
  const startChannel = useStartChannel(capture.id);
  const { playingChannels, stopAll, masterVolume, setMasterVolume } = useAudio();
  const toast = useToast();

  const [showNewChannel, setShowNewChannel] = useState(false);
  const [newFrequency, setNewFrequency] = useState(capture.centerHz);
  const [newMode, setNewMode] = useState<"wbfm" | "nbfm" | "am" | "ssb">("wbfm");
  const [newSquelch, setNewSquelch] = useState(-60);

  const hasPlayingChannels = playingChannels.size > 0;
  const channelCount = channels?.length ?? 0;

  const handleCreateChannel = async () => {
    try {
      const result = await createChannel.mutateAsync({
        captureId: capture.id,
        request: {
          mode: newMode,
          offsetHz: newFrequency - capture.centerHz,
          audioRate: 48000,
          squelchDb: newSquelch,
        },
      });

      // Auto-start the new channel
      await startChannel.mutateAsync(result.id);
      toast.success("Channel created");
      setShowNewChannel(false);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Failed to create channel");
    }
  };

  if (isLoading) {
    return (
      <Flex direction="column" gap={2}>
        <SkeletonChannelCard />
        <SkeletonChannelCard />
      </Flex>
    );
  }

  return (
    <Flex direction="column" gap={2} style={{ height: "100%", overflowY: "auto" }}>
      {/* Header with controls */}
      <div className="d-flex align-items-center gap-2 p-2 bg-light rounded border">
        <span className="fw-semibold small">
          Channels ({channelCount})
        </span>

        {hasPlayingChannels && (
          <>
            <div style={{ width: "100px" }}>
              <Slider
                label="Volume"
                value={masterVolume}
                min={0}
                max={1}
                step={0.01}
                onChange={setMasterVolume}
              />
            </div>
            <Button size="sm" use="warning" onClick={stopAll}>
              <VolumeX size={14} className="me-1" />
              Stop All
            </Button>
          </>
        )}

        <Button
          size="sm"
          use="primary"
          onClick={() => setShowNewChannel(!showNewChannel)}
          className="ms-auto"
        >
          <Plus size={14} className="me-1" />
          Add
        </Button>
      </div>

      {/* New Channel Form */}
      {showNewChannel && (
        <div className="card shadow-sm">
          <div className="card-header bg-body-tertiary py-1 px-2">
            <small className="fw-semibold">New Channel</small>
          </div>
          <div className="card-body p-2">
            <Flex direction="column" gap={2}>
              <FrequencySelector
                label="Frequency"
                value={newFrequency}
                min={capture.centerHz - capture.sampleRate / 2}
                max={capture.centerHz + capture.sampleRate / 2}
                step={1000}
                onChange={setNewFrequency}
              />

              <Flex direction="column" gap={1}>
                <label className="form-label small mb-0">Mode</label>
                <select
                  className="form-select form-select-sm"
                  value={newMode}
                  onChange={(e) => setNewMode(e.target.value as typeof newMode)}
                >
                  <option value="wbfm">WBFM</option>
                  <option value="nbfm">NBFM</option>
                  <option value="am">AM</option>
                  <option value="ssb">SSB</option>
                </select>
              </Flex>

              <Slider
                label="Squelch"
                value={newSquelch}
                min={-80}
                max={0}
                step={1}
                unit="dB"
                onChange={setNewSquelch}
              />

              <Flex gap={2}>
                <Button
                  use="primary"
                  size="sm"
                  onClick={handleCreateChannel}
                  disabled={createChannel.isPending}
                >
                  {createChannel.isPending ? <Spinner size="sm" /> : "Create"}
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
        </div>
      )}

      {/* Channel List */}
      {channels?.length === 0 && !showNewChannel && (
        <div className="text-center text-muted py-4">
          <Volume2 size={32} className="mb-2 opacity-50" />
          <p className="small mb-0">No channels yet</p>
          <p className="small text-muted">Click "Add" to create a channel</p>
        </div>
      )}

      {channels?.map((channel) => (
        <ChannelCard key={channel.id} channel={channel} capture={capture} />
      ))}
    </Flex>
  );
}
