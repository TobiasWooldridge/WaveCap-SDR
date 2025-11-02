import { useEffect, useRef, useState } from "react";
import { Play, Pause, Volume2, VolumeX } from "lucide-react";
import Button from "./primitives/Button.react";
import Flex from "./primitives/Flex.react";
import Slider from "./primitives/Slider.react";

interface AudioPlayerProps {
  channelId: string | null;
  captureState: string;
}

export const AudioPlayer = ({ channelId, captureState }: AudioPlayerProps) => {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [volume, setVolume] = useState(0.7);
  const [isMuted, setIsMuted] = useState(false);

  const streamUrl = channelId ? `/api/v1/stream/channels/${channelId}.pcm` : null;

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio || !streamUrl) return;

    const handlePlay = () => setIsPlaying(true);
    const handlePause = () => setIsPlaying(false);
    const handleError = (e: Event) => {
      console.error("Audio error:", e);
      setIsPlaying(false);
    };

    audio.addEventListener("play", handlePlay);
    audio.addEventListener("pause", handlePause);
    audio.addEventListener("error", handleError);

    return () => {
      audio.removeEventListener("play", handlePlay);
      audio.removeEventListener("pause", handlePause);
      audio.removeEventListener("error", handleError);
    };
  }, [streamUrl]);

  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.volume = isMuted ? 0 : volume;
    }
  }, [volume, isMuted]);

  const handlePlayPause = () => {
    const audio = audioRef.current;
    if (!audio) return;

    if (isPlaying) {
      audio.pause();
    } else {
      audio.play().catch((error) => {
        console.error("Failed to play audio:", error);
      });
    }
  };

  const toggleMute = () => {
    setIsMuted(!isMuted);
  };

  const isDisabled = !streamUrl || captureState !== "running";

  return (
    <div className="card shadow-sm">
      <div className="card-header bg-body-tertiary">
        <h3 className="h6 mb-0">Audio Player</h3>
      </div>
      <div className="card-body">
        <Flex direction="column" gap={3}>
          {streamUrl && (
            <audio ref={audioRef} src={streamUrl} preload="none" />
          )}

          <Flex gap={2} align="center">
            <Button
              use={isPlaying ? "warning" : "success"}
              size="lg"
              onClick={handlePlayPause}
              disabled={isDisabled}
              startContent={isPlaying ? <Pause size={20} /> : <Play size={20} />}
            >
              {isPlaying ? "Pause" : "Play"}
            </Button>

            <Button
              use="secondary"
              appearance="outline"
              onClick={toggleMute}
              disabled={isDisabled}
              startContent={isMuted ? <VolumeX size={20} /> : <Volume2 size={20} />}
              isCondensed
            >
              {isMuted ? "Unmute" : "Mute"}
            </Button>
          </Flex>

          <Slider
            label="Volume"
            value={volume}
            min={0}
            max={1}
            step={0.01}
            onChange={setVolume}
            disabled={isDisabled}
            showMinMax={false}
            formatValue={(v) => `${Math.round(v * 100)}%`}
          />

          {!channelId && (
            <div className="alert alert-info mb-0">
              <small>No audio channel available. Select a capture with channels.</small>
            </div>
          )}

          {captureState === "stopped" && (
            <div className="alert alert-warning mb-0">
              <small>Capture is stopped. Start the capture to hear audio.</small>
            </div>
          )}

          {captureState === "failed" && (
            <div className="alert alert-danger mb-0">
              <small>Capture failed. Check the error message and try restarting.</small>
            </div>
          )}
        </Flex>
      </div>
    </div>
  );
};
