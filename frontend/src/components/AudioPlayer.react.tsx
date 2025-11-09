import { useEffect, useRef, useState, useCallback } from "react";
import { Play, Pause, Volume2, VolumeX, AlertCircle, RefreshCw } from "lucide-react";
import Button from "./primitives/Button.react";
import Flex from "./primitives/Flex.react";
import Slider from "./primitives/Slider.react";

interface AudioPlayerProps {
  channelId: string | null;
  captureState: string;
}

type ConnectionState = "connected" | "reconnecting" | "failed";

const MAX_RETRY_DELAY = 30000; // 30 seconds
const INITIAL_RETRY_DELAY = 1000; // 1 second
const MAX_RETRIES = 10;

export const AudioPlayer = ({ channelId, captureState }: AudioPlayerProps) => {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [volume, setVolume] = useState(0.7);
  const [isMuted, setIsMuted] = useState(false);
  const [connectionState, setConnectionState] = useState<ConnectionState>("connected");
  const [retryCount, setRetryCount] = useState(0);
  const [userPaused, setUserPaused] = useState(false);
  const retryTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const lastPlayAttemptRef = useRef<number>(0);

  const streamUrl = channelId ? `/api/v1/stream/channels/${channelId}.pcm?t=${Date.now()}` : null;

  const clearRetryTimeout = useCallback(() => {
    if (retryTimeoutRef.current) {
      clearTimeout(retryTimeoutRef.current);
      retryTimeoutRef.current = null;
    }
  }, []);

  const attemptReconnect = useCallback(() => {
    if (!audioRef.current || !streamUrl || userPaused) return;

    clearRetryTimeout();

    const delay = Math.min(
      INITIAL_RETRY_DELAY * Math.pow(2, retryCount),
      MAX_RETRY_DELAY
    );

    console.log(`Attempting reconnection in ${delay}ms (attempt ${retryCount + 1}/${MAX_RETRIES})`);
    setConnectionState("reconnecting");

    retryTimeoutRef.current = setTimeout(() => {
      const audio = audioRef.current;
      if (!audio || userPaused) return;

      // Explicitly abort existing stream before loading new one
      audio.src = "";

      // Update stream URL with new timestamp to avoid caching
      const newUrl = `/api/v1/stream/channels/${channelId}.pcm?t=${Date.now()}`;
      audio.src = newUrl;
      audio.load();

      lastPlayAttemptRef.current = Date.now();
      audio.play()
        .then(() => {
          console.log("Reconnection successful");
          setConnectionState("connected");
          setRetryCount(0);
        })
        .catch((error) => {
          console.error("Reconnection failed:", error);
          if (retryCount < MAX_RETRIES - 1) {
            setRetryCount(retryCount + 1);
            attemptReconnect();
          } else {
            console.error("Max retries reached, giving up");
            setConnectionState("failed");
            setIsPlaying(false);
          }
        });
    }, delay);
  }, [streamUrl, channelId, retryCount, userPaused, clearRetryTimeout]);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio || !streamUrl) return;

    const handlePlay = () => {
      setIsPlaying(true);
      setConnectionState("connected");
      setRetryCount(0);
    };

    const handlePause = () => {
      setIsPlaying(false);
      // Only clear connection state if user manually paused
      if (userPaused) {
        setConnectionState("connected");
        clearRetryTimeout();
      }
    };

    const handleError = (e: Event) => {
      const error = (e.target as HTMLAudioElement)?.error;
      console.error("Audio error:", error?.code, error?.message);

      // Only attempt reconnection if we were playing and user didn't pause
      if (isPlaying && !userPaused && retryCount < MAX_RETRIES) {
        console.log("Stream error detected, attempting reconnection...");
        attemptReconnect();
      } else {
        setIsPlaying(false);
        if (retryCount >= MAX_RETRIES) {
          setConnectionState("failed");
        }
      }
    };

    const handleEnded = () => {
      console.log("Stream ended unexpectedly");
      // Stream shouldn't end naturally - this indicates a connection issue
      if (isPlaying && !userPaused && retryCount < MAX_RETRIES) {
        console.log("Unexpected stream end, attempting reconnection...");
        attemptReconnect();
      } else {
        setIsPlaying(false);
      }
    };

    audio.addEventListener("play", handlePlay);
    audio.addEventListener("pause", handlePause);
    audio.addEventListener("error", handleError);
    audio.addEventListener("ended", handleEnded);

    return () => {
      audio.removeEventListener("play", handlePlay);
      audio.removeEventListener("pause", handlePause);
      audio.removeEventListener("error", handleError);
      audio.removeEventListener("ended", handleEnded);
      clearRetryTimeout();
    };
  }, [streamUrl, isPlaying, userPaused, retryCount, attemptReconnect, clearRetryTimeout]);

  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.volume = isMuted ? 0 : volume;
    }
  }, [volume, isMuted]);

  const handlePlayPause = () => {
    const audio = audioRef.current;
    if (!audio) return;

    if (isPlaying || connectionState === "reconnecting") {
      // User explicitly paused
      setUserPaused(true);
      setConnectionState("connected");
      setRetryCount(0);
      clearRetryTimeout();
      audio.pause();
      // Explicitly close the stream
      audio.src = "";
    } else {
      // User explicitly played
      setUserPaused(false);
      setConnectionState("connected");
      setRetryCount(0);

      // Explicitly abort any existing stream before loading new one
      audio.src = "";

      // Reload the stream with a new timestamp to avoid caching
      const newUrl = `/api/v1/stream/channels/${channelId}.pcm?t=${Date.now()}`;
      audio.src = newUrl;
      audio.load();

      audio.play().catch((error) => {
        console.error("Failed to play audio:", error);
        setConnectionState("failed");
      });
    }
  };

  const handleManualRetry = () => {
    setRetryCount(0);
    setUserPaused(false);
    setConnectionState("connected");
    clearRetryTimeout();

    const audio = audioRef.current;
    if (!audio) return;

    // Explicitly abort existing stream before loading new one
    audio.src = "";

    const newUrl = `/api/v1/stream/channels/${channelId}.pcm?t=${Date.now()}`;
    audio.src = newUrl;
    audio.load();

    audio.play().catch((error) => {
      console.error("Failed to play audio:", error);
      setConnectionState("failed");
    });
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

          {connectionState === "reconnecting" && (
            <div className="alert alert-warning mb-0 d-flex align-items-center gap-2">
              <RefreshCw size={16} className="spin" />
              <small>Connection lost. Reconnecting... (attempt {retryCount + 1}/{MAX_RETRIES})</small>
            </div>
          )}

          {connectionState === "failed" && (
            <div className="alert alert-danger mb-0">
              <Flex direction="row" align="center" gap={2} justify="between">
                <Flex direction="row" align="center" gap={2}>
                  <AlertCircle size={16} />
                  <small>Connection failed after {MAX_RETRIES} attempts.</small>
                </Flex>
                <Button
                  use="danger"
                  size="sm"
                  onClick={handleManualRetry}
                  startContent={<RefreshCw size={16} />}
                >
                  Retry
                </Button>
              </Flex>
            </div>
          )}
        </Flex>
      </div>
    </div>
  );
};
