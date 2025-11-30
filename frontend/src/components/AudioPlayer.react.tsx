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

type AudioFormat = "pcm" | "mp3" | "opus" | "aac";

// Detect Safari/iOS for special audio handling
const isSafari = /^((?!chrome|android).)*safari/i.test(navigator.userAgent) ||
  /iPad|iPhone|iPod/.test(navigator.userAgent);

export const AudioPlayer = ({ channelId, captureState }: AudioPlayerProps) => {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [volume, setVolume] = useState(0.7);
  const [isMuted, setIsMuted] = useState(false);
  // Safari works best with PCM via AudioContext
  const [audioFormat, setAudioFormat] = useState<AudioFormat>(isSafari ? "pcm" : "mp3");
  const [connectionState, setConnectionState] = useState<ConnectionState>("connected");
  const [retryCount, setRetryCount] = useState(0);
  const [userPaused, setUserPaused] = useState(false);
  const retryTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const lastPlayAttemptRef = useRef<number>(0);

  // AudioContext for PCM streaming (Safari-compatible approach)
  const audioContextRef = useRef<AudioContext | null>(null);
  const streamReaderRef = useRef<ReadableStreamDefaultReader<Uint8Array> | null>(null);
  const gainNodeRef = useRef<GainNode | null>(null);
  const shouldPlayRef = useRef<boolean>(false);
  const nextStartTimeRef = useRef<number>(0);

  const streamUrl = channelId ? `/api/v1/stream/channels/${channelId}.${audioFormat}?t=${Date.now()}` : null;

  const clearRetryTimeout = useCallback(() => {
    if (retryTimeoutRef.current) {
      clearTimeout(retryTimeoutRef.current);
      retryTimeoutRef.current = null;
    }
  }, []);

  // Initialize AudioContext (Safari-compatible with webkitAudioContext fallback)
  const initAudioContext = useCallback(() => {
    if (!audioContextRef.current) {
      const AudioContextClass = window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      audioContextRef.current = new AudioContextClass({ sampleRate: 48000 });
      gainNodeRef.current = audioContextRef.current.createGain();
      gainNodeRef.current.connect(audioContextRef.current.destination);
    }
    // Resume if suspended (required for Safari/iOS after user gesture)
    if (audioContextRef.current.state === "suspended") {
      audioContextRef.current.resume().catch((err) => {
        console.error("Failed to resume AudioContext:", err);
      });
    }
    return audioContextRef.current;
  }, []);

  // Stop PCM streaming (handles both WebSocket and ReadableStream)
  const stopPCMStream = useCallback(() => {
    shouldPlayRef.current = false;
    if (streamReaderRef.current) {
      // Check if it's a WebSocket wrapper or a ReadableStreamReader
      const ref = streamReaderRef.current as unknown as { ws?: WebSocket; closed?: boolean; cancel?: () => Promise<void> };
      if (ref.ws) {
        // WebSocket cleanup
        ref.closed = true;
        ref.ws.close();
      } else if (ref.cancel) {
        // ReadableStreamReader cleanup
        ref.cancel().catch(() => {});
      }
      streamReaderRef.current = null;
    }
  }, []);

  // Play PCM audio using WebSocket + AudioContext (most reliable for Safari)
  const playPCMStream = useCallback(async () => {
    if (!channelId) return;

    try {
      console.log("[AudioPlayer] Starting PCM stream for channel:", channelId);
      const audioContext = initAudioContext();
      console.log("[AudioPlayer] AudioContext state:", audioContext.state, "sampleRate:", audioContext.sampleRate);

      const gainNode = gainNodeRef.current!;
      gainNode.gain.value = isMuted ? 0 : volume;

      // Use WebSocket for Safari - more reliable than fetch streaming
      const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const wsUrl = `${wsProtocol}//${window.location.host}/api/v1/stream/channels/${channelId}?format=pcm16`;
      console.log("[AudioPlayer] Connecting to WebSocket:", wsUrl);

      const ws = new WebSocket(wsUrl);
      ws.binaryType = "arraybuffer";

      // Store WebSocket reference for cleanup
      const wsRef = { ws, closed: false };
      streamReaderRef.current = wsRef as unknown as ReadableStreamDefaultReader<Uint8Array>;
      shouldPlayRef.current = true;
      nextStartTimeRef.current = audioContext.currentTime;

      const bufferSize = 4096;
      let pcmBuffer: number[] = [];

      ws.onopen = () => {
        console.log("[AudioPlayer] WebSocket connected");
        setIsPlaying(true);
        setConnectionState("connected");
        setRetryCount(0);
      };

      ws.onmessage = (event) => {
        if (!shouldPlayRef.current || wsRef.closed) return;

        const data = event.data;
        if (!(data instanceof ArrayBuffer)) {
          console.log("[AudioPlayer] Received non-binary message:", data);
          return;
        }

        const dataView = new DataView(data);
        const sampleCount = Math.floor(data.byteLength / 2);
        for (let i = 0; i < sampleCount; i++) {
          const sample = dataView.getInt16(i * 2, true) / 32768.0;
          pcmBuffer.push(sample);
        }

        // Process buffered audio
        while (pcmBuffer.length >= bufferSize && shouldPlayRef.current) {
          const chunk = pcmBuffer.splice(0, bufferSize);
          const audioBuffer = audioContext.createBuffer(1, chunk.length, 48000);
          const channelData = audioBuffer.getChannelData(0);

          for (let i = 0; i < chunk.length; i++) {
            channelData[i] = chunk[i];
          }

          const source = audioContext.createBufferSource();
          source.buffer = audioBuffer;
          source.connect(gainNode);

          const startTime = Math.max(nextStartTimeRef.current, audioContext.currentTime);
          source.start(startTime);
          nextStartTimeRef.current = startTime + audioBuffer.duration;
        }
      };

      ws.onerror = (error) => {
        console.error("[AudioPlayer] WebSocket error:", error);
        if (!wsRef.closed) {
          setConnectionState("failed");
          setIsPlaying(false);
        }
      };

      ws.onclose = (event) => {
        console.log("[AudioPlayer] WebSocket closed:", event.code, event.reason);
        wsRef.closed = true;
        if (shouldPlayRef.current) {
          // Unexpected close - try to reconnect
          console.log("[AudioPlayer] Unexpected WebSocket close, will show failed state");
          setConnectionState("failed");
        }
        setIsPlaying(false);
      };

    } catch (error) {
      console.error("[AudioPlayer] Failed to start PCM stream:", error);
      setConnectionState("failed");
      setIsPlaying(false);
    }
  }, [channelId, volume, isMuted, initAudioContext]);

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
      const newUrl = `/api/v1/stream/channels/${channelId}.${audioFormat}?t=${Date.now()}`;
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
  }, [streamUrl, channelId, audioFormat, retryCount, userPaused, clearRetryTimeout]);

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

  // Update volume for both HTMLAudioElement and AudioContext gain
  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.volume = isMuted ? 0 : volume;
    }
    if (gainNodeRef.current) {
      gainNodeRef.current.gain.value = isMuted ? 0 : volume;
    }
  }, [volume, isMuted]);

  // Cleanup PCM stream on unmount or channel change
  useEffect(() => {
    return () => {
      stopPCMStream();
    };
  }, [channelId, stopPCMStream]);

  const handlePlayPause = () => {
    if (isPlaying || connectionState === "reconnecting") {
      // User explicitly paused
      setUserPaused(true);
      setConnectionState("connected");
      setRetryCount(0);
      clearRetryTimeout();

      // Stop PCM stream if active
      stopPCMStream();

      // Stop HTMLAudioElement if active
      const audio = audioRef.current;
      if (audio) {
        audio.pause();
        audio.src = "";
      }
      setIsPlaying(false);
    } else {
      // User explicitly played
      setUserPaused(false);
      setConnectionState("connected");
      setRetryCount(0);

      if (audioFormat === "pcm") {
        // Use AudioContext-based streaming for PCM (Safari-compatible)
        playPCMStream();
      } else {
        // Use HTMLAudioElement for compressed formats
        const audio = audioRef.current;
        if (!audio) return;

        // Explicitly abort any existing stream before loading new one
        audio.src = "";

        // Reload the stream with a new timestamp to avoid caching
        const newUrl = `/api/v1/stream/channels/${channelId}.${audioFormat}?t=${Date.now()}`;
        audio.src = newUrl;
        audio.load();

        audio.play().catch((error) => {
          console.error("Failed to play audio:", error);
          setConnectionState("failed");
        });
      }
    }
  };

  const handleManualRetry = () => {
    setRetryCount(0);
    setUserPaused(false);
    setConnectionState("connected");
    clearRetryTimeout();

    // Stop any existing streams
    stopPCMStream();
    const audio = audioRef.current;
    if (audio) {
      audio.src = "";
    }

    if (audioFormat === "pcm") {
      // Use AudioContext-based streaming for PCM (Safari-compatible)
      playPCMStream();
    } else {
      // Use HTMLAudioElement for compressed formats
      if (!audio) return;

      const newUrl = `/api/v1/stream/channels/${channelId}.${audioFormat}?t=${Date.now()}`;
      audio.src = newUrl;
      audio.load();

      audio.play().catch((error) => {
        console.error("Failed to play audio:", error);
        setConnectionState("failed");
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
            <audio ref={audioRef} src={streamUrl} preload="none" playsInline />
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

          <Flex direction="column" gap={1}>
            <label className="form-label small mb-1">Audio Quality</label>
            <select
              className="form-select form-select-sm"
              value={audioFormat}
              onChange={(e) => setAudioFormat(e.target.value as AudioFormat)}
              disabled={isDisabled || isPlaying}
            >
              <option value="pcm">PCM (Uncompressed){isSafari ? " - Recommended" : ""}</option>
              <option value="mp3">MP3 (Compressed, Low Latency)</option>
              <option value="opus">Opus (Best Quality, Low Bandwidth)</option>
              <option value="aac">AAC (Balanced Quality)</option>
            </select>
            <small className="text-muted">
              {isSafari
                ? "Safari detected - PCM is recommended for best compatibility."
                : "Change quality when stopped. Format affects bandwidth and latency."}
            </small>
          </Flex>

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
