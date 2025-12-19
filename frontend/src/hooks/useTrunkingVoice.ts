/**
 * useTrunkingVoice - Hook for subscribing to trunking voice streams via WebSocket.
 *
 * Provides real-time audio streaming with metadata from P25 trunked radio voice channels.
 * Supports subscribing to all voice streams (multiplexed) or a single stream.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import {
  VoiceStream,
  VoiceStreamMessage,
  VoiceAudioMessage,
  RadioLocation,
} from "../types/trunking";

const API_BASE = "/api/v1/trunking";

interface UseVoiceStreamsOptions {
  systemId: string;
  enabled?: boolean;
  onAudio?: (message: VoiceAudioMessage) => void;
  onStreamEnded?: (streamId: string) => void;
}

interface UseVoiceStreamsResult {
  streams: VoiceStream[];
  isConnected: boolean;
  error: string | null;
  refetch: () => Promise<void>;
}

/**
 * Hook for subscribing to all voice streams from a trunking system.
 *
 * Connects via WebSocket to receive multiplexed audio from all active voice channels.
 */
export function useVoiceStreams({
  systemId,
  enabled = true,
  onAudio,
  onStreamEnded,
}: UseVoiceStreamsOptions): UseVoiceStreamsResult {
  const [streams, setStreams] = useState<VoiceStream[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout>>();

  // Fetch current voice streams via REST
  const fetchStreams = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/systems/${systemId}/voice-streams`);
      if (response.ok) {
        const data = await response.json();
        setStreams(data);
      }
    } catch (e) {
      console.error("Failed to fetch voice streams:", e);
    }
  }, [systemId]);

  // Connect to WebSocket
  const connect = useCallback(() => {
    if (!enabled || !systemId) return;

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}${API_BASE}/stream/${systemId}/voice`;

    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setIsConnected(true);
        setError(null);
        fetchStreams();
      };

      ws.onmessage = (event) => {
        try {
          // Handle text messages (JSON)
          if (typeof event.data === "string") {
            const message: VoiceStreamMessage = JSON.parse(event.data);

            if (message.type === "audio") {
              onAudio?.(message);
            } else if (message.type === "ended") {
              onStreamEnded?.(message.streamId);
              fetchStreams();
            }
          } else if (event.data instanceof Blob) {
            // Handle binary messages (raw audio)
            event.data.text().then((text) => {
              try {
                const message: VoiceStreamMessage = JSON.parse(text);
                if (message.type === "audio") {
                  onAudio?.(message);
                } else if (message.type === "ended") {
                  onStreamEnded?.(message.streamId);
                  fetchStreams();
                }
              } catch (e) {
                console.error("Failed to parse binary message:", e);
              }
            });
          }
        } catch (e) {
          console.error("Failed to parse voice stream message:", e);
        }
      };

      ws.onclose = () => {
        setIsConnected(false);
        wsRef.current = null;

        // Reconnect after delay
        if (enabled) {
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, 3000);
        }
      };

      ws.onerror = (e) => {
        console.error("Voice stream WebSocket error:", e);
        setError("WebSocket connection error");
      };
    } catch (e) {
      console.error("Failed to create WebSocket:", e);
      setError("Failed to connect to voice stream");
    }
  }, [systemId, enabled, onAudio, onStreamEnded, fetchStreams]);

  // Cleanup
  useEffect(() => {
    connect();

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [connect]);

  return {
    streams,
    isConnected,
    error,
    refetch: fetchStreams,
  };
}

interface UseSingleVoiceStreamOptions {
  systemId: string;
  streamId: string;
  enabled?: boolean;
  onAudio?: (message: VoiceAudioMessage) => void;
  onStreamEnded?: () => void;
}

interface UseSingleVoiceStreamResult {
  stream: VoiceStream | null;
  isConnected: boolean;
  error: string | null;
}

/**
 * Hook for subscribing to a single voice stream.
 *
 * Connects via WebSocket to receive audio from one specific voice channel.
 */
export function useSingleVoiceStream({
  systemId,
  streamId,
  enabled = true,
  onAudio,
  onStreamEnded,
}: UseSingleVoiceStreamOptions): UseSingleVoiceStreamResult {
  const [stream, setStream] = useState<VoiceStream | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!enabled || !systemId || !streamId) return;

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}${API_BASE}/stream/${systemId}/voice/${streamId}`;

    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setIsConnected(true);
        setError(null);
      };

      ws.onmessage = (event) => {
        try {
          const text = typeof event.data === "string" ? event.data : null;
          if (text) {
            const message: VoiceStreamMessage = JSON.parse(text);

            if (message.type === "audio") {
              onAudio?.(message);
              // Update stream info from audio message
              setStream((prev) => ({
                ...prev,
                id: message.streamId,
                systemId: message.systemId,
                callId: message.callId,
                recorderId: message.recorderId,
                state: "active",
                talkgroupId: message.talkgroupId,
                talkgroupName: message.talkgroupName,
                sourceId: message.sourceId,
                sourceLocation: message.sourceLocation,
                encrypted: message.encrypted,
                startTime: prev?.startTime ?? Date.now() / 1000,
                durationSeconds: prev?.durationSeconds ?? 0,
                silenceSeconds: 0,
                audioFrameCount: message.frameNumber,
                audioBytesSent: prev?.audioBytesSent ?? 0,
                subscriberCount: 1,
              }));
            } else if (message.type === "ended") {
              onStreamEnded?.();
              setStream((prev) => prev ? { ...prev, state: "ended" } : null);
            }
          }
        } catch (e) {
          console.error("Failed to parse voice stream message:", e);
        }
      };

      ws.onclose = () => {
        setIsConnected(false);
        wsRef.current = null;
      };

      ws.onerror = () => {
        setError("WebSocket connection error");
      };
    } catch (e) {
      setError("Failed to connect to voice stream");
    }

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [systemId, streamId, enabled, onAudio, onStreamEnded]);

  return {
    stream,
    isConnected,
    error,
  };
}

interface UseRadioLocationsOptions {
  systemId: string;
  pollInterval?: number;  // milliseconds
  enabled?: boolean;
}

interface UseRadioLocationsResult {
  locations: RadioLocation[];
  isLoading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
}

/**
 * Hook for fetching radio locations from a trunking system.
 *
 * Polls the location cache endpoint at a configurable interval.
 */
export function useRadioLocations({
  systemId,
  pollInterval = 10000,  // 10 seconds default
  enabled = true,
}: UseRadioLocationsOptions): UseRadioLocationsResult {
  const [locations, setLocations] = useState<RadioLocation[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchLocations = useCallback(async () => {
    if (!systemId) return;

    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE}/systems/${systemId}/locations`);
      if (response.ok) {
        const data = await response.json();
        setLocations(data);
        setError(null);
      } else {
        setError(`Failed to fetch locations: ${response.status}`);
      }
    } catch (e) {
      setError("Failed to fetch radio locations");
      console.error("Failed to fetch radio locations:", e);
    } finally {
      setIsLoading(false);
    }
  }, [systemId]);

  useEffect(() => {
    if (!enabled) return;

    fetchLocations();

    const interval = setInterval(fetchLocations, pollInterval);
    return () => clearInterval(interval);
  }, [enabled, pollInterval, fetchLocations]);

  return {
    locations,
    isLoading,
    error,
    refetch: fetchLocations,
  };
}

/**
 * Decode base64 audio to Int16Array for Web Audio API playback.
 */
export function decodeBase64Audio(base64: string): Int16Array {
  const binaryString = atob(base64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return new Int16Array(bytes.buffer);
}

/**
 * Convert Int16Array to Float32Array for Web Audio API.
 */
export function int16ToFloat32(int16: Int16Array): Float32Array {
  const float32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) {
    float32[i] = int16[i] / 32768.0;
  }
  return float32;
}
