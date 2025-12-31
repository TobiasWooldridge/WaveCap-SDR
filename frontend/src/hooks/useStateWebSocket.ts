import { useEffect, useRef, useCallback, useSyncExternalStore } from "react";
import { useQueryClient } from "@tanstack/react-query";
import type {
  Capture,
  Channel,
  Device,
  Scanner,
  StateMessage,
  StateChangeMessage,
  StateSnapshotMessage,
} from "../types";

let stateStreamConnected = false;
const stateStreamListeners = new Set<() => void>();

const notifyStateStreamListeners = () => {
  for (const listener of stateStreamListeners) {
    listener();
  }
};

const setStateStreamConnected = (connected: boolean) => {
  if (stateStreamConnected === connected) return;
  stateStreamConnected = connected;
  notifyStateStreamListeners();
};

export function useStateStreamStatus() {
  return useSyncExternalStore(
    (listener) => {
      stateStreamListeners.add(listener);
      return () => {
        stateStreamListeners.delete(listener);
      };
    },
    () => stateStreamConnected,
    () => false,
  );
}

/**
 * WebSocket hook for real-time state updates.
 *
 * Subscribes to /api/v1/stream/state and updates React Query cache
 * when capture/channel/scanner state changes.
 *
 * This replaces polling with push-based updates for better efficiency.
 */
export function useStateWebSocket() {
  const queryClient = useQueryClient();
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 10;
  const mountedRef = useRef(true);
  const shouldReconnectRef = useRef(true);

  const updateCaptureCache = useCallback(
    (action: string, id: string, data: Capture | null) => {
      queryClient.setQueryData<Capture[]>(["captures"], (old) => {
        if (!old) return old;

        switch (action) {
          case "created":
            if (data && !old.find((c) => c.id === id)) {
              return [...old, data];
            }
            return old;

          case "updated":
          case "started":
          case "stopped":
            if (data) {
              return old.map((c) => (c.id === id ? { ...c, ...data } : c));
            }
            return old;

          case "deleted":
            return old.filter((c) => c.id !== id);

          default:
            return old;
        }
      });
    },
    [queryClient],
  );

  const updateChannelCache = useCallback(
    (action: string, id: string, data: Channel | null) => {
      // We need to update the channels for the specific capture
      // For deleted channels, we need to check all capture channel lists
      if (action === "deleted") {
        // Get all query keys matching ["channels", *]
        const queries = queryClient.getQueriesData<Channel[]>({
          queryKey: ["channels"],
        });
        for (const [key, channels] of queries) {
          if (channels && Array.isArray(channels)) {
            const captureId = key[1] as string;
            if (channels.some((ch) => ch.id === id)) {
              queryClient.setQueryData<Channel[]>(
                ["channels", captureId],
                (old) => (old ? old.filter((ch) => ch.id !== id) : old),
              );
            }
          }
        }
        return;
      }

      if (!data) return;
      const captureId = data.captureId;

      queryClient.setQueryData<Channel[]>(["channels", captureId], (old) => {
        if (!old) {
          // If no cache exists yet, create with this channel
          return action === "created" ? [data] : undefined;
        }

        switch (action) {
          case "created":
            if (!old.find((ch) => ch.id === id)) {
              return [...old, data];
            }
            return old;

          case "updated":
          case "started":
          case "stopped":
            return old.map((ch) => (ch.id === id ? { ...ch, ...data } : ch));

          default:
            return old;
        }
      });
    },
    [queryClient],
  );

  const updateScannerCache = useCallback(
    (action: string, id: string, data: Scanner | null) => {
      queryClient.setQueryData<Scanner[]>(["scanners"], (old) => {
        if (!old) return old;

        switch (action) {
          case "created":
            if (data && !old.find((s) => s.id === id)) {
              return [...old, data];
            }
            return old;

          case "updated":
          case "started":
          case "stopped":
            if (data) {
              return old.map((s) => (s.id === id ? { ...s, ...data } : s));
            }
            return old;

          case "deleted":
            return old.filter((s) => s.id !== id);

          default:
            return old;
        }
      });

      if (action === "deleted") {
        queryClient.removeQueries({ queryKey: ["scanners", id] });
        return;
      }

      if (data) {
        queryClient.setQueryData<Scanner>(["scanners", id], (old) =>
          old ? { ...old, ...data } : (data as Scanner),
        );
      }
    },
    [queryClient],
  );

  const updateDeviceCache = useCallback(
    (action: string, id: string, data: Device | null) => {
      queryClient.setQueryData<Device[]>(["devices"], (old) => {
        if (!old) return old;

        switch (action) {
          case "created":
            if (data && !old.find((d) => d.id === id)) {
              return [...old, data];
            }
            return old;

          case "updated":
            if (data) {
              return old.map((d) => (d.id === id ? { ...d, ...data } : d));
            }
            return old;

          case "deleted":
            return old.filter((d) => d.id !== id);

          default:
            return old;
        }
      });
    },
    [queryClient],
  );

  const handleStateChange = useCallback(
    (message: StateChangeMessage) => {
      const { type, action, id, data } = message;
      console.log(`[StateWS] ${type}.${action}:`, id);

      if (type === "capture") {
        updateCaptureCache(action, id, data as Capture | null);
      } else if (type === "channel") {
        updateChannelCache(action, id, data as Channel | null);
      } else if (type === "scanner") {
        updateScannerCache(action, id, data as Scanner | null);
      } else if (type === "device") {
        updateDeviceCache(action, id, data as Device | null);
      }
    },
    [
      updateCaptureCache,
      updateChannelCache,
      updateScannerCache,
      updateDeviceCache,
    ],
  );

  const handleSnapshot = useCallback(
    (message: StateSnapshotMessage) => {
      console.log("[StateWS] Received snapshot:", {
        captures: message.captures.length,
        channels: message.channels.length,
        scanners: message.scanners.length,
        devices: message.devices.length,
      });

      // Update captures cache
      queryClient.setQueryData<Capture[]>(["captures"], message.captures);

      // Update channels cache per capture
      const channelsByCapture = new Map<string, Channel[]>();
      for (const channel of message.channels) {
        const existing = channelsByCapture.get(channel.captureId) || [];
        existing.push(channel);
        channelsByCapture.set(channel.captureId, existing);
      }
      for (const [captureId, channels] of channelsByCapture) {
        queryClient.setQueryData<Channel[]>(["channels", captureId], channels);
      }

      // Update scanners cache
      queryClient.setQueryData<Scanner[]>(["scanners"], message.scanners);

      // Update devices cache
      queryClient.setQueryData<Device[]>(["devices"], message.devices);
    },
    [queryClient],
  );

  const handleMessage = useCallback(
    (message: StateMessage) => {
      if (message.type === "ping") {
        // Keepalive - no action needed
        return;
      }

      if (message.type === "snapshot") {
        // Full state snapshot - replace all cached data
        handleSnapshot(message);
        return;
      }

      // Incremental state change
      handleStateChange(message);
    },
    [handleSnapshot, handleStateChange],
  );

  const connect = useCallback(() => {
    if (!mountedRef.current || !shouldReconnectRef.current) return;

    if (
      wsRef.current &&
      (wsRef.current.readyState === WebSocket.OPEN ||
        wsRef.current.readyState === WebSocket.CONNECTING)
    ) {
      return;
    }

    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    // Clean up any stale connection
    if (wsRef.current) {
      wsRef.current.onopen = null;
      wsRef.current.onmessage = null;
      wsRef.current.onerror = null;
      wsRef.current.onclose = null;
      wsRef.current.close();
      wsRef.current = null;
    }

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/api/v1/stream/state`;

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log("[StateWS] Connected");
      reconnectAttempts.current = 0;
      setStateStreamConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const message: StateMessage = JSON.parse(event.data);
        handleMessage(message);
      } catch (e) {
        console.error("[StateWS] Failed to parse message:", e);
      }
    };

    ws.onerror = (error) => {
      console.error("[StateWS] Error:", error);
    };

    ws.onclose = (event) => {
      if (!mountedRef.current || !shouldReconnectRef.current) {
        wsRef.current = null;
        setStateStreamConnected(false);
        return;
      }
      console.log("[StateWS] Disconnected", event.code, event.reason);
      wsRef.current = null;
      setStateStreamConnected(false);

      // Attempt reconnection with exponential backoff
      if (reconnectAttempts.current < maxReconnectAttempts) {
        const delay = Math.min(
          1000 * Math.pow(2, reconnectAttempts.current),
          30000,
        );
        reconnectAttempts.current++;
        console.log(
          `[StateWS] Reconnecting in ${delay}ms (attempt ${reconnectAttempts.current})`,
        );
        reconnectTimeoutRef.current = window.setTimeout(connect, delay);
      } else {
        console.error("[StateWS] Max reconnection attempts reached");
      }
    };
  }, [handleMessage]);

  // Connect on mount, disconnect on unmount
  useEffect(() => {
    mountedRef.current = true;
    shouldReconnectRef.current = true;
    connect();

    return () => {
      mountedRef.current = false;
      shouldReconnectRef.current = false;
      setStateStreamConnected(false);
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.onopen = null;
        wsRef.current.onmessage = null;
        wsRef.current.onerror = null;
        wsRef.current.onclose = null;
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [connect]);

  return {
    isConnected: wsRef.current?.readyState === WebSocket.OPEN,
  };
}
