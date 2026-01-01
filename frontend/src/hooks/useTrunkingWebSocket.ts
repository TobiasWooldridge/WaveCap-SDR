import { useEffect, useRef, useCallback, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import type {
  TrunkingEvent,
  TrunkingSystem,
  ActiveCall,
  P25Message,
  CallHistoryEntry,
} from "../types/trunking";
import { trunkingKeys } from "./useTrunking";

const MAX_MESSAGES = 500; // Keep last 500 messages

interface UseTrunkingWebSocketOptions {
  systemId?: string | null; // If provided, subscribe to specific system
  enabled?: boolean;
  onCallStart?: (call: ActiveCall) => void;
  onCallEnd?: (call: ActiveCall) => void;
  onMessage?: (message: P25Message) => void;
}

interface UseTrunkingWebSocketResult {
  isConnected: boolean;
  systems: TrunkingSystem[];
  activeCalls: ActiveCall[];
  messages: P25Message[];
  callHistory: CallHistoryEntry[];  // Buffered call history from server
  error: string | null;
}

export function useTrunkingWebSocket(
  options: UseTrunkingWebSocketOptions = {}
): UseTrunkingWebSocketResult {
  const { systemId, enabled = true, onCallStart, onCallEnd, onMessage } = options;
  const queryClient = useQueryClient();
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mountedRef = useRef(true);
  const shouldReconnectRef = useRef(true);

  const [isConnected, setIsConnected] = useState(false);
  const [systems, setSystems] = useState<TrunkingSystem[]>([]);
  const [activeCalls, setActiveCalls] = useState<ActiveCall[]>([]);
  const [messages, setMessages] = useState<P25Message[]>([]);
  const [callHistory, setCallHistory] = useState<CallHistoryEntry[]>([]);
  const [error, setError] = useState<string | null>(null);

  const upsertSystemCache = useCallback(
    (system: TrunkingSystem) => {
      queryClient.setQueryData(trunkingKeys.system(system.id), system);
      queryClient.setQueryData<TrunkingSystem[]>(trunkingKeys.systems(), (old) => {
        if (!old || old.length === 0) {
          return [system];
        }
        let found = false;
        const next = old.map((existing) => {
          if (existing.id === system.id) {
            found = true;
            return system;
          }
          return existing;
        });
        if (!found) {
          next.push(system);
        }
        return next;
      });
    },
    [queryClient],
  );

  const handleEvent = useCallback(
    (event: TrunkingEvent) => {
      switch (event.type) {
        case "snapshot":
          setSystems(event.systems);
          setActiveCalls(event.activeCalls);
          // Load buffered messages from server (newest first, so prepend to empty array)
          if (event.messages && event.messages.length > 0) {
            // Messages are already sorted newest-first from server, but we store oldest-first
            setMessages(event.messages.slice().reverse());
          }
          // Load buffered call history from server
          if (event.callHistory && event.callHistory.length > 0) {
            setCallHistory(event.callHistory);
          }
          // Update query cache (avoid overwriting list with a single-system snapshot)
          if (systemId) {
            const system = event.systems.find((s) => s.id === systemId);
            if (system) {
              upsertSystemCache(system);
            }
          } else {
            queryClient.setQueryData(trunkingKeys.systems(), event.systems);
          }
          break;

        case "system_update":
          setSystems((prev) =>
            prev.map((s) => (s.id === event.systemId ? event.system : s))
          );
          // Update query cache
          upsertSystemCache(event.system);
          break;

        case "call_start":
          setActiveCalls((prev) => [...prev, event.call]);
          onCallStart?.(event.call);
          // Update query cache
          queryClient.invalidateQueries({
            queryKey: trunkingKeys.calls(event.systemId),
          });
          queryClient.invalidateQueries({ queryKey: trunkingKeys.allCalls() });
          break;

        case "call_update":
          setActiveCalls((prev) =>
            prev.map((c) => (c.id === event.call.id ? event.call : c))
          );
          break;

        case "call_end":
          setActiveCalls((prev) => prev.filter((c) => c.id !== event.callId));
          onCallEnd?.(event.call);
          // Update query cache
          queryClient.invalidateQueries({
            queryKey: trunkingKeys.calls(event.systemId),
          });
          queryClient.invalidateQueries({ queryKey: trunkingKeys.allCalls() });
          break;

        case "message":
          // Add message to the list, keeping only the last MAX_MESSAGES
          setMessages((prev) => [...prev.slice(-MAX_MESSAGES + 1), event.message]);
          onMessage?.(event.message);
          break;
      }
    },
    [queryClient, onCallStart, onCallEnd, onMessage, systemId, upsertSystemCache]
  );

  const connect = useCallback(() => {
    if (!mountedRef.current || !shouldReconnectRef.current) return;
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }
    if (wsRef.current?.readyState === WebSocket.CONNECTING) {
      return;
    }

    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.onopen = null;
      wsRef.current.onmessage = null;
      wsRef.current.onerror = null;
      wsRef.current.onclose = null;
      wsRef.current.close();
      wsRef.current = null;
    }

    const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsHost = window.location.host;
    const wsPath = systemId
      ? `/api/v1/trunking/stream/${systemId}`
      : "/api/v1/trunking/stream";
    const wsUrl = `${wsProtocol}//${wsHost}${wsPath}`;

    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      if (!mountedRef.current) return;
      setIsConnected(true);
      setError(null);
    };

    ws.onclose = () => {
      if (!mountedRef.current || !shouldReconnectRef.current) {
        wsRef.current = null;
        return;
      }
      setIsConnected(false);
      wsRef.current = null;

      // Reconnect after delay if enabled
      if (enabled) {
        reconnectTimeoutRef.current = setTimeout(() => {
          connect();
        }, 3000);
      }
    };

    ws.onerror = () => {
      if (!mountedRef.current) return;
      setError("WebSocket connection failed");
    };

    ws.onmessage = (event) => {
      try {
        const message: TrunkingEvent = JSON.parse(event.data);
        handleEvent(message);
      } catch (e) {
        console.error("Failed to parse trunking event:", e);
      }
    };

    wsRef.current = ws;
  }, [enabled, handleEvent, systemId]);

  useEffect(() => {
    if (enabled) {
      mountedRef.current = true;
      shouldReconnectRef.current = true;
      connect();
    }

    return () => {
      mountedRef.current = false;
      shouldReconnectRef.current = false;
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
  }, [enabled, connect]);

  return {
    isConnected,
    systems,
    activeCalls,
    messages,
    callHistory,
    error,
  };
}
