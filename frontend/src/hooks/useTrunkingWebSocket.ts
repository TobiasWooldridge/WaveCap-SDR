import { useEffect, useRef, useCallback, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import type {
  TrunkingEvent,
  TrunkingSystem,
  ActiveCall,
} from "../types/trunking";
import { trunkingKeys } from "./useTrunking";

interface UseTrunkingWebSocketOptions {
  systemId?: string | null; // If provided, subscribe to specific system
  enabled?: boolean;
  onCallStart?: (call: ActiveCall) => void;
  onCallEnd?: (call: ActiveCall) => void;
}

interface UseTrunkingWebSocketResult {
  isConnected: boolean;
  systems: TrunkingSystem[];
  activeCalls: ActiveCall[];
  error: string | null;
}

export function useTrunkingWebSocket(
  options: UseTrunkingWebSocketOptions = {}
): UseTrunkingWebSocketResult {
  const { systemId, enabled = true, onCallStart, onCallEnd } = options;
  const queryClient = useQueryClient();
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const [isConnected, setIsConnected] = useState(false);
  const [systems, setSystems] = useState<TrunkingSystem[]>([]);
  const [activeCalls, setActiveCalls] = useState<ActiveCall[]>([]);
  const [error, setError] = useState<string | null>(null);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsHost = window.location.host;
    const wsPath = systemId
      ? `/api/v1/trunking/stream/${systemId}`
      : "/api/v1/trunking/stream";
    const wsUrl = `${wsProtocol}//${wsHost}${wsPath}`;

    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      setIsConnected(true);
      setError(null);
    };

    ws.onclose = () => {
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
  }, [systemId, enabled]);

  const handleEvent = useCallback(
    (event: TrunkingEvent) => {
      switch (event.type) {
        case "snapshot":
          setSystems(event.systems);
          setActiveCalls(event.activeCalls);
          // Update query cache
          queryClient.setQueryData(trunkingKeys.systems(), event.systems);
          break;

        case "system_update":
          setSystems((prev) =>
            prev.map((s) => (s.id === event.systemId ? event.system : s))
          );
          // Update query cache
          queryClient.setQueryData(
            trunkingKeys.system(event.systemId),
            event.system
          );
          queryClient.invalidateQueries({ queryKey: trunkingKeys.systems() });
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
      }
    },
    [queryClient, onCallStart, onCallEnd]
  );

  useEffect(() => {
    if (enabled) {
      connect();
    }

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [enabled, connect]);

  return {
    isConnected,
    systems,
    activeCalls,
    error,
  };
}
