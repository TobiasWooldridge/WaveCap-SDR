import { useState, useEffect, useRef, useCallback } from "react";
import type { ErrorEvent, ErrorStats, ErrorType, HealthMessage } from "../types";

interface HealthStreamState {
  isConnected: boolean;
  stats: Partial<Record<ErrorType, ErrorStats>>;
  recentErrors: ErrorEvent[];
  hasActiveErrors: boolean;
}

const MAX_RECENT_ERRORS = 50;
const RECONNECT_DELAY_MS = 3000;

export function useHealthStream(): HealthStreamState {
  const [state, setState] = useState<HealthStreamState>({
    isConnected: false,
    stats: {},
    recentErrors: [],
    hasActiveErrors: false,
  });

  const wsRef = useRef<WebSocket | null>(null);
  const errorsRef = useRef<ErrorEvent[]>([]);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const connect = useCallback(() => {
    // Clear any pending reconnect
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    // Don't reconnect if already connected
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${protocol}//${window.location.host}/api/v1/stream/health`);
    wsRef.current = ws;

    ws.onopen = () => {
      setState((s) => ({ ...s, isConnected: true }));
    };

    ws.onclose = () => {
      setState((s) => ({ ...s, isConnected: false }));
      // Schedule reconnect
      reconnectTimeoutRef.current = setTimeout(connect, RECONNECT_DELAY_MS);
    };

    ws.onerror = () => {
      // Will trigger onclose
    };

    ws.onmessage = (event) => {
      try {
        const msg: HealthMessage = JSON.parse(event.data);

        if (msg.type === "error") {
          // Add to recent errors (newest first)
          errorsRef.current = [msg.event, ...errorsRef.current.slice(0, MAX_RECENT_ERRORS - 1)];
          setState((s) => ({
            ...s,
            recentErrors: errorsRef.current,
            hasActiveErrors: true,
          }));
        } else if (msg.type === "stats") {
          // Check if any error rate is above zero
          const hasActiveErrors = Object.values(msg.data).some((stat) => stat && stat.rate > 0);
          setState((s) => ({
            ...s,
            stats: msg.data,
            hasActiveErrors,
          }));
        }
      } catch {
        // Ignore malformed messages
      }
    };
  }, []);

  useEffect(() => {
    connect();

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connect]);

  return state;
}
