import { useState, useEffect, useRef } from "react";
import type { ErrorEvent, ErrorStats, ErrorType, HealthMessage } from "../types";

interface HealthStreamState {
  isConnected: boolean;
  stats: Partial<Record<ErrorType, ErrorStats>>;
  recentErrors: ErrorEvent[];
  hasActiveErrors: boolean;
}

const MAX_RECENT_ERRORS = 50;
const RECONNECT_DELAY_MS = 5000;
const MAX_RECONNECT_ATTEMPTS = 10;

export function useHealthStream(): HealthStreamState {
  const [state, setState] = useState<HealthStreamState>({
    isConnected: false,
    stats: {},
    recentErrors: [],
    hasActiveErrors: false,
  });

  const wsRef = useRef<WebSocket | null>(null);
  const errorsRef = useRef<ErrorEvent[]>([]);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;

    const connect = () => {
      // Don't connect if unmounted
      if (!mountedRef.current) return;

      // Don't reconnect if already connected or connecting
      if (wsRef.current?.readyState === WebSocket.OPEN ||
          wsRef.current?.readyState === WebSocket.CONNECTING) {
        return;
      }

      // Give up after max attempts
      if (reconnectAttemptsRef.current >= MAX_RECONNECT_ATTEMPTS) {
        return;
      }

      try {
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const ws = new WebSocket(`${protocol}//${window.location.host}/api/v1/stream/health`);
        wsRef.current = ws;

        ws.onopen = () => {
          if (!mountedRef.current) return;
          reconnectAttemptsRef.current = 0; // Reset on successful connection
          setState((s) => ({ ...s, isConnected: true }));
        };

        ws.onclose = () => {
          if (!mountedRef.current) return;
          wsRef.current = null;
          setState((s) => ({ ...s, isConnected: false }));

          // Schedule reconnect with backoff
          reconnectAttemptsRef.current++;
          if (reconnectAttemptsRef.current < MAX_RECONNECT_ATTEMPTS) {
            const delay = RECONNECT_DELAY_MS * Math.min(reconnectAttemptsRef.current, 3);
            reconnectTimeoutRef.current = setTimeout(connect, delay);
          }
        };

        ws.onerror = () => {
          // Will trigger onclose
        };

        ws.onmessage = (event) => {
          if (!mountedRef.current) return;
          try {
            const msg: HealthMessage = JSON.parse(event.data);

            if (msg.type === "error") {
              errorsRef.current = [msg.event, ...errorsRef.current.slice(0, MAX_RECENT_ERRORS - 1)];
              setState((s) => ({
                ...s,
                recentErrors: [...errorsRef.current],
                hasActiveErrors: true,
              }));
            } else if (msg.type === "stats") {
              const hasActiveErrors = Object.values(msg.data).some((stat) => stat && stat.rate > 0);
              setState((s) => ({
                ...s,
                stats: { ...msg.data },
                hasActiveErrors,
              }));
            }
          } catch {
            // Ignore malformed messages
          }
        };
      } catch {
        // WebSocket constructor failed, schedule retry
        reconnectAttemptsRef.current++;
        if (reconnectAttemptsRef.current < MAX_RECONNECT_ATTEMPTS) {
          reconnectTimeoutRef.current = setTimeout(connect, RECONNECT_DELAY_MS);
        }
      }
    };

    // Delay initial connection slightly to avoid race conditions during mount
    const initialTimeout = setTimeout(connect, 100);

    return () => {
      mountedRef.current = false;
      clearTimeout(initialTimeout);
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.onclose = null; // Prevent reconnect on cleanup
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, []);

  return state;
}
