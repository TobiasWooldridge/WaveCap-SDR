import { useState, useEffect, useRef, useCallback } from "react";
import type {
  SystemMetrics,
  CaptureMetrics,
  LogEntry,
  ErrorEvent,
  SystemStreamMessage,
} from "../types";

interface SystemStreamState {
  isConnected: boolean;
  systemMetrics: SystemMetrics | null;
  captureMetrics: CaptureMetrics[];
  logs: LogEntry[];
  errors: ErrorEvent[];
}

const MAX_LOGS = 500;
const MAX_ERRORS = 100;
const RECONNECT_DELAY_MS = 3000;
const MAX_RECONNECT_ATTEMPTS = 10;

export function useSystemStream(): SystemStreamState & {
  clearLogs: () => void;
  clearErrors: () => void;
} {
  const [state, setState] = useState<SystemStreamState>({
    isConnected: false,
    systemMetrics: null,
    captureMetrics: [],
    logs: [],
    errors: [],
  });

  const wsRef = useRef<WebSocket | null>(null);
  const logsRef = useRef<LogEntry[]>([]);
  const errorsRef = useRef<ErrorEvent[]>([]);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(
    null,
  );
  const reconnectAttemptsRef = useRef(0);
  const mountedRef = useRef(true);

  const clearLogs = useCallback(() => {
    logsRef.current = [];
    setState((s) => ({ ...s, logs: [] }));
  }, []);

  const clearErrors = useCallback(() => {
    errorsRef.current = [];
    setState((s) => ({ ...s, errors: [] }));
  }, []);

  useEffect(() => {
    mountedRef.current = true;

    const connect = () => {
      // Don't connect if unmounted
      if (!mountedRef.current) return;

      // Don't reconnect if already connected or connecting
      if (
        wsRef.current?.readyState === WebSocket.OPEN ||
        wsRef.current?.readyState === WebSocket.CONNECTING
      ) {
        return;
      }

      // Give up after max attempts
      if (reconnectAttemptsRef.current >= MAX_RECONNECT_ATTEMPTS) {
        return;
      }

      try {
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const ws = new WebSocket(
          `${protocol}//${window.location.host}/api/v1/stream/system`,
        );
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
            const delay =
              RECONNECT_DELAY_MS * Math.min(reconnectAttemptsRef.current, 3);
            reconnectTimeoutRef.current = setTimeout(connect, delay);
          }
        };

        ws.onerror = () => {
          // Will trigger onclose
        };

        ws.onmessage = (event) => {
          if (!mountedRef.current) return;
          try {
            const msg: SystemStreamMessage = JSON.parse(event.data);

            if (msg.type === "metrics") {
              setState((s) => ({
                ...s,
                systemMetrics: msg.system,
                captureMetrics: msg.captures,
              }));
            } else if (msg.type === "log") {
              // Add new log to front, trim to max
              logsRef.current = [
                msg.entry,
                ...logsRef.current.slice(0, MAX_LOGS - 1),
              ];
              setState((s) => ({
                ...s,
                logs: [...logsRef.current],
              }));
            } else if (msg.type === "logs_snapshot") {
              // Replace logs with snapshot
              logsRef.current = msg.entries.slice(0, MAX_LOGS);
              setState((s) => ({
                ...s,
                logs: [...logsRef.current],
              }));
            } else if (msg.type === "error") {
              // Add new error to front, trim to max
              errorsRef.current = [
                msg.event,
                ...errorsRef.current.slice(0, MAX_ERRORS - 1),
              ];
              setState((s) => ({
                ...s,
                errors: [...errorsRef.current],
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

  return { ...state, clearLogs, clearErrors };
}
