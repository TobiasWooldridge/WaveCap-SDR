import { useState, useEffect, useRef } from 'react';
import type { Capture } from '../types';

interface SpectrumData {
  power: number[];
  freqs: number[];
  centerHz: number;
  sampleRate: number;
}

interface SpectrumDataHook {
  spectrumData: SpectrumData | null;
  isConnected: boolean;
  isIdle: boolean;
}

// Shared WebSocket connection manager
class SpectrumWebSocketManager {
  private ws: WebSocket | null = null;
  private subscribers: Set<(data: SpectrumData) => void> = new Set();
  private connectionListeners: Set<(connected: boolean) => void> = new Set();
  private captureId: string | null = null;
  private captureState: string | null = null;
  private isIdle: boolean = false;
  private reconnectTimeout: NodeJS.Timeout | null = null;

  subscribe(
    captureId: string,
    captureState: string,
    isIdle: boolean,
    isPaused: boolean,
    onData: (data: SpectrumData) => void,
    onConnectionChange: (connected: boolean) => void
  ) {
    // Track active (non-paused) subscribers separately
    if (!isPaused) {
      this.subscribers.add(onData);
    }
    this.connectionListeners.add(onConnectionChange);

    // Check if we need to reconnect with different parameters
    const needsReconnect =
      this.captureId !== captureId ||
      this.captureState !== captureState ||
      this.isIdle !== isIdle;

    if (needsReconnect) {
      this.captureId = captureId;
      this.captureState = captureState;
      this.isIdle = isIdle;
      this.connect();
    } else if (this.ws && this.ws.readyState === WebSocket.OPEN && !isPaused) {
      // Already connected, notify new subscriber
      onConnectionChange(true);
    } else if (isPaused) {
      // Paused subscriber, disconnect if no active subscribers
      if (this.subscribers.size === 0) {
        this.disconnect();
      }
      onConnectionChange(false);
    } else if (!isPaused && this.subscribers.size > 0 && (!this.ws || this.ws.readyState !== WebSocket.OPEN)) {
      // We have active subscribers but no connection - reconnect
      this.connect();
    }

    return () => {
      this.subscribers.delete(onData);
      this.connectionListeners.delete(onConnectionChange);

      // If no more active subscribers, disconnect
      if (this.subscribers.size === 0) {
        this.disconnect();
      }
    };
  }

  private connect() {
    // Disconnect existing connection
    this.disconnect();

    // Don't connect if capture is not running or UI is idle
    if (this.captureState !== 'running' || this.isIdle || !this.captureId) {
      this.notifyConnectionChange(false);
      return;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/api/v1/stream/captures/${this.captureId}/spectrum`;

    try {
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        console.log('Shared spectrum WebSocket connected');
        this.notifyConnectionChange(true);
      };

      this.ws.onmessage = (event) => {
        try {
          const data: SpectrumData = JSON.parse(event.data);
          this.notifyData(data);
        } catch (error) {
          console.error('Error parsing spectrum data:', error);
        }
      };

      this.ws.onerror = (error) => {
        console.error('Shared spectrum WebSocket error:', error);
      };

      this.ws.onclose = () => {
        console.log('Shared spectrum WebSocket disconnected');
        this.notifyConnectionChange(false);
        this.ws = null;

        // Auto-reconnect if we still have subscribers and should be connected
        if (
          this.subscribers.size > 0 &&
          this.captureState === 'running' &&
          !this.isIdle
        ) {
          // Exponential backoff reconnection
          if (this.reconnectTimeout) {
            clearTimeout(this.reconnectTimeout);
          }
          this.reconnectTimeout = setTimeout(() => {
            console.log('Attempting to reconnect spectrum WebSocket...');
            this.connect();
          }, 2000);
        }
      };
    } catch (error) {
      console.error('Error creating WebSocket:', error);
      this.notifyConnectionChange(false);
    }
  }

  private disconnect() {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }

    if (this.ws) {
      const ws = this.ws;
      ws.onopen = null;
      ws.onmessage = null;
      ws.onerror = null;
      ws.onclose = null;
      if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
        ws.close();
      }
      this.ws = null;
    }
  }

  private notifyData(data: SpectrumData) {
    this.subscribers.forEach((callback) => callback(data));
  }

  private notifyConnectionChange(connected: boolean) {
    this.connectionListeners.forEach((callback) => callback(connected));
  }
}

// Singleton instance
const spectrumManager = new SpectrumWebSocketManager();

export function useSpectrumData(capture: Capture, isPaused: boolean = false): SpectrumDataHook {
  const [spectrumData, setSpectrumData] = useState<SpectrumData | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isIdle, setIsIdle] = useState(false);
  const idleTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Track user activity to detect idle state
  useEffect(() => {
    const IDLE_TIMEOUT = 60000; // 60 seconds

    const resetIdleTimer = () => {
      setIsIdle(false);

      if (idleTimerRef.current) {
        clearTimeout(idleTimerRef.current);
      }

      idleTimerRef.current = setTimeout(() => {
        console.log('Spectrum data: UI idle, pausing');
        setIsIdle(true);
      }, IDLE_TIMEOUT);
    };

    const activityEvents = ['mousedown', 'mousemove', 'keydown', 'scroll', 'touchstart'];

    activityEvents.forEach((event) => {
      window.addEventListener(event, resetIdleTimer, { passive: true });
    });

    resetIdleTimer();

    return () => {
      activityEvents.forEach((event) => {
        window.removeEventListener(event, resetIdleTimer);
      });
      if (idleTimerRef.current) {
        clearTimeout(idleTimerRef.current);
      }
    };
  }, []);

  // Subscribe to shared WebSocket
  useEffect(() => {
    // Clear spectrum data if capture stopped or paused
    if (capture.state !== 'running' || isPaused) {
      setSpectrumData(null);
    }

    const unsubscribe = spectrumManager.subscribe(
      capture.id,
      capture.state,
      isIdle,
      isPaused,
      setSpectrumData,
      setIsConnected
    );

    return unsubscribe;
  }, [capture.id, capture.state, isIdle, isPaused]);

  return { spectrumData, isConnected, isIdle };
}
