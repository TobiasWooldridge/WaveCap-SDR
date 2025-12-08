import { useState, useEffect } from 'react';
import type { Capture } from '../types';

export interface SpectrumData {
  power: number[];
  freqs: number[];
  centerHz: number;
  sampleRate: number;
  fftSize?: number;  // FFT bin count
  actualFps?: number;  // Measured frames per second
}

interface SpectrumDataHook {
  spectrumData: SpectrumData | null;
  isConnected: boolean;
  isIdle: boolean;
}

// Shared WebSocket connection manager with built-in idle tracking
class SpectrumWebSocketManager {
  private ws: WebSocket | null = null;
  private subscribers: Set<(data: SpectrumData) => void> = new Set();
  private connectionListeners: Set<(connected: boolean) => void> = new Set();
  private captureId: string | null = null;
  private captureState: string | null = null;
  private reconnectTimeout: NodeJS.Timeout | null = null;
  private disconnectTimeout: NodeJS.Timeout | null = null;

  // Reconnection with exponential backoff
  private reconnectAttempts: number = 0;
  private readonly MAX_RECONNECT_ATTEMPTS = 10;
  private readonly INITIAL_RECONNECT_DELAY = 1000;
  private readonly MAX_RECONNECT_DELAY = 30000;

  // Shared idle tracking
  private isIdle: boolean = false;
  private idleTimeout: NodeJS.Timeout | null = null;
  private readonly IDLE_TIMEOUT = 60000; // 60 seconds
  private activityBound = false;

  private initActivityTracking() {
    if (this.activityBound) return;
    this.activityBound = true;

    const resetIdleTimer = () => {
      const wasIdle = this.isIdle;
      this.isIdle = false;

      if (this.idleTimeout) {
        clearTimeout(this.idleTimeout);
      }

      this.idleTimeout = setTimeout(() => {
        console.log('Spectrum data: UI idle, disconnecting');
        this.isIdle = true;
        // Disconnect if idle and no paused override
        if (this.subscribers.size > 0) {
          this.disconnect();
          this.notifyConnectionChange(false);
        }
      }, this.IDLE_TIMEOUT);

      // Reconnect if we were idle and now active
      if (wasIdle && this.subscribers.size > 0 && this.captureState === 'running') {
        console.log('Spectrum data: UI active, reconnecting');
        this.connect();
      }
    };

    const activityEvents = ['mousedown', 'mousemove', 'keydown', 'scroll', 'touchstart'];
    activityEvents.forEach((event) => {
      window.addEventListener(event, resetIdleTimer, { passive: true });
    });

    resetIdleTimer();
  }

  subscribe(
    captureId: string,
    captureState: string,
    isPaused: boolean,
    onData: (data: SpectrumData) => void,
    onConnectionChange: (connected: boolean) => void
  ) {
    // Initialize activity tracking on first subscription
    this.initActivityTracking();

    // Cancel any pending disconnect (handles React StrictMode remount)
    if (this.disconnectTimeout) {
      clearTimeout(this.disconnectTimeout);
      this.disconnectTimeout = null;
    }

    // Track active (non-paused) subscribers
    if (!isPaused) {
      this.subscribers.add(onData);
    }
    this.connectionListeners.add(onConnectionChange);

    // Check if we need to reconnect with different capture
    // Only reconnect if capture ID changed or if capture just started running
    const captureChanged = this.captureId !== captureId;
    const justStartedRunning = this.captureState !== 'running' && captureState === 'running';
    const needsReconnect = captureChanged || justStartedRunning;

    // Update tracked state
    this.captureId = captureId;
    this.captureState = captureState;

    if (needsReconnect) {
      // Reset reconnection attempts on capture change
      this.reconnectAttempts = 0;
      this.connect();
    } else if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING) && !isPaused) {
      // Already connected or connecting, notify new subscriber
      if (this.ws.readyState === WebSocket.OPEN) {
        onConnectionChange(true);
      }
    } else if (isPaused) {
      // Paused subscriber, schedule disconnect if no active subscribers
      if (this.subscribers.size === 0) {
        this.scheduleDisconnect();
      }
      onConnectionChange(false);
    } else if (!isPaused && this.subscribers.size > 0 && !this.ws && !this.isIdle) {
      // We have active subscribers but no WebSocket - connect (unless idle)
      this.connect();
    }

    return () => {
      this.subscribers.delete(onData);
      this.connectionListeners.delete(onConnectionChange);

      // If no more active subscribers, schedule disconnect with delay
      // (allows React StrictMode to remount without losing connection)
      if (this.subscribers.size === 0) {
        this.scheduleDisconnect();
      }
    };
  }

  private scheduleDisconnect() {
    // Cancel any existing disconnect timeout
    if (this.disconnectTimeout) {
      clearTimeout(this.disconnectTimeout);
    }

    // Delay disconnect to handle React StrictMode double-invocation
    this.disconnectTimeout = setTimeout(() => {
      this.disconnectTimeout = null;
      // Only disconnect if still no subscribers
      if (this.subscribers.size === 0) {
        this.disconnect();
      }
    }, 100);
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
        console.log('Spectrum WebSocket connected');
        this.reconnectAttempts = 0; // Reset on successful connection
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

      this.ws.onclose = (event) => {
        console.log(`Spectrum WebSocket closed: code=${event.code}, wasClean=${event.wasClean}`);
        this.notifyConnectionChange(false);
        this.ws = null;

        // Auto-reconnect if we still have subscribers and should be connected
        if (
          this.subscribers.size > 0 &&
          this.captureState === 'running' &&
          !this.isIdle
        ) {
          // Check if we've exceeded max attempts
          if (this.reconnectAttempts >= this.MAX_RECONNECT_ATTEMPTS) {
            console.error('Spectrum WebSocket: max reconnection attempts reached');
            return;
          }

          // Exponential backoff reconnection
          if (this.reconnectTimeout) {
            clearTimeout(this.reconnectTimeout);
          }

          const delay = Math.min(
            this.INITIAL_RECONNECT_DELAY * Math.pow(2, this.reconnectAttempts),
            this.MAX_RECONNECT_DELAY
          );
          this.reconnectAttempts++;

          console.log(`Spectrum WebSocket: reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
          this.reconnectTimeout = setTimeout(() => {
            this.connect();
          }, delay);
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

    if (this.disconnectTimeout) {
      clearTimeout(this.disconnectTimeout);
      this.disconnectTimeout = null;
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

  getIsIdle(): boolean {
    return this.isIdle;
  }
}

// Singleton instance
const spectrumManager = new SpectrumWebSocketManager();

export function useSpectrumData(capture: Capture, isPaused: boolean = false): SpectrumDataHook {
  const [spectrumData, setSpectrumData] = useState<SpectrumData | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  // Subscribe to shared WebSocket
  useEffect(() => {
    // Clear spectrum data if capture stopped or paused
    if (capture.state !== 'running' || isPaused) {
      setSpectrumData(null);
    }

    const unsubscribe = spectrumManager.subscribe(
      capture.id,
      capture.state,
      isPaused,
      setSpectrumData,
      setIsConnected
    );

    return unsubscribe;
  }, [capture.id, capture.state, isPaused]);

  return { spectrumData, isConnected, isIdle: spectrumManager.getIsIdle() };
}
