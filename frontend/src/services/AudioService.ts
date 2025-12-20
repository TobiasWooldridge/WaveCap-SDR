/**
 * Centralized audio service for managing WebSocket audio streams.
 *
 * This singleton handles:
 * - AudioContext initialization with Safari/iOS compatibility
 * - Per-channel WebSocket connections for PCM audio streaming
 * - Multi-channel mixing with automatic gain normalization
 * - Master volume control
 *
 * React components use the useAudio hook to interact with this service.
 */

interface ChannelStream {
  ws: WebSocket;
  gainNode: GainNode;
  shouldPlay: boolean;
  nextStartTime: number;
  pcmBuffer: number[];
}

export interface AudioServiceState {
  playingChannels: Set<string>;
  masterVolume: number;
  isContextReady: boolean;
}

type StateListener = (state: AudioServiceState) => void;

class AudioService {
  private context: AudioContext | null = null;
  private masterGain: GainNode | null = null;
  private channels: Map<string, ChannelStream> = new Map();
  private listeners: Set<StateListener> = new Set();
  private _masterVolume = 0.7;

  // Cached state for useSyncExternalStore - must return same reference if unchanged
  private _cachedState: AudioServiceState = {
    playingChannels: new Set(),
    masterVolume: 0.7,
    isContextReady: false,
  };

  private readonly BUFFER_SIZE = 4096;
  private readonly SAMPLE_RATE = 48000;

  /**
   * Get the current state of the audio service.
   * Returns cached state to prevent unnecessary re-renders with useSyncExternalStore.
   */
  getState(): AudioServiceState {
    return this._cachedState;
  }

  /**
   * Update the cached state. Only creates new object if state actually changed.
   */
  private updateCachedState(): void {
    const newPlayingChannels = new Set(this.channels.keys());
    const newIsContextReady = this.context !== null && this.context.state === "running";

    // Check if anything actually changed
    const playingChanged =
      newPlayingChannels.size !== this._cachedState.playingChannels.size ||
      [...newPlayingChannels].some((id) => !this._cachedState.playingChannels.has(id));

    if (
      playingChanged ||
      this._masterVolume !== this._cachedState.masterVolume ||
      newIsContextReady !== this._cachedState.isContextReady
    ) {
      this._cachedState = {
        playingChannels: newPlayingChannels,
        masterVolume: this._masterVolume,
        isContextReady: newIsContextReady,
      };
    }
  }

  /**
   * Subscribe to state changes.
   * Returns unsubscribe function.
   */
  subscribe(listener: StateListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private notifyListeners(): void {
    this.updateCachedState();
    const state = this._cachedState;
    this.listeners.forEach((listener) => listener(state));
  }

  /**
   * Initialize AudioContext with Safari/iOS compatibility.
   * Must be called from a user gesture (click, tap, etc).
   */
  async init(): Promise<AudioContext> {
    if (this.context) {
      // Resume if suspended (Safari/iOS requirement)
      if (this.context.state === "suspended") {
        await this.context.resume();
        this.notifyListeners();
      }
      return this.context;
    }

    // Use webkitAudioContext for older Safari versions
    const AudioContextClass =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;

    this.context = new AudioContextClass({ sampleRate: this.SAMPLE_RATE });
    this.masterGain = this.context.createGain();
    this.masterGain.gain.value = this._masterVolume;
    this.masterGain.connect(this.context.destination);

    // Resume if needed
    if (this.context.state === "suspended") {
      await this.context.resume();
    }

    this.notifyListeners();
    return this.context;
  }

  /**
   * Start playing audio for a channel.
   * Opens a WebSocket connection and streams PCM audio.
   */
  async play(channelId: string): Promise<void> {
    console.log("[AudioService] play() called for channel:", channelId);

    // Already playing
    if (this.channels.has(channelId)) {
      console.log("[AudioService] Already playing:", channelId);
      return;
    }

    try {
      const audioContext = await this.init();
      console.log("[AudioService] AudioContext state:", audioContext.state);

      // Create a gain node for this channel
      const channelGain = audioContext.createGain();
      channelGain.gain.value = 1.0; // Set explicit initial gain
      channelGain.connect(this.masterGain!);

      // Build WebSocket URL
      const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const wsUrl = `${wsProtocol}//${window.location.host}/api/v1/stream/channels/${channelId}?format=pcm16`;
      console.log("[AudioService] Connecting to WebSocket:", wsUrl);
      console.log("[AudioService] AudioContext state:", audioContext.state, "sampleRate:", audioContext.sampleRate);

      const ws = new WebSocket(wsUrl);
      ws.binaryType = "arraybuffer";

    const stream: ChannelStream = {
      ws,
      gainNode: channelGain,
      shouldPlay: true,
      nextStartTime: audioContext.currentTime,
      pcmBuffer: [],
    };

    this.channels.set(channelId, stream);
    this.updateChannelGains();
    this.notifyListeners();

    ws.onopen = () => {
      console.log("[AudioService] WebSocket connected:", channelId, "readyState:", ws.readyState);
    };

    ws.onmessage = (event) => {
      if (!stream.shouldPlay || !this.context) return;

      const data = event.data;
      if (!(data instanceof ArrayBuffer)) return;

      // Decode PCM16 samples
      const dataView = new DataView(data);
      const sampleCount = Math.floor(data.byteLength / 2);
      for (let i = 0; i < sampleCount; i++) {
        const sample = dataView.getInt16(i * 2, true) / 32768.0;
        stream.pcmBuffer.push(sample);
      }

      // Process buffered audio
      while (stream.pcmBuffer.length >= this.BUFFER_SIZE && stream.shouldPlay) {
        const chunk = stream.pcmBuffer.splice(0, this.BUFFER_SIZE);
        const audioBuffer = this.context.createBuffer(1, chunk.length, this.SAMPLE_RATE);
        const channelData = audioBuffer.getChannelData(0);

        for (let i = 0; i < chunk.length; i++) {
          channelData[i] = chunk[i];
        }

        const source = this.context.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(stream.gainNode);

        const startTime = Math.max(stream.nextStartTime, this.context.currentTime);
        source.start(startTime);
        stream.nextStartTime = startTime + audioBuffer.duration;
      }
    };

    ws.onerror = (error) => {
      console.error("[AudioService] WebSocket error:", channelId, error);
      this.stopChannel(channelId);
    };

    ws.onclose = (event) => {
      console.log("[AudioService] WebSocket closed:", channelId, event.code);
      if (stream.shouldPlay) {
        // Unexpected close - clean up
        this.stopChannel(channelId);
      }
    };
    } catch (error) {
      console.error("[AudioService] Error starting playback:", channelId, error);
      throw error;
    }
  }

  /**
   * Stop playing audio for a specific channel.
   */
  stop(channelId: string): void {
    this.stopChannel(channelId);
    this.notifyListeners();
  }

  private stopChannel(channelId: string): void {
    const stream = this.channels.get(channelId);
    if (stream) {
      stream.shouldPlay = false;
      stream.ws.close();
      stream.gainNode.disconnect();
      this.channels.delete(channelId);
      this.updateChannelGains();
      this.notifyListeners();
    }
  }

  /**
   * Stop all channels.
   */
  stopAll(): void {
    const channelIds = Array.from(this.channels.keys());
    channelIds.forEach((id) => this.stopChannel(id));
    this.notifyListeners();
  }

  /**
   * Check if a channel is currently playing.
   */
  isPlaying(channelId: string): boolean {
    return this.channels.has(channelId);
  }

  /**
   * Set master volume (0.0 - 1.0).
   */
  setMasterVolume(volume: number): void {
    this._masterVolume = Math.max(0, Math.min(1, volume));
    if (this.masterGain) {
      this.masterGain.gain.value = this._masterVolume;
    }
    this.notifyListeners();
  }

  /**
   * Get current master volume.
   */
  getMasterVolume(): number {
    return this._masterVolume;
  }

  /**
   * Update individual channel gains for proper mixing.
   * Uses 1/sqrt(n) normalization to prevent clipping when mixing multiple channels.
   */
  private updateChannelGains(): void {
    const numChannels = this.channels.size;
    if (numChannels === 0) return;

    const mixGain = 1.0 / Math.sqrt(numChannels);
    this.channels.forEach((stream) => {
      stream.gainNode.gain.value = mixGain;
    });
  }

  /**
   * Clean up resources.
   */
  destroy(): void {
    this.stopAll();
    if (this.context) {
      this.context.close();
      this.context = null;
      this.masterGain = null;
    }
    this.listeners.clear();
  }
}

// Singleton instance
export const audioService = new AudioService();
