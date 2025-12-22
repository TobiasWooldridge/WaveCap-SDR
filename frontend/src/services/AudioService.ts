/**
 * Centralized audio service for managing WebSocket audio streams.
 *
 * This singleton handles:
 * - AudioContext initialization with Safari/iOS compatibility
 * - Per-channel WebSocket connections for PCM audio streaming
 * - Trunking system voice streams (multiplexed and per-call)
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

interface TrunkingStream {
  ws: WebSocket;
  gainNode: GainNode;
  shouldPlay: boolean;
  nextStartTime: number;
  pcmBuffer: number[];
  systemId: string;
  streamId?: string; // undefined for system-level multiplexed stream
}

export interface AudioServiceState {
  playingChannels: Set<string>;
  playingTrunkingSystems: Set<string>;
  playingTrunkingStreams: Set<string>;
  masterVolume: number;
  isContextReady: boolean;
}

type StateListener = (state: AudioServiceState) => void;

class AudioService {
  private context: AudioContext | null = null;
  private masterGain: GainNode | null = null;
  private channels: Map<string, ChannelStream> = new Map();
  private trunkingSystems: Map<string, TrunkingStream> = new Map(); // systemId -> stream
  private trunkingStreams: Map<string, TrunkingStream> = new Map(); // streamId -> stream
  private listeners: Set<StateListener> = new Set();
  private _masterVolume = 0.7;

  // Cached state for useSyncExternalStore - must return same reference if unchanged
  private _cachedState: AudioServiceState = {
    playingChannels: new Set(),
    playingTrunkingSystems: new Set(),
    playingTrunkingStreams: new Set(),
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
    const newPlayingTrunkingSystems = new Set(this.trunkingSystems.keys());
    const newPlayingTrunkingStreams = new Set(this.trunkingStreams.keys());
    const newIsContextReady = this.context !== null && this.context.state === "running";

    // Check if anything actually changed
    const playingChanged =
      newPlayingChannels.size !== this._cachedState.playingChannels.size ||
      [...newPlayingChannels].some((id) => !this._cachedState.playingChannels.has(id));

    const trunkingSystemsChanged =
      newPlayingTrunkingSystems.size !== this._cachedState.playingTrunkingSystems.size ||
      [...newPlayingTrunkingSystems].some((id) => !this._cachedState.playingTrunkingSystems.has(id));

    const trunkingStreamsChanged =
      newPlayingTrunkingStreams.size !== this._cachedState.playingTrunkingStreams.size ||
      [...newPlayingTrunkingStreams].some((id) => !this._cachedState.playingTrunkingStreams.has(id));

    if (
      playingChanged ||
      trunkingSystemsChanged ||
      trunkingStreamsChanged ||
      this._masterVolume !== this._cachedState.masterVolume ||
      newIsContextReady !== this._cachedState.isContextReady
    ) {
      this._cachedState = {
        playingChannels: newPlayingChannels,
        playingTrunkingSystems: newPlayingTrunkingSystems,
        playingTrunkingStreams: newPlayingTrunkingStreams,
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

  // ============================================================================
  // Trunking System Audio Methods
  // ============================================================================

  /**
   * Start playing all voice audio from a trunking system (multiplexed).
   * Opens a WebSocket connection to receive all active calls.
   */
  async playTrunkingSystem(systemId: string): Promise<void> {
    console.log("[AudioService] playTrunkingSystem() called for:", systemId);

    if (this.trunkingSystems.has(systemId)) {
      console.log("[AudioService] Already playing trunking system:", systemId);
      return;
    }

    try {
      const audioContext = await this.init();
      console.log("[AudioService] AudioContext state:", audioContext.state);

      const trunkingGain = audioContext.createGain();
      trunkingGain.gain.value = 1.0;
      trunkingGain.connect(this.masterGain!);

      const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const wsUrl = `${wsProtocol}//${window.location.host}/api/v1/trunking/stream/${systemId}/voice`;
      console.log("[AudioService] Connecting to trunking WebSocket:", wsUrl);

      const ws = new WebSocket(wsUrl);
      // Trunking sends text messages (JSON with base64 audio)
      ws.binaryType = "arraybuffer";

      const stream: TrunkingStream = {
        ws,
        gainNode: trunkingGain,
        shouldPlay: true,
        nextStartTime: audioContext.currentTime,
        pcmBuffer: [],
        systemId,
      };

      this.trunkingSystems.set(systemId, stream);
      this.updateAllGains();
      this.notifyListeners();

      ws.onopen = () => {
        console.log("[AudioService] Trunking WebSocket connected:", systemId);
      };

      ws.onmessage = (event) => {
        if (!stream.shouldPlay || !this.context) return;
        this.handleTrunkingMessage(stream, event.data);
      };

      ws.onerror = (error) => {
        console.error("[AudioService] Trunking WebSocket error:", systemId, error);
        this.stopTrunkingSystemInternal(systemId);
      };

      ws.onclose = (event) => {
        console.log("[AudioService] Trunking WebSocket closed:", systemId, event.code);
        if (stream.shouldPlay) {
          this.stopTrunkingSystemInternal(systemId);
        }
      };
    } catch (error) {
      console.error("[AudioService] Error starting trunking playback:", systemId, error);
      throw error;
    }
  }

  /**
   * Start playing audio from a specific voice stream.
   */
  async playTrunkingVoiceStream(systemId: string, streamId: string): Promise<void> {
    console.log("[AudioService] playTrunkingVoiceStream() called:", systemId, streamId);

    if (this.trunkingStreams.has(streamId)) {
      console.log("[AudioService] Already playing trunking stream:", streamId);
      return;
    }

    try {
      const audioContext = await this.init();

      const streamGain = audioContext.createGain();
      streamGain.gain.value = 1.0;
      streamGain.connect(this.masterGain!);

      const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const wsUrl = `${wsProtocol}//${window.location.host}/api/v1/trunking/stream/${systemId}/voice/${streamId}`;
      console.log("[AudioService] Connecting to voice stream WebSocket:", wsUrl);

      const ws = new WebSocket(wsUrl);
      ws.binaryType = "arraybuffer";

      const stream: TrunkingStream = {
        ws,
        gainNode: streamGain,
        shouldPlay: true,
        nextStartTime: audioContext.currentTime,
        pcmBuffer: [],
        systemId,
        streamId,
      };

      this.trunkingStreams.set(streamId, stream);
      this.updateAllGains();
      this.notifyListeners();

      ws.onopen = () => {
        console.log("[AudioService] Voice stream WebSocket connected:", streamId);
      };

      ws.onmessage = (event) => {
        if (!stream.shouldPlay || !this.context) return;
        this.handleTrunkingMessage(stream, event.data);
      };

      ws.onerror = (error) => {
        console.error("[AudioService] Voice stream WebSocket error:", streamId, error);
        this.stopTrunkingVoiceStreamInternal(streamId);
      };

      ws.onclose = (event) => {
        console.log("[AudioService] Voice stream WebSocket closed:", streamId, event.code);
        if (stream.shouldPlay) {
          this.stopTrunkingVoiceStreamInternal(streamId);
        }
      };
    } catch (error) {
      console.error("[AudioService] Error starting voice stream:", streamId, error);
      throw error;
    }
  }

  /**
   * Handle incoming trunking WebSocket messages.
   * Messages are JSON with base64-encoded PCM audio.
   */
  private handleTrunkingMessage(stream: TrunkingStream, data: unknown): void {
    if (!this.context) return;

    try {
      // Parse JSON message
      let message: { type: string; audio?: string; sampleRate?: number };
      if (typeof data === "string") {
        message = JSON.parse(data);
      } else if (data instanceof ArrayBuffer) {
        // Sometimes sent as binary containing JSON
        const text = new TextDecoder().decode(data);
        message = JSON.parse(text);
      } else {
        return;
      }

      // Handle stream ended
      if (message.type === "ended") {
        console.log("[AudioService] Voice stream ended:", stream.streamId || stream.systemId);
        if (stream.streamId) {
          this.stopTrunkingVoiceStreamInternal(stream.streamId);
        }
        return;
      }

      // Handle audio
      if (message.type === "audio" && message.audio) {
        // Decode base64 to PCM samples
        const binaryString = atob(message.audio);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
          bytes[i] = binaryString.charCodeAt(i);
        }

        // Convert to float samples (PCM16)
        const int16 = new Int16Array(bytes.buffer);
        for (let i = 0; i < int16.length; i++) {
          stream.pcmBuffer.push(int16[i] / 32768.0);
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
      }
    } catch (e) {
      console.error("[AudioService] Error parsing trunking message:", e);
    }
  }

  /**
   * Stop playing audio from a trunking system.
   */
  stopTrunkingSystem(systemId: string): void {
    this.stopTrunkingSystemInternal(systemId);
    this.notifyListeners();
  }

  private stopTrunkingSystemInternal(systemId: string): void {
    const stream = this.trunkingSystems.get(systemId);
    if (stream) {
      stream.shouldPlay = false;
      stream.ws.close();
      stream.gainNode.disconnect();
      this.trunkingSystems.delete(systemId);
      this.updateAllGains();
      this.notifyListeners();
    }
  }

  /**
   * Stop playing a specific voice stream.
   */
  stopTrunkingVoiceStream(streamId: string): void {
    this.stopTrunkingVoiceStreamInternal(streamId);
    this.notifyListeners();
  }

  private stopTrunkingVoiceStreamInternal(streamId: string): void {
    const stream = this.trunkingStreams.get(streamId);
    if (stream) {
      stream.shouldPlay = false;
      stream.ws.close();
      stream.gainNode.disconnect();
      this.trunkingStreams.delete(streamId);
      this.updateAllGains();
      this.notifyListeners();
    }
  }

  /**
   * Check if a trunking system is currently playing.
   */
  isTrunkingSystemPlaying(systemId: string): boolean {
    return this.trunkingSystems.has(systemId);
  }

  /**
   * Check if a voice stream is currently playing.
   */
  isTrunkingStreamPlaying(streamId: string): boolean {
    return this.trunkingStreams.has(streamId);
  }

  /**
   * Stop all trunking audio.
   */
  stopAllTrunking(): void {
    const systemIds = Array.from(this.trunkingSystems.keys());
    systemIds.forEach((id) => this.stopTrunkingSystemInternal(id));
    const streamIds = Array.from(this.trunkingStreams.keys());
    streamIds.forEach((id) => this.stopTrunkingVoiceStreamInternal(id));
    this.notifyListeners();
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
   * Update all gains across channels, trunking systems, and voice streams.
   */
  private updateAllGains(): void {
    const totalStreams =
      this.channels.size + this.trunkingSystems.size + this.trunkingStreams.size;
    if (totalStreams === 0) return;

    const mixGain = 1.0 / Math.sqrt(totalStreams);
    this.channels.forEach((stream) => {
      stream.gainNode.gain.value = mixGain;
    });
    this.trunkingSystems.forEach((stream) => {
      stream.gainNode.gain.value = mixGain;
    });
    this.trunkingStreams.forEach((stream) => {
      stream.gainNode.gain.value = mixGain;
    });
  }

  /**
   * Clean up resources.
   */
  destroy(): void {
    this.stopAll();
    this.stopAllTrunking();
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
