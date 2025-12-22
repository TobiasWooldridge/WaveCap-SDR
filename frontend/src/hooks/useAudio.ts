import { useCallback, useSyncExternalStore } from "react";
import { audioService } from "../services/AudioService";

/**
 * React hook for audio playback.
 *
 * Provides access to the centralized AudioService with React state integration.
 * Handles WebSocket PCM audio streaming with Safari/iOS compatibility.
 *
 * @example
 * const { playingChannels, play, stop, masterVolume, setMasterVolume } = useAudio();
 *
 * // Toggle playback
 * if (playingChannels.has(channelId)) {
 *   stop(channelId);
 * } else {
 *   await play(channelId);
 * }
 */
export function useAudio() {
  // Subscribe to audio service state using useSyncExternalStore for consistent updates
  const state = useSyncExternalStore(
    (callback) => audioService.subscribe(callback),
    () => audioService.getState(),
    () => audioService.getState()
  );

  // Channel methods
  const play = useCallback(async (channelId: string) => {
    await audioService.play(channelId);
  }, []);

  const stop = useCallback((channelId: string) => {
    audioService.stop(channelId);
  }, []);

  const stopAll = useCallback(() => {
    audioService.stopAll();
  }, []);

  const setMasterVolume = useCallback((volume: number) => {
    audioService.setMasterVolume(volume);
  }, []);

  const isPlaying = useCallback((channelId: string) => {
    return state.playingChannels.has(channelId);
  }, [state.playingChannels]);

  // Trunking system methods
  const playTrunkingSystem = useCallback(async (systemId: string) => {
    await audioService.playTrunkingSystem(systemId);
  }, []);

  const stopTrunkingSystem = useCallback((systemId: string) => {
    audioService.stopTrunkingSystem(systemId);
  }, []);

  const isTrunkingSystemPlaying = useCallback((systemId: string) => {
    return state.playingTrunkingSystems.has(systemId);
  }, [state.playingTrunkingSystems]);

  // Trunking voice stream methods
  const playTrunkingVoiceStream = useCallback(async (systemId: string, streamId: string) => {
    await audioService.playTrunkingVoiceStream(systemId, streamId);
  }, []);

  const stopTrunkingVoiceStream = useCallback((streamId: string) => {
    audioService.stopTrunkingVoiceStream(streamId);
  }, []);

  const isTrunkingStreamPlaying = useCallback((streamId: string) => {
    return state.playingTrunkingStreams.has(streamId);
  }, [state.playingTrunkingStreams]);

  const stopAllTrunking = useCallback(() => {
    audioService.stopAllTrunking();
  }, []);

  return {
    // Channel state
    playingChannels: state.playingChannels,
    // Trunking state
    playingTrunkingSystems: state.playingTrunkingSystems,
    playingTrunkingStreams: state.playingTrunkingStreams,
    // Global state
    masterVolume: state.masterVolume,
    isContextReady: state.isContextReady,
    // Channel methods
    play,
    stop,
    stopAll,
    isPlaying,
    // Trunking methods
    playTrunkingSystem,
    stopTrunkingSystem,
    isTrunkingSystemPlaying,
    playTrunkingVoiceStream,
    stopTrunkingVoiceStream,
    isTrunkingStreamPlaying,
    stopAllTrunking,
    // Volume
    setMasterVolume,
  };
}

/**
 * Hook to track audio state for a specific channel.
 * Useful when you only care about one channel's playback state.
 */
export function useChannelAudio(channelId: string) {
  const { playingChannels, play, stop } = useAudio();

  const isPlaying = playingChannels.has(channelId);

  const toggle = useCallback(async () => {
    if (isPlaying) {
      stop(channelId);
    } else {
      await play(channelId);
    }
  }, [channelId, isPlaying, play, stop]);

  return {
    isPlaying,
    play: () => play(channelId),
    stop: () => stop(channelId),
    toggle,
  };
}

/**
 * Hook to track audio state for a trunking system.
 * Provides play/stop/toggle for the multiplexed voice stream.
 */
export function useTrunkingSystemAudio(systemId: string) {
  const {
    playingTrunkingSystems,
    playTrunkingSystem,
    stopTrunkingSystem,
  } = useAudio();

  const isPlaying = playingTrunkingSystems.has(systemId);

  const toggle = useCallback(async () => {
    if (isPlaying) {
      stopTrunkingSystem(systemId);
    } else {
      await playTrunkingSystem(systemId);
    }
  }, [systemId, isPlaying, playTrunkingSystem, stopTrunkingSystem]);

  return {
    isPlaying,
    play: () => playTrunkingSystem(systemId),
    stop: () => stopTrunkingSystem(systemId),
    toggle,
  };
}

/**
 * Hook to track audio state for a specific trunking voice stream.
 */
export function useTrunkingStreamAudio(systemId: string, streamId: string) {
  const {
    playingTrunkingStreams,
    playTrunkingVoiceStream,
    stopTrunkingVoiceStream,
  } = useAudio();

  const isPlaying = playingTrunkingStreams.has(streamId);

  const toggle = useCallback(async () => {
    if (isPlaying) {
      stopTrunkingVoiceStream(streamId);
    } else {
      await playTrunkingVoiceStream(systemId, streamId);
    }
  }, [systemId, streamId, isPlaying, playTrunkingVoiceStream, stopTrunkingVoiceStream]);

  return {
    isPlaying,
    play: () => playTrunkingVoiceStream(systemId, streamId),
    stop: () => stopTrunkingVoiceStream(streamId),
    toggle,
  };
}
