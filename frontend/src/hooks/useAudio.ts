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

  return {
    playingChannels: state.playingChannels,
    masterVolume: state.masterVolume,
    isContextReady: state.isContextReady,
    play,
    stop,
    stopAll,
    setMasterVolume,
    isPlaying,
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
