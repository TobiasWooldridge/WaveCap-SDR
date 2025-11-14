import { useState, useEffect, useCallback } from 'react';
import type { Capture, Channel } from '../types';

export interface MemoryBank {
  id: string;
  name: string;
  timestamp: number;
  captureConfig: {
    centerHz: number;
    sampleRate: number;
    gain: number | null;
    bandwidth: number | null;
    ppm: number | null;
    antenna: string | null;
    deviceId?: string;
  };
  channels: Array<{
    mode: Channel['mode'];
    offsetHz: number;
    audioRate: number;
    squelchDb: number | null;
    name: string | null;
  }>;
}

const MEMORY_BANKS_KEY = 'wavecapsdr_memory_banks';
const MAX_MEMORY_BANKS = 50;

export function useMemoryBanks() {
  const [memoryBanks, setMemoryBanks] = useState<MemoryBank[]>([]);

  // Load memory banks from localStorage on mount
  useEffect(() => {
    try {
      const stored = localStorage.getItem(MEMORY_BANKS_KEY);
      if (stored) {
        const parsed = JSON.parse(stored) as MemoryBank[];
        setMemoryBanks(parsed);
      }
    } catch (error) {
      console.error('Failed to load memory banks:', error);
    }
  }, []);

  // Save memory banks to localStorage
  const saveMemoryBanks = useCallback((banks: MemoryBank[]) => {
    try {
      localStorage.setItem(MEMORY_BANKS_KEY, JSON.stringify(banks));
      setMemoryBanks(banks);
    } catch (error) {
      console.error('Failed to save memory banks:', error);
    }
  }, []);

  // Save current capture configuration to a memory bank
  const saveToMemoryBank = useCallback((
    name: string,
    capture: Capture,
    channels: Channel[]
  ): MemoryBank => {
    const newBank: MemoryBank = {
      id: `mb_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`,
      name,
      timestamp: Date.now(),
      captureConfig: {
        centerHz: capture.centerHz,
        sampleRate: capture.sampleRate,
        gain: capture.gain,
        bandwidth: capture.bandwidth,
        ppm: capture.ppm,
        antenna: capture.antenna,
        deviceId: capture.deviceId,
      },
      channels: channels.map(ch => ({
        mode: ch.mode,
        offsetHz: ch.offsetHz,
        audioRate: ch.audioRate,
        squelchDb: ch.squelchDb,
        name: ch.name,
      })),
    };

    // Add to beginning and limit to MAX_MEMORY_BANKS
    const newBanks = [newBank, ...memoryBanks].slice(0, MAX_MEMORY_BANKS);
    saveMemoryBanks(newBanks);
    return newBank;
  }, [memoryBanks, saveMemoryBanks]);

  // Delete a memory bank
  const deleteMemoryBank = useCallback((id: string) => {
    const newBanks = memoryBanks.filter(bank => bank.id !== id);
    saveMemoryBanks(newBanks);
  }, [memoryBanks, saveMemoryBanks]);

  // Update a memory bank name
  const renameMemoryBank = useCallback((id: string, newName: string) => {
    const newBanks = memoryBanks.map(bank =>
      bank.id === id ? { ...bank, name: newName } : bank
    );
    saveMemoryBanks(newBanks);
  }, [memoryBanks, saveMemoryBanks]);

  // Get a specific memory bank
  const getMemoryBank = useCallback((id: string): MemoryBank | undefined => {
    return memoryBanks.find(bank => bank.id === id);
  }, [memoryBanks]);

  // Clear all memory banks
  const clearMemoryBanks = useCallback(() => {
    try {
      localStorage.removeItem(MEMORY_BANKS_KEY);
      setMemoryBanks([]);
    } catch (error) {
      console.error('Failed to clear memory banks:', error);
    }
  }, []);

  return {
    memoryBanks,
    saveToMemoryBank,
    deleteMemoryBank,
    renameMemoryBank,
    getMemoryBank,
    clearMemoryBanks,
  };
}
