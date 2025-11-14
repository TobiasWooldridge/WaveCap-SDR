import { useState, useEffect, useCallback } from 'react';

export interface FrequencyHistoryEntry {
  frequencyHz: number;
  timestamp: number;
  captureName?: string;
  mode?: string;
}

const HISTORY_KEY = 'wavecapsdr_frequency_history';
const MAX_HISTORY_ENTRIES = 100;

export function useFrequencyHistory() {
  const [history, setHistory] = useState<FrequencyHistoryEntry[]>([]);

  // Load history from localStorage on mount
  useEffect(() => {
    try {
      const stored = localStorage.getItem(HISTORY_KEY);
      if (stored) {
        const parsed = JSON.parse(stored) as FrequencyHistoryEntry[];
        setHistory(parsed);
      }
    } catch (error) {
      console.error('Failed to load frequency history:', error);
    }
  }, []);

  // Save history to localStorage whenever it changes
  const saveHistory = useCallback((newHistory: FrequencyHistoryEntry[]) => {
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(newHistory));
      setHistory(newHistory);
    } catch (error) {
      console.error('Failed to save frequency history:', error);
    }
  }, []);

  // Add a frequency to history
  const addToHistory = useCallback((entry: Omit<FrequencyHistoryEntry, 'timestamp'>) => {
    const newEntry: FrequencyHistoryEntry = {
      ...entry,
      timestamp: Date.now(),
    };

    // Remove duplicates (same frequency within last 5 minutes)
    const fiveMinutesAgo = Date.now() - (5 * 60 * 1000);
    const filteredHistory = history.filter(h =>
      h.frequencyHz !== entry.frequencyHz || h.timestamp < fiveMinutesAgo
    );

    // Add new entry at the beginning and limit to MAX_HISTORY_ENTRIES
    const newHistory = [newEntry, ...filteredHistory].slice(0, MAX_HISTORY_ENTRIES);
    saveHistory(newHistory);
  }, [history, saveHistory]);

  // Clear all history
  const clearHistory = useCallback(() => {
    try {
      localStorage.removeItem(HISTORY_KEY);
      setHistory([]);
    } catch (error) {
      console.error('Failed to clear frequency history:', error);
    }
  }, []);

  // Get recent history (default: last 20 entries)
  const getRecentHistory = useCallback((limit: number = 20): FrequencyHistoryEntry[] => {
    return history.slice(0, limit);
  }, [history]);

  return {
    history,
    addToHistory,
    clearHistory,
    getRecentHistory,
  };
}
