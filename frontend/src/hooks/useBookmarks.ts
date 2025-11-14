import { useState, useEffect } from 'react';

export interface Bookmark {
  id: string;
  name: string;
  frequency: number;
  mode?: string;
  notes?: string;
  createdAt: number;
}

const STORAGE_KEY = 'wavecap_frequency_bookmarks';

export function useBookmarks() {
  const [bookmarks, setBookmarks] = useState<Bookmark[]>(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      return stored ? JSON.parse(stored) : [];
    } catch (error) {
      console.error('Failed to load bookmarks:', error);
      return [];
    }
  });

  // Persist to localStorage whenever bookmarks change
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(bookmarks));
    } catch (error) {
      console.error('Failed to save bookmarks:', error);
    }
  }, [bookmarks]);

  const addBookmark = (bookmark: Omit<Bookmark, 'id' | 'createdAt'>) => {
    const newBookmark: Bookmark = {
      ...bookmark,
      id: `bookmark_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      createdAt: Date.now(),
    };
    setBookmarks((prev) => [...prev, newBookmark]);
    return newBookmark;
  };

  const updateBookmark = (id: string, updates: Partial<Omit<Bookmark, 'id' | 'createdAt'>>) => {
    setBookmarks((prev) =>
      prev.map((b) => (b.id === id ? { ...b, ...updates } : b))
    );
  };

  const deleteBookmark = (id: string) => {
    setBookmarks((prev) => prev.filter((b) => b.id !== id));
  };

  const getBookmarkByFrequency = (frequency: number, tolerance: number = 1000) => {
    return bookmarks.find((b) => Math.abs(b.frequency - frequency) < tolerance);
  };

  return {
    bookmarks,
    addBookmark,
    updateBookmark,
    deleteBookmark,
    getBookmarkByFrequency,
  };
}
