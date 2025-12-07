import { useState, useEffect, useRef, useCallback } from "react";

/**
 * A hook that combines local state with debounced server mutations.
 *
 * This eliminates the common anti-pattern of:
 * 1. Local state mirroring server state
 * 2. useDebounce hook
 * 3. useEffect to sync server -> local
 * 4. useEffect to trigger mutation when debounced value changes
 *
 * Instead, this hook provides:
 * - Immediate local state updates for responsive UI
 * - Debounced mutations to avoid excessive API calls
 * - Automatic sync when server value changes from external sources
 * - Pending state indicator
 *
 * @param serverValue - The current value from the server (e.g., from React Query)
 * @param mutate - Function to call with the new value after debounce
 * @param options.delay - Debounce delay in milliseconds (default: 150)
 * @param options.isEqual - Custom equality function (default: ===)
 *
 * @returns [localValue, setLocalValue, isPending]
 */
export function useDebouncedMutation<T>(
  serverValue: T,
  mutate: (value: T) => void,
  options: {
    delay?: number;
    isEqual?: (a: T, b: T) => boolean;
  } = {}
): [T, (value: T) => void, boolean] {
  const { delay = 150, isEqual = (a: T, b: T) => a === b } = options;

  const [localValue, setLocalValue] = useState<T>(serverValue);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Track the last value we sent to the server to avoid resetting from our own mutation
  const pendingMutationRef = useRef<T | null>(null);

  // Track if we have a pending mutation
  const [isPending, setIsPending] = useState(false);

  // Sync from server when it changes externally (not from our own mutation)
  useEffect(() => {
    // If we have a pending mutation and server caught up, clear it
    if (pendingMutationRef.current !== null && isEqual(serverValue, pendingMutationRef.current)) {
      pendingMutationRef.current = null;
      setIsPending(false);
      return;
    }

    // If server changed and we don't have a pending mutation, sync local
    if (pendingMutationRef.current === null && !isEqual(localValue, serverValue)) {
      setLocalValue(serverValue);
    }
  }, [serverValue, isEqual, localValue]);

  // Debounced setter
  const setValue = useCallback(
    (newValue: T) => {
      setLocalValue(newValue);
      setIsPending(true);

      // Clear any existing timeout
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }

      // Set up new debounced mutation
      timeoutRef.current = setTimeout(() => {
        if (!isEqual(newValue, serverValue)) {
          pendingMutationRef.current = newValue;
          mutate(newValue);
        } else {
          // Value matches server, no mutation needed
          setIsPending(false);
        }
        timeoutRef.current = null;
      }, delay);
    },
    [serverValue, mutate, delay, isEqual]
  );

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  return [localValue, setValue, isPending];
}

/**
 * Variant for object values with partial updates.
 * Useful when you want to update individual fields of an object
 * while keeping the debouncing behavior.
 */
export function useDebouncedObjectMutation<T extends Record<string, unknown>>(
  serverValue: T,
  mutate: (value: Partial<T>) => void,
  options: { delay?: number } = {}
): [T, (updates: Partial<T>) => void, boolean] {
  const { delay = 150 } = options;

  const [localValue, setLocalValue] = useState<T>(serverValue);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pendingUpdatesRef = useRef<Partial<T>>({});
  const [isPending, setIsPending] = useState(false);

  // Sync from server
  useEffect(() => {
    // Merge server value with any pending local changes
    const hasPendingChanges = Object.keys(pendingUpdatesRef.current).length > 0;
    if (!hasPendingChanges) {
      setLocalValue(serverValue);
    }
  }, [serverValue]);

  const updateValue = useCallback(
    (updates: Partial<T>) => {
      setLocalValue((prev) => ({ ...prev, ...updates }));
      pendingUpdatesRef.current = { ...pendingUpdatesRef.current, ...updates };
      setIsPending(true);

      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }

      timeoutRef.current = setTimeout(() => {
        const pending = pendingUpdatesRef.current;
        pendingUpdatesRef.current = {};

        if (Object.keys(pending).length > 0) {
          mutate(pending);
        }
        timeoutRef.current = null;
      }, delay);
    },
    [mutate, delay]
  );

  // Clear pending state when server catches up
  useEffect(() => {
    const pending = pendingUpdatesRef.current;
    if (Object.keys(pending).length === 0) {
      setIsPending(false);
    }
  }, [serverValue]);

  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  return [localValue, updateValue, isPending];
}
