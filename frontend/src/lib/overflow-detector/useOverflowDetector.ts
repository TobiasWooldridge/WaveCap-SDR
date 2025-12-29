import { useEffect, useRef } from 'react';
import { OverflowDetector } from './detector';
import type { OverflowDetectorConfig, OverflowIssue } from './types';

// Simple logger that sends to console
const defaultLogger = {
  warn: (message: string, data?: unknown) => {
    console.warn(`[OverflowDetector] ${message}`, data);
  },
};

/**
 * React hook for detecting visual overflow issues in the DOM
 *
 * @param enabled - Whether detection is enabled (typically tied to dev mode or URL param)
 * @param config - Additional configuration options
 *
 * @example
 * ```tsx
 * function App() {
 *   // Detection runs when ?debugOverflow=true is in URL
 *   useOverflowDetector();
 *
 *   return <div>...</div>;
 * }
 * ```
 */
export function useOverflowDetector(
  enabled = false,
  config?: Omit<OverflowDetectorConfig, 'enabled'>
): void {
  const detectorRef = useRef<OverflowDetector | null>(null);

  useEffect(() => {
    // Check URL param or explicit enable
    const shouldRun =
      enabled ||
      (typeof window !== 'undefined' &&
        new URLSearchParams(window.location.search).get('debugOverflow') === 'true');

    if (!shouldRun) {
      return;
    }

    // Create and start detector
    detectorRef.current = new OverflowDetector(defaultLogger, {
      ...config,
      enabled: true,
    });
    detectorRef.current.start();

    // Log that detector is active
    console.info('[OverflowDetector] Started - monitoring for visual issues');

    // Cleanup on unmount or when disabled
    return () => {
      detectorRef.current?.stop();
      detectorRef.current = null;
    };
  }, [enabled, config]);
}

/**
 * React hook for overflow detection with custom logger
 *
 * Use this when you want to integrate with the app's existing logging system
 */
export function useOverflowDetectorWithLogger(
  logger: { warn: (message: string, data?: unknown) => void },
  enabled = false,
  config?: Omit<OverflowDetectorConfig, 'enabled'>
): void {
  const detectorRef = useRef<OverflowDetector | null>(null);

  useEffect(() => {
    const shouldRun =
      enabled ||
      (typeof window !== 'undefined' &&
        new URLSearchParams(window.location.search).get('debugOverflow') === 'true');

    if (!shouldRun) {
      return;
    }

    detectorRef.current = new OverflowDetector(logger, {
      ...config,
      enabled: true,
    });
    detectorRef.current.start();

    return () => {
      detectorRef.current?.stop();
      detectorRef.current = null;
    };
  }, [logger, enabled, config]);
}

/**
 * Get issues detected during the current session
 * Useful for displaying in dev overlays
 */
export function useOverflowIssues(
  enabled = false,
  config?: Omit<OverflowDetectorConfig, 'enabled'>
): OverflowIssue[] {
  const issuesRef = useRef<OverflowIssue[]>([]);
  const detectorRef = useRef<OverflowDetector | null>(null);

  useEffect(() => {
    if (!enabled) {
      return;
    }

    const customLogger = {
      warn: (message: string, data?: unknown) => {
        const issueData = data as { issue?: OverflowIssue } | undefined;
        if (issueData?.issue) {
          issuesRef.current = [...issuesRef.current.slice(-99), issueData.issue];
        }
        console.warn(`[OverflowDetector] ${message}`, data);
      },
    };

    detectorRef.current = new OverflowDetector(customLogger, {
      ...config,
      enabled: true,
    });
    detectorRef.current.start();

    return () => {
      detectorRef.current?.stop();
      detectorRef.current = null;
      issuesRef.current = [];
    };
  }, [enabled, config]);

  return issuesRef.current;
}
