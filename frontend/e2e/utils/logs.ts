import type { Page, ConsoleMessage } from "@playwright/test";
import { expect } from "@playwright/test";

/**
 * Console log capture and validation utilities for e2e tests.
 *
 * These helpers capture browser console output and validate that
 * no unexpected errors or warnings occur during tests.
 */

/** Captured console log entry */
export interface LogEntry {
  type: "log" | "info" | "warn" | "error" | "debug";
  text: string;
  location?: string;
  timestamp: number;
}

/** Log capture result with helper methods */
export interface LogCapture {
  logs: LogEntry[];
  errors: LogEntry[];
  warnings: LogEntry[];
  stop: () => void;
}

/**
 * Start capturing console logs from a page.
 * Call stop() when done to clean up the listener.
 *
 * @param page - Playwright page object
 * @returns LogCapture object with logs and helper methods
 */
export function captureConsoleLogs(page: Page): LogCapture {
  const logs: LogEntry[] = [];

  const listener = (msg: ConsoleMessage) => {
    const entry: LogEntry = {
      type: msg.type() as LogEntry["type"],
      text: msg.text(),
      location: msg.location().url,
      timestamp: Date.now(),
    };
    logs.push(entry);
  };

  page.on("console", listener);

  return {
    get logs() {
      return logs;
    },
    get errors() {
      return logs.filter((l) => l.type === "error");
    },
    get warnings() {
      return logs.filter((l) => l.type === "warn");
    },
    stop() {
      page.removeListener("console", listener);
    },
  };
}

/**
 * Default patterns that are safe to ignore in console output.
 * These are common browser/framework messages that don't indicate bugs.
 *
 * Note: Visual issues (overflow, font-size, viewport-clip, overlap) are
 * excluded from general warning checks and handled by assertNoVisualIssues().
 */
export const DEFAULT_WARNING_ALLOWLIST = [
  // React development mode warnings
  /Download the React DevTools/i,
  /Warning: ReactDOM.render is no longer supported/i,
  // Vite HMR messages
  /\[vite\]/i,
  // Browser extensions
  /extension/i,
  // Service worker lifecycle
  /service worker/i,
  // WebSocket connection messages (expected during reconnect)
  /WebSocket connection/i,
  /WebSocket is already in CLOSING or CLOSED state/i,
  // Visual issues - handled by assertNoVisualIssues() instead
  /Visual overflow detected:/i,
  /Font size too small for screen type/i,
  /Viewport boundary clipping/i,
  /Sibling element overlap/i,
  /Screen real-estate underutilized:/i,
  /Positioned element clipped/i,
];

export const DEFAULT_ERROR_ALLOWLIST = [
  // Network errors that may occur during test setup/teardown
  /Failed to load resource.*favicon/i,
  // WebSocket errors during intentional disconnection
  /WebSocket.*closed/i,
  // Audio context errors (expected when audio isn't playing)
  /AudioContext/i,
  // FFT/spectrum errors when capture is stopped
  /FFT data unavailable/i,
];

/**
 * Patterns that match visual issue warnings from the overflow detector.
 * These indicate UI problems that should fail tests.
 */
export const VISUAL_ISSUE_PATTERNS = [
  // Overflow detection - content clipped by overflow:hidden
  /Visual overflow detected:/i,
  // Font size too small for screen type
  /Font size too small for screen type/i,
  // Element extends beyond viewport boundaries
  /Viewport boundary clipping/i,
  // Sibling elements overlapping each other
  /Sibling element overlap/i,
  // Screen using too little of available viewport
  /Screen real-estate underutilized:/i,
  // Positioned element clipped by overflow-hidden ancestor
  /Positioned element clipped/i,
];

/**
 * Assert that no console errors occurred (except allowed ones).
 *
 * @param capture - LogCapture from captureConsoleLogs
 * @param allowlist - Patterns to ignore (defaults to common safe errors)
 */
export function assertNoErrors(
  capture: LogCapture,
  allowlist: RegExp[] = DEFAULT_ERROR_ALLOWLIST,
): void {
  const unexpectedErrors = capture.errors.filter(
    (error) =>
      !allowlist.some(
        (pattern) =>
          pattern.test(error.text) ||
          (error.location && pattern.test(error.location)),
      ),
  );

  expect(unexpectedErrors, "Unexpected console errors").toHaveLength(0);
}

/**
 * Assert that no console warnings occurred (except allowed ones).
 *
 * @param capture - LogCapture from captureConsoleLogs
 * @param allowlist - Patterns to ignore (defaults to common safe warnings)
 */
export function assertNoWarnings(
  capture: LogCapture,
  allowlist: RegExp[] = DEFAULT_WARNING_ALLOWLIST,
): void {
  const unexpectedWarnings = capture.warnings.filter(
    (warning) => !allowlist.some((pattern) => pattern.test(warning.text)),
  );

  expect(unexpectedWarnings, "Unexpected console warnings").toHaveLength(0);
}

/**
 * Assert that neither errors nor warnings occurred.
 * Convenience function combining assertNoErrors and assertNoWarnings.
 *
 * @param capture - LogCapture from captureConsoleLogs
 * @param errorAllowlist - Error patterns to ignore
 * @param warningAllowlist - Warning patterns to ignore
 */
export function assertCleanConsole(
  capture: LogCapture,
  errorAllowlist: RegExp[] = DEFAULT_ERROR_ALLOWLIST,
  warningAllowlist: RegExp[] = DEFAULT_WARNING_ALLOWLIST,
): void {
  assertNoErrors(capture, errorAllowlist);
  assertNoWarnings(capture, warningAllowlist);
}

/**
 * Assert that no visual issues were detected by the overflow detector.
 * This checks for font size warnings, overflow clipping, viewport boundary
 * issues, and element overlaps.
 *
 * @param capture - LogCapture from captureConsoleLogs
 * @throws AssertionError if any visual issues are detected
 */
export function assertNoVisualIssues(capture: LogCapture): void {
  const visualIssues = capture.warnings.filter((warning) =>
    VISUAL_ISSUE_PATTERNS.some((pattern) => pattern.test(warning.text)),
  );

  if (visualIssues.length > 0) {
    const issueDetails = visualIssues
      .map((issue) => `  - ${issue.text}`)
      .join("\n");

    expect(
      visualIssues,
      `Visual issues detected:\n${issueDetails}`,
    ).toHaveLength(0);
  }
}

/**
 * Get all visual issues from captured logs.
 * Useful for debugging or custom handling.
 *
 * @param capture - LogCapture from captureConsoleLogs
 * @returns Array of visual issue log entries
 */
export function getVisualIssues(capture: LogCapture): LogEntry[] {
  return capture.warnings.filter((warning) =>
    VISUAL_ISSUE_PATTERNS.some((pattern) => pattern.test(warning.text)),
  );
}

/**
 * Assert clean console including visual issue checks.
 * This is the strictest validation - fails on errors, warnings, and visual issues.
 *
 * @param capture - LogCapture from captureConsoleLogs
 * @param errorAllowlist - Error patterns to ignore
 * @param warningAllowlist - Warning patterns to ignore (visual issues are never ignored)
 */
export function assertCleanConsoleStrict(
  capture: LogCapture,
  errorAllowlist: RegExp[] = DEFAULT_ERROR_ALLOWLIST,
  warningAllowlist: RegExp[] = DEFAULT_WARNING_ALLOWLIST,
): void {
  assertNoErrors(capture, errorAllowlist);
  assertNoWarnings(capture, warningAllowlist);
  assertNoVisualIssues(capture);
}

/**
 * Find log entries matching a pattern.
 * Useful for verifying expected log messages.
 *
 * @param capture - LogCapture from captureConsoleLogs
 * @param pattern - Pattern to search for
 * @returns Matching log entries
 */
export function findLogs(capture: LogCapture, pattern: RegExp): LogEntry[] {
  return capture.logs.filter((log) => pattern.test(log.text));
}

/**
 * Wait for a specific log message to appear.
 *
 * @param capture - LogCapture from captureConsoleLogs
 * @param pattern - Pattern to wait for
 * @param timeout - Maximum time to wait in milliseconds
 */
export async function waitForLog(
  capture: LogCapture,
  pattern: RegExp,
  timeout = 5000,
): Promise<LogEntry> {
  const startTime = Date.now();

  while (Date.now() - startTime < timeout) {
    const match = capture.logs.find((log) => pattern.test(log.text));
    if (match) {
      return match;
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }

  throw new Error(`Log matching ${pattern} not found within ${timeout}ms`);
}

/**
 * Print captured logs for debugging purposes.
 *
 * @param capture - LogCapture from captureConsoleLogs
 * @param filter - Optional filter by log type
 */
export function printLogs(
  capture: LogCapture,
  filter?: LogEntry["type"],
): void {
  const logsToPrint = filter
    ? capture.logs.filter((l) => l.type === filter)
    : capture.logs;

  for (const log of logsToPrint) {
    console.log(`[${log.type.toUpperCase()}] ${log.text}`);
  }
}
