import {
  test as base,
  expect,
  type Page,
  type BrowserContext,
} from "@playwright/test";
import {
  captureConsoleLogs,
  assertNoVisualIssues,
  type LogCapture,
} from "./utils/logs";

/**
 * Shared Playwright fixtures for WaveCap-SDR e2e tests.
 *
 * Provides pre-configured page contexts for desktop and responsive testing
 * with automatic log capture and visual issue detection.
 *
 * Visual issue detection is automatic - tests will fail if any
 * overflow, font-size, viewport-clip, or overlap issues are detected.
 */

/** Viewport dimensions for different device types */
const VIEWPORTS = {
  desktop: { width: 1920, height: 1080 },
  laptop: { width: 1440, height: 900 },
  tablet: { width: 1024, height: 768 },
} as const;

/** Extended test fixtures available in tests */
export interface WaveCapFixtures {
  /** Desktop browser page for main dashboard */
  dashboardPage: Page;
  /** Create additional pages for multi-tab tests */
  createPage: (viewport?: keyof typeof VIEWPORTS) => Promise<Page>;
  /** Log capture for dashboard page */
  dashboardLogs: LogCapture;
  /** Desktop browser context */
  desktopContext: BrowserContext;
}

/**
 * Extended test function with WaveCap-SDR fixtures.
 *
 * Usage:
 * ```ts
 * import { test, expect } from '../fixtures';
 *
 * test('my test', async ({ dashboardPage, dashboardLogs }) => {
 *   await dashboardPage.goto('/');
 *   // Visual issues are checked automatically after test
 * });
 * ```
 */
export const test = base.extend<WaveCapFixtures>({
  // Desktop browser context
  desktopContext: async ({ browser }, use) => {
    const context = await browser.newContext({
      viewport: VIEWPORTS.desktop,
    });
    await use(context);
    await context.close();
  },

  // Main dashboard page with desktop viewport
  dashboardPage: async ({ browser }, use) => {
    const context = await browser.newContext({
      viewport: VIEWPORTS.desktop,
    });
    const page = await context.newPage();
    await use(page);
    await context.close();
  },

  // Factory for creating additional pages with configurable viewports
  createPage: async ({ browser }, use) => {
    const contexts: BrowserContext[] = [];

    const createPage = async (viewport: keyof typeof VIEWPORTS = "desktop") => {
      const context = await browser.newContext({
        viewport: VIEWPORTS[viewport],
      });
      contexts.push(context);
      const page = await context.newPage();
      return page;
    };

    await use(createPage);

    // Cleanup: close all created contexts
    for (const context of contexts) {
      await context.close();
    }
  },

  // Automatic log capture for dashboard page with visual issue detection
  dashboardLogs: async ({ dashboardPage }, use) => {
    const capture = captureConsoleLogs(dashboardPage);
    await use(capture);
    // After test completes, check for visual issues
    assertNoVisualIssues(capture);
    capture.stop();
  },
});

// Re-export expect for convenience
export { expect };
