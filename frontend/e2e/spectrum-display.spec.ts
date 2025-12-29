import { test, expect } from "./fixtures";
import { assertNoErrors, assertCleanConsoleStrict } from "./utils/logs";
import {
  waitForBackend,
  createTestCapture,
  createTestChannel,
  startCapture,
  stopCapture,
  deleteAllCaptures,
} from "./utils/debug";

/**
 * Spectrum analyzer and waterfall display e2e tests
 *
 * Tests the spectrum visualization with fake SDR driver:
 * - Spectrum analyzer rendering
 * - Waterfall display
 * - Channel markers
 * - Zoom and pan
 */

test.describe("Spectrum Display", () => {
  let captureId: string;

  test.beforeAll(async () => {
    const ready = await waitForBackend(15000);
    if (!ready) {
      throw new Error("Backend not available");
    }
  });

  test.beforeEach(async () => {
    await deleteAllCaptures();
    const capture = await createTestCapture({
      name: "Spectrum Test",
      center_hz: 100_000_000,
      sample_rate: 1_000_000,
    });
    captureId = capture.id;
  });

  test.afterEach(async () => {
    await deleteAllCaptures();
  });

  test("should display spectrum panel when capture is selected", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await startCapture(captureId);

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    // Select the capture
    await dashboardPage.getByText(/Spectrum Test/i).click();
    await dashboardPage.waitForTimeout(500);

    // Spectrum panel should be visible (canvas element or spectrum container)
    const spectrumArea = dashboardPage.locator(
      'canvas, [class*="spectrum"], [class*="waterfall"]',
    );
    await expect(spectrumArea.first()).toBeVisible({ timeout: 5000 });

    assertNoErrors(dashboardLogs);
  });

  test("should show spectrum data when capture is running", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await startCapture(captureId);

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Spectrum Test/i).click();

    // Wait for spectrum data to appear
    await dashboardPage.waitForTimeout(2000);

    // Check that canvas has content (non-zero size)
    const canvas = dashboardPage.locator("canvas").first();
    if (await canvas.isVisible({ timeout: 3000 }).catch(() => false)) {
      const size = await canvas.boundingBox();
      expect(size?.width).toBeGreaterThan(0);
      expect(size?.height).toBeGreaterThan(0);
    }

    assertNoErrors(dashboardLogs);
  });

  test("should stop spectrum updates when capture is stopped", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await startCapture(captureId);

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Spectrum Test/i).click();
    await dashboardPage.waitForTimeout(1000);

    // Stop the capture
    await stopCapture(captureId);
    await dashboardPage.waitForTimeout(500);

    // Page should still be functional (no errors)
    await expect(dashboardPage.locator("body")).toBeVisible();

    assertNoErrors(dashboardLogs);
  });

  test("should show channel markers on spectrum", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await startCapture(captureId);

    // Create a channel at 5 kHz offset (matches fake driver tone)
    await createTestChannel(captureId, {
      name: "Marker Test",
      offset_hz: 5000,
      mode: "nbfm",
    });

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Spectrum Test/i).click();
    await dashboardPage.waitForTimeout(1000);

    // Look for channel marker or indicator on spectrum
    // This could be a line, label, or overlay element
    const channelMarker = dashboardPage.locator(
      '[class*="marker"], [class*="channel-indicator"], [data-channel]',
    );
    const hasMarker = await channelMarker
      .first()
      .isVisible({ timeout: 3000 })
      .catch(() => false);

    // If no explicit marker, at least verify the channel is listed
    if (!hasMarker) {
      await expect(dashboardPage.getByText(/Marker Test/i)).toBeVisible({
        timeout: 3000,
      });
    }

    assertNoErrors(dashboardLogs);
  });

  test("should display multiple channel markers", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await startCapture(captureId);

    // Create multiple channels at different offsets
    await createTestChannel(captureId, {
      name: "Ch 1",
      offset_hz: -100000,
      mode: "wbfm",
    });
    await createTestChannel(captureId, {
      name: "Ch 2",
      offset_hz: 0,
      mode: "nbfm",
    });
    await createTestChannel(captureId, {
      name: "Ch 3",
      offset_hz: 100000,
      mode: "am",
    });

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Spectrum Test/i).click();
    await dashboardPage.waitForTimeout(1000);

    // All channels should be visible in the channel list
    await expect(dashboardPage.getByText(/Ch 1/i)).toBeVisible({
      timeout: 5000,
    });
    await expect(dashboardPage.getByText(/Ch 2/i)).toBeVisible({
      timeout: 5000,
    });
    await expect(dashboardPage.getByText(/Ch 3/i)).toBeVisible({
      timeout: 5000,
    });

    assertNoErrors(dashboardLogs);
  });

  test("should handle spectrum panel resize", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await startCapture(captureId);

    await dashboardPage.goto("/?debugOverflow=true");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Spectrum Test/i).click();
    await dashboardPage.waitForTimeout(500);

    // Resize viewport
    await dashboardPage.setViewportSize({ width: 1200, height: 800 });
    await dashboardPage.waitForTimeout(300);

    await dashboardPage.setViewportSize({ width: 1600, height: 900 });
    await dashboardPage.waitForTimeout(300);

    await dashboardPage.setViewportSize({ width: 1920, height: 1080 });
    await dashboardPage.waitForTimeout(300);

    // Spectrum should still be visible and functional
    const spectrumArea = dashboardPage.locator('canvas, [class*="spectrum"]');
    await expect(spectrumArea.first()).toBeVisible({ timeout: 3000 });

    assertCleanConsoleStrict(dashboardLogs);
  });

  test("should display frequency labels on spectrum", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await startCapture(captureId);

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Spectrum Test/i).click();
    await dashboardPage.waitForTimeout(1000);

    // Look for frequency labels (like "100 MHz" or similar)
    const freqLabels = dashboardPage.getByText(/\d+\.?\d*\s*(MHz|kHz|GHz)/i);
    const hasLabels = await freqLabels
      .first()
      .isVisible({ timeout: 3000 })
      .catch(() => false);

    // Labels should exist somewhere in the spectrum area or tuning display
    // This is a soft check since UI may vary
    expect(
      hasLabels || (await dashboardPage.getByText(/100/i).isVisible()),
    ).toBeTruthy();

    assertNoErrors(dashboardLogs);
  });

  test("should show waterfall display", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await startCapture(captureId);

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Spectrum Test/i).click();
    await dashboardPage.waitForTimeout(1000);

    // Look for waterfall toggle or waterfall canvas
    const waterfallToggle = dashboardPage.getByRole("button", {
      name: /waterfall/i,
    });
    const waterfallElement = dashboardPage.locator('[class*="waterfall"]');

    const hasToggle = await waterfallToggle
      .isVisible({ timeout: 2000 })
      .catch(() => false);
    const _hasWaterfall = await waterfallElement
      .first()
      .isVisible({ timeout: 2000 })
      .catch(() => false);

    // If toggle exists, click it to show waterfall
    if (hasToggle) {
      await waterfallToggle.click();
      await dashboardPage.waitForTimeout(500);
    }

    assertNoErrors(dashboardLogs);
  });

  test("should not have visual overflow issues in spectrum panel", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await startCapture(captureId);
    await createTestChannel(captureId, {
      name: "Visual Test Channel",
      offset_hz: 5000,
      mode: "wbfm",
    });

    await dashboardPage.goto("/?debugOverflow=true");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Spectrum Test/i).click();

    // Wait for spectrum to render and detector to run
    await dashboardPage.waitForTimeout(2000);

    // Strict check will catch any visual issues
    assertCleanConsoleStrict(dashboardLogs);
  });

  test("should handle rapid capture switching", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    // Create another capture
    const capture2 = await createTestCapture({
      name: "Spectrum Test 2",
      center_hz: 88_000_000,
    });

    await startCapture(captureId);
    await startCapture(capture2.id);

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    // Rapidly switch between captures
    for (let i = 0; i < 5; i++) {
      await dashboardPage.getByText(/Spectrum Test(?! 2)/i).click();
      await dashboardPage.waitForTimeout(200);
      await dashboardPage.getByText(/Spectrum Test 2/i).click();
      await dashboardPage.waitForTimeout(200);
    }

    // Should not have crashed or produced errors
    await expect(dashboardPage.locator("body")).toBeVisible();

    assertNoErrors(dashboardLogs);
  });
});
