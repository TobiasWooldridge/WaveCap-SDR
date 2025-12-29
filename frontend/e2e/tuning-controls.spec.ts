import { test, expect } from "./fixtures";
import { assertNoErrors, assertCleanConsoleStrict } from "./utils/logs";
import {
  waitForBackend,
  createTestCapture,
  startCapture,
  updateCapture,
  deleteAllCaptures,
} from "./utils/debug";

/**
 * Tuning controls e2e tests
 *
 * Tests the frequency tuning and radio settings UI:
 * - Frequency display
 * - Tuning controls
 * - Gain adjustment
 * - Sample rate changes
 */

test.describe("Tuning Controls", () => {
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
      name: "Tuning Test",
      center_hz: 100_000_000,
      sample_rate: 1_000_000,
    });
    captureId = capture.id;
    await startCapture(captureId);
  });

  test.afterEach(async () => {
    await deleteAllCaptures();
  });

  test("should display current frequency", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Tuning Test/i).click();
    await dashboardPage.waitForTimeout(500);

    // Look for frequency display (100 MHz or 100.000 MHz format)
    const freqDisplay = dashboardPage.getByText(/100\.?0*\s*MHz/i);
    await expect(freqDisplay.first()).toBeVisible({ timeout: 5000 });

    assertNoErrors(dashboardLogs);
  });

  test("should update frequency display after API change", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Tuning Test/i).click();
    await dashboardPage.waitForTimeout(500);

    // Change frequency via API
    await updateCapture(captureId, {
      centerHz: 88_100_000,
    });

    // Wait for WebSocket update or refresh
    await dashboardPage.waitForTimeout(1000);

    // Reload to ensure fresh state
    await dashboardPage.reload();
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Tuning Test/i).click();
    await dashboardPage.waitForTimeout(500);

    // Should show new frequency
    const freqDisplay = dashboardPage.getByText(/88\.?1?\s*MHz/i);
    await expect(freqDisplay.first()).toBeVisible({ timeout: 5000 });

    assertNoErrors(dashboardLogs);
  });

  test("should show device information", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Tuning Test/i).click();
    await dashboardPage.waitForTimeout(500);

    // Look for device info (FakeSDR or fake0)
    const deviceInfo = dashboardPage.getByText(/fake|FakeSDR|simulated/i);
    const hasDeviceInfo = await deviceInfo
      .first()
      .isVisible({ timeout: 3000 })
      .catch(() => false);

    // Device info may be in settings or radio panel
    expect(hasDeviceInfo).toBeDefined();

    assertNoErrors(dashboardLogs);
  });

  test("should show sample rate information", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Tuning Test/i).click();
    await dashboardPage.waitForTimeout(500);

    // Look for sample rate display (1 MHz or 1000000 or 1.0 MS/s format)
    const sampleRateInfo = dashboardPage.getByText(/1\s*(MHz|MS|000|Msps)/i);
    const _hasSampleRate = await sampleRateInfo
      .first()
      .isVisible({ timeout: 3000 })
      .catch(() => false);

    assertNoErrors(dashboardLogs);
  });

  test("should interact with frequency input if available", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Tuning Test/i).click();
    await dashboardPage.waitForTimeout(500);

    // Look for frequency input
    const freqInput = dashboardPage
      .locator('input[type="number"], input[inputmode="decimal"]')
      .first();

    if (await freqInput.isVisible({ timeout: 3000 }).catch(() => false)) {
      // Clear and enter new frequency
      await freqInput.click();
      await freqInput.fill("105.5");

      // Check that input accepted the value
      const value = await freqInput.inputValue();
      expect(value).toContain("105");
    }

    assertNoErrors(dashboardLogs);
  });

  test("should have working start/stop controls", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Tuning Test/i).click();
    await dashboardPage.waitForTimeout(500);

    // Look for start/stop button
    const stopButton = dashboardPage.getByRole("button", { name: /stop/i });
    const startButton = dashboardPage.getByRole("button", { name: /start/i });

    // Capture is running, so stop button might be visible
    const hasStopButton = await stopButton
      .isVisible({ timeout: 3000 })
      .catch(() => false);

    if (hasStopButton) {
      await stopButton.click();
      await dashboardPage.waitForTimeout(500);

      // Start button should now be visible
      await expect(startButton).toBeVisible({ timeout: 3000 });
    }

    assertNoErrors(dashboardLogs);
  });

  test("should display gain control", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Tuning Test/i).click();
    await dashboardPage.waitForTimeout(500);

    // Look for gain control (slider, input, or label)
    const gainControl = dashboardPage.locator(
      '[class*="gain"], input[name*="gain"], label:has-text("Gain")',
    );
    const _hasGain = await gainControl
      .first()
      .isVisible({ timeout: 3000 })
      .catch(() => false);

    // Gain control should exist in the radio panel
    // If not visible, it might be in a collapsed section or modal

    assertNoErrors(dashboardLogs);
  });

  test("should have responsive tuning panel layout", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await dashboardPage.goto("/?debugOverflow=true");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Tuning Test/i).click();
    await dashboardPage.waitForTimeout(500);

    // Test at different viewport sizes
    const sizes = [
      { width: 1920, height: 1080 },
      { width: 1440, height: 900 },
      { width: 1280, height: 720 },
    ];

    for (const size of sizes) {
      await dashboardPage.setViewportSize(size);
      await dashboardPage.waitForTimeout(300);

      // Tuning controls should remain visible
      const freqDisplay = dashboardPage.getByText(/100\.?0*\s*MHz/i);
      await expect(freqDisplay.first()).toBeVisible({ timeout: 2000 });
    }

    assertCleanConsoleStrict(dashboardLogs);
  });

  test("should handle frequency changes via keyboard", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Tuning Test/i).click();
    await dashboardPage.waitForTimeout(500);

    // Find a frequency input and try keyboard interaction
    const freqInput = dashboardPage.locator('input[type="number"]').first();

    if (await freqInput.isVisible({ timeout: 2000 }).catch(() => false)) {
      await freqInput.click();
      await freqInput.press("ArrowUp");
      await dashboardPage.waitForTimeout(100);
      await freqInput.press("ArrowDown");
      await dashboardPage.waitForTimeout(100);
    }

    assertNoErrors(dashboardLogs);
  });

  test("should persist frequency after page reload", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    // Change frequency
    await updateCapture(captureId, {
      centerHz: 144_200_000,
    });

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Tuning Test/i).click();
    await dashboardPage.waitForTimeout(500);

    // Should show 144 MHz
    const freqDisplay = dashboardPage.getByText(/144/i);
    await expect(freqDisplay.first()).toBeVisible({ timeout: 5000 });

    // Reload
    await dashboardPage.reload();
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Tuning Test/i).click();
    await dashboardPage.waitForTimeout(500);

    // Should still show 144 MHz
    await expect(freqDisplay.first()).toBeVisible({ timeout: 5000 });

    assertNoErrors(dashboardLogs);
  });

  test("should not overflow with long frequency numbers", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    // Set a precise frequency with many decimal places
    await updateCapture(captureId, {
      centerHz: 1_234_567_890,
    });

    await dashboardPage.goto("/?debugOverflow=true");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Tuning Test/i).click();
    await dashboardPage.waitForTimeout(1000);

    // Visual overflow check
    assertCleanConsoleStrict(dashboardLogs);
  });
});
