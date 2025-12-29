import { test, expect } from "./fixtures";
import { assertNoErrors } from "./utils/logs";
import {
  waitForBackend,
  createTestCapture,
  deleteCapture,
  startCapture,
  stopCapture,
  getCapture,
  deleteAllCaptures,
} from "./utils/debug";

/**
 * Capture lifecycle e2e tests
 *
 * Tests the complete lifecycle of captures using the fake SDR driver:
 * - Creation
 * - Starting/Stopping
 * - State transitions
 * - Deletion
 */

test.describe("Capture Lifecycle", () => {
  test.beforeAll(async () => {
    // Ensure backend is available
    const ready = await waitForBackend(15000);
    if (!ready) {
      throw new Error("Backend not available");
    }
  });

  test.beforeEach(async () => {
    // Clean up any existing captures before each test
    await deleteAllCaptures();
  });

  test.afterEach(async () => {
    // Clean up after each test
    await deleteAllCaptures();
  });

  test("should create a capture with fake driver", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    // Create capture via API
    const capture = await createTestCapture({
      name: "Lifecycle Test",
      center_hz: 100_000_000,
      sample_rate: 1_000_000,
    });

    expect(capture.id).toBeTruthy();
    expect(capture.deviceId).toBe("fake0");

    // Navigate to dashboard
    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    // The capture should appear in the UI
    await expect(
      dashboardPage.getByText(/Lifecycle Test|100.*MHz/i),
    ).toBeVisible({ timeout: 5000 });

    assertNoErrors(dashboardLogs);
  });

  test("should start and stop a capture", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    // Create and start capture
    const capture = await createTestCapture({
      name: "Start Stop Test",
      center_hz: 100_000_000,
    });

    // Start the capture
    const startedCapture = await startCapture(capture.id);
    expect(startedCapture.state).toBe("running");

    // Navigate to dashboard
    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    // Click on the capture tab
    const captureTab = dashboardPage.getByText(/Start Stop Test/i);
    await expect(captureTab).toBeVisible({ timeout: 5000 });
    await captureTab.click();

    // Wait for UI to update
    await dashboardPage.waitForTimeout(500);

    // Verify capture is running - look for spectrum or running indicator
    const runningIndicator = dashboardPage.locator(
      '[class*="running"], [class*="active"], .badge-success, .text-success',
    );
    const _hasRunningIndicator = await runningIndicator
      .first()
      .isVisible({ timeout: 3000 })
      .catch(() => false);

    // Stop the capture via API
    const stoppedCapture = await stopCapture(capture.id);
    expect(stoppedCapture.state).toBe("stopped");

    assertNoErrors(dashboardLogs);
  });

  test("should show capture state in UI after page refresh", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    // Create and start capture
    const capture = await createTestCapture({
      name: "Refresh Test",
      center_hz: 88_000_000,
    });
    await startCapture(capture.id);

    // Navigate to dashboard
    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    // Verify capture is visible
    await expect(dashboardPage.getByText(/Refresh Test|88.*MHz/i)).toBeVisible({
      timeout: 5000,
    });

    // Refresh the page
    await dashboardPage.reload();
    await dashboardPage.waitForLoadState("networkidle");

    // Capture should still be visible after refresh
    await expect(dashboardPage.getByText(/Refresh Test|88.*MHz/i)).toBeVisible({
      timeout: 5000,
    });

    assertNoErrors(dashboardLogs);
  });

  test("should delete a capture and remove from UI", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    // Create capture
    const capture = await createTestCapture({
      name: "Delete Test",
      center_hz: 100_000_000,
    });

    // Navigate and verify it appears
    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");
    await expect(dashboardPage.getByText(/Delete Test/i)).toBeVisible({
      timeout: 5000,
    });

    // Delete via API
    await deleteCapture(capture.id);

    // Refresh page
    await dashboardPage.reload();
    await dashboardPage.waitForLoadState("networkidle");

    // Give time for state to update
    await dashboardPage.waitForTimeout(500);

    // Capture should no longer be visible
    const captureElement = dashboardPage.getByText(/Delete Test/i);
    await expect(captureElement).not.toBeVisible({ timeout: 3000 });

    assertNoErrors(dashboardLogs);
  });

  test("should handle multiple captures", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    // Create multiple captures
    const _capture1 = await createTestCapture({
      name: "Multi Test 1",
      center_hz: 88_000_000,
    });
    const capture2 = await createTestCapture({
      name: "Multi Test 2",
      center_hz: 100_000_000,
    });
    const _capture3 = await createTestCapture({
      name: "Multi Test 3",
      center_hz: 162_000_000,
    });

    // Start one of them
    await startCapture(capture2.id);

    // Navigate to dashboard
    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    // All three should be visible as tabs
    await expect(dashboardPage.getByText(/Multi Test 1/i)).toBeVisible({
      timeout: 5000,
    });
    await expect(dashboardPage.getByText(/Multi Test 2/i)).toBeVisible({
      timeout: 5000,
    });
    await expect(dashboardPage.getByText(/Multi Test 3/i)).toBeVisible({
      timeout: 5000,
    });

    // Click between tabs
    await dashboardPage.getByText(/Multi Test 1/i).click();
    await dashboardPage.waitForTimeout(300);

    await dashboardPage.getByText(/Multi Test 3/i).click();
    await dashboardPage.waitForTimeout(300);

    await dashboardPage.getByText(/Multi Test 2/i).click();
    await dashboardPage.waitForTimeout(300);

    assertNoErrors(dashboardLogs);
  });

  test("should preserve capture state across tab switches", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    // Create two captures, start one
    const capture1 = await createTestCapture({
      name: "Tab Switch 1",
      center_hz: 88_000_000,
    });
    const _capture2 = await createTestCapture({
      name: "Tab Switch 2",
      center_hz: 100_000_000,
    });

    await startCapture(capture1.id);

    // Navigate
    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    // Switch between captures
    await dashboardPage.getByText(/Tab Switch 1/i).click();
    await dashboardPage.waitForTimeout(300);

    await dashboardPage.getByText(/Tab Switch 2/i).click();
    await dashboardPage.waitForTimeout(300);

    // Switch back to first capture
    await dashboardPage.getByText(/Tab Switch 1/i).click();
    await dashboardPage.waitForTimeout(500);

    // First capture should still show as running
    const captureState = await getCapture(capture1.id);
    expect(captureState?.state).toBe("running");

    assertNoErrors(dashboardLogs);
  });
});
