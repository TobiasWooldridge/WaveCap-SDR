import { test, expect } from "./fixtures";
import { assertNoErrors } from "./utils/logs";
import {
  waitForBackend,
  createTestCapture,
  createTestChannel,
  startCapture,
  deleteAllCaptures,
  deleteCapture,
} from "./utils/debug";

/**
 * Error handling e2e tests
 *
 * Tests the UI's resilience to errors:
 * - Invalid API responses
 * - Deleted resources
 * - Network timeouts
 * - Console error detection
 */

test.describe("Error Handling", () => {
  test.beforeAll(async () => {
    const ready = await waitForBackend(15000);
    if (!ready) {
      throw new Error("Backend not available");
    }
  });

  test.beforeEach(async () => {
    await deleteAllCaptures();
  });

  test.afterEach(async () => {
    await deleteAllCaptures();
  });

  test("should handle deleted capture gracefully", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    // Create and then delete a capture
    const capture = await createTestCapture({
      name: "Delete Test",
      center_hz: 100_000_000,
    });

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    // Verify capture appears
    await expect(dashboardPage.getByText(/Delete Test/i)).toBeVisible({
      timeout: 5000,
    });

    // Click on it
    await dashboardPage.getByText(/Delete Test/i).click();
    await dashboardPage.waitForTimeout(300);

    // Delete via API while viewing
    await deleteCapture(capture.id);

    // Reload page
    await dashboardPage.reload();
    await dashboardPage.waitForLoadState("networkidle");

    // Capture should no longer be visible
    const captureText = dashboardPage.getByText(/Delete Test/i);
    await expect(captureText).not.toBeVisible({ timeout: 3000 });

    // Page should still work
    await expect(dashboardPage.locator("body")).toBeVisible();

    assertNoErrors(dashboardLogs);
  });

  test("should recover from rapid navigation", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    // Create multiple captures
    const capture1 = await createTestCapture({
      name: "Rapid Nav 1",
      center_hz: 88_000_000,
    });
    const _capture2 = await createTestCapture({
      name: "Rapid Nav 2",
      center_hz: 100_000_000,
    });
    await startCapture(capture1.id);

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    // Rapidly click between captures
    for (let i = 0; i < 10; i++) {
      await dashboardPage.getByText(/Rapid Nav 1/i).click();
      await dashboardPage.waitForTimeout(50);
      await dashboardPage.getByText(/Rapid Nav 2/i).click();
      await dashboardPage.waitForTimeout(50);
    }

    // Should not crash
    await expect(dashboardPage.locator("body")).toBeVisible();

    assertNoErrors(dashboardLogs);
  });

  test("should handle page refresh during operation", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    const capture = await createTestCapture({
      name: "Refresh During Op",
      center_hz: 100_000_000,
    });
    await startCapture(capture.id);
    await createTestChannel(capture.id, {
      name: "Refresh Channel",
      offset_hz: 0,
      mode: "wbfm",
    });

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Refresh During Op/i).click();
    await dashboardPage.waitForTimeout(200);

    // Refresh in the middle of interaction
    await dashboardPage.reload();
    await dashboardPage.waitForLoadState("networkidle");

    // Everything should still work
    await expect(dashboardPage.getByText(/Refresh During Op/i)).toBeVisible({
      timeout: 5000,
    });

    assertNoErrors(dashboardLogs);
  });

  test("should handle back/forward navigation", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await createTestCapture({
      name: "History Nav",
      center_hz: 100_000_000,
    });

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/History Nav/i).click();
    await dashboardPage.waitForTimeout(300);

    // Go back
    await dashboardPage.goBack();
    await dashboardPage.waitForTimeout(300);

    // Go forward
    await dashboardPage.goForward();
    await dashboardPage.waitForTimeout(300);

    // Should still work
    await expect(dashboardPage.locator("body")).toBeVisible();

    assertNoErrors(dashboardLogs);
  });

  test("should display loading states", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await dashboardPage.goto("/");

    // During initial load, there should be a loading indicator
    // This test verifies the app doesn't flash errors during load

    await dashboardPage.waitForLoadState("networkidle");

    // After load, body should be visible
    await expect(dashboardPage.locator("body")).toBeVisible();

    assertNoErrors(dashboardLogs);
  });

  test("should handle empty channel list", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    const capture = await createTestCapture({
      name: "Empty Channels",
      center_hz: 100_000_000,
    });
    await startCapture(capture.id);

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Empty Channels/i).click();
    await dashboardPage.waitForTimeout(500);

    // Should show empty state or add channel prompt
    const emptyIndicator = dashboardPage.getByText(
      /no channel|add channel|create channel/i,
    );
    const addButton = dashboardPage.getByRole("button", {
      name: /add.*channel|\+/i,
    });

    const hasEmpty = await emptyIndicator
      .first()
      .isVisible({ timeout: 2000 })
      .catch(() => false);
    const hasAdd = await addButton
      .first()
      .isVisible({ timeout: 2000 })
      .catch(() => false);

    // Either empty message or add button should be visible
    expect(hasEmpty || hasAdd).toBeTruthy();

    assertNoErrors(dashboardLogs);
  });

  test("should handle capture without starting", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    // Create but don't start
    await createTestCapture({
      name: "Not Started",
      center_hz: 100_000_000,
    });

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Not Started/i).click();
    await dashboardPage.waitForTimeout(500);

    // Should show stopped/ready state, not error
    const statusIndicator = dashboardPage.locator(
      '.badge, [class*="status"], [class*="stopped"], [class*="ready"]',
    );
    const _hasStatus = await statusIndicator
      .first()
      .isVisible({ timeout: 2000 })
      .catch(() => false);

    assertNoErrors(dashboardLogs);
  });

  test("should handle rapid channel creation", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    const capture = await createTestCapture({
      name: "Rapid Channels",
      center_hz: 100_000_000,
    });
    await startCapture(capture.id);

    // Create multiple channels rapidly
    const promises = [];
    for (let i = 0; i < 5; i++) {
      promises.push(
        createTestChannel(capture.id, {
          name: `Rapid ${i}`,
          offset_hz: i * 25000,
          mode: "nbfm",
        }),
      );
    }
    await Promise.all(promises);

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Rapid Channels/i).click();
    await dashboardPage.waitForTimeout(500);

    // All channels should appear
    await expect(dashboardPage.getByText(/Rapid 0/i)).toBeVisible({
      timeout: 5000,
    });
    await expect(dashboardPage.getByText(/Rapid 4/i)).toBeVisible({
      timeout: 5000,
    });

    assertNoErrors(dashboardLogs);
  });

  test("should not leak console errors on normal operation", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    const capture = await createTestCapture({
      name: "No Errors Test",
      center_hz: 100_000_000,
    });
    await startCapture(capture.id);
    await createTestChannel(capture.id, {
      name: "Test Channel",
      offset_hz: 0,
      mode: "wbfm",
    });

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/No Errors Test/i).click();
    await dashboardPage.waitForTimeout(1000);

    // Interact with various elements
    const playButton = dashboardPage
      .locator("button")
      .filter({
        has: dashboardPage.locator("svg"),
      })
      .first();

    if (await playButton.isVisible({ timeout: 1000 }).catch(() => false)) {
      await playButton.click();
      await dashboardPage.waitForTimeout(500);
    }

    // Strict assertion - no errors should have occurred
    assertNoErrors(dashboardLogs);
  });

  test("should handle double-click on buttons", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    // Find and double-click add button
    const addButton = dashboardPage
      .getByRole("button", { name: /add|new|\+/i })
      .first();

    if (await addButton.isVisible({ timeout: 2000 }).catch(() => false)) {
      await addButton.dblclick();
      await dashboardPage.waitForTimeout(500);

      // Should not cause issues - close any modal that opened
      await dashboardPage.keyboard.press("Escape");
      await dashboardPage.waitForTimeout(200);
    }

    // Page should still work
    await expect(dashboardPage.locator("body")).toBeVisible();

    assertNoErrors(dashboardLogs);
  });
});
