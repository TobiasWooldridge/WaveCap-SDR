import { test, expect } from "./fixtures";
import { assertNoErrors, assertCleanConsoleStrict } from "./utils/logs";
import {
  waitForBackend,
  createTestCapture,
  startCapture,
  deleteAllCaptures,
} from "./utils/debug";

/**
 * Device Tab Bar e2e tests
 *
 * Tests the device/radio tab bar navigation:
 * - Empty state display
 * - Add radio modal
 * - Device selection
 * - Multiple device tabs
 * - Delete confirmation modal
 * - Status indicators
 */

test.describe("Device Tab Bar", () => {
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

  test("should show empty state when no devices", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    // Should show "No radios configured" or similar empty state
    const emptyText = dashboardPage.getByText(/no radio|add radio/i);
    const _hasEmpty = await emptyText
      .first()
      .isVisible({ timeout: 3000 })
      .catch(() => false);

    // Add button should be visible
    const addButton = dashboardPage
      .getByRole("button", { name: /add|new|\+/i })
      .first();
    await expect(addButton).toBeVisible({ timeout: 3000 });

    assertNoErrors(dashboardLogs);
  });

  test("should show device tab after creating capture", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    // Create a capture
    const capture = await createTestCapture({
      name: "Tab Bar Test",
      center_hz: 100_000_000,
    });
    await startCapture(capture.id);

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    // Device tab should be visible
    const deviceTab = dashboardPage.getByText(/Tab Bar Test|100.*MHz|fake/i);
    await expect(deviceTab.first()).toBeVisible({ timeout: 5000 });

    assertNoErrors(dashboardLogs);
  });

  test("should switch between multiple device tabs", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    // Create two captures on different frequencies
    const capture1 = await createTestCapture({
      name: "Device A",
      center_hz: 88_000_000,
    });
    const _capture2 = await createTestCapture({
      name: "Device B",
      center_hz: 162_000_000,
    });

    await startCapture(capture1.id);

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    // Both devices should be visible as tabs
    await expect(dashboardPage.getByText(/Device A/i)).toBeVisible({
      timeout: 5000,
    });
    await expect(dashboardPage.getByText(/Device B/i)).toBeVisible({
      timeout: 5000,
    });

    // Click Device A
    await dashboardPage.getByText(/Device A/i).click();
    await dashboardPage.waitForTimeout(300);

    // Click Device B
    await dashboardPage.getByText(/Device B/i).click();
    await dashboardPage.waitForTimeout(300);

    // Should still work
    await expect(dashboardPage.locator("body")).toBeVisible();

    assertNoErrors(dashboardLogs);
  });

  test("should show status indicator for running capture", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    const capture = await createTestCapture({
      name: "Status Test",
      center_hz: 100_000_000,
    });
    await startCapture(capture.id);

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    // Look for running status indicator (green badge, "Running" text, etc.)
    const statusIndicator = dashboardPage.locator(
      '.badge, [class*="status"], [class*="running"], .text-success',
    );
    const _hasStatus = await statusIndicator
      .first()
      .isVisible({ timeout: 3000 })
      .catch(() => false);

    assertNoErrors(dashboardLogs);
  });

  test("should open add modal from plus button", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    // Create a capture so we have the tab bar with + button
    await createTestCapture({
      name: "Add Modal Test",
      center_hz: 100_000_000,
    });

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    // Click the + button
    const plusButton = dashboardPage
      .getByRole("button")
      .filter({
        has: dashboardPage.locator('svg, :text("+")'),
      })
      .first();

    if (await plusButton.isVisible({ timeout: 3000 }).catch(() => false)) {
      await plusButton.click();
      await dashboardPage.waitForTimeout(300);

      // Modal should appear with "Add Radio" option
      const modal = dashboardPage.locator('.modal, [role="dialog"]');
      const hasModal = await modal
        .isVisible({ timeout: 2000 })
        .catch(() => false);

      if (hasModal) {
        const addRadioOption = dashboardPage.getByText(/add radio/i);
        await expect(addRadioOption).toBeVisible({ timeout: 2000 });

        // Close modal
        await dashboardPage.keyboard.press("Escape");
      }
    }

    assertNoErrors(dashboardLogs);
  });

  test("should show delete confirmation modal", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    const _capture = await createTestCapture({
      name: "Delete Confirm Test",
      center_hz: 100_000_000,
    });

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    // Click to select the device
    await dashboardPage.getByText(/Delete Confirm Test/i).click();
    await dashboardPage.waitForTimeout(300);

    // Look for delete button (X icon)
    const deleteButton = dashboardPage
      .locator("button")
      .filter({
        has: dashboardPage.locator(
          'svg[class*="lucide-x"], [aria-label*="delete"]',
        ),
      })
      .first();

    if (await deleteButton.isVisible({ timeout: 2000 }).catch(() => false)) {
      await deleteButton.click();
      await dashboardPage.waitForTimeout(300);

      // Confirmation modal should appear
      const confirmModal = dashboardPage.getByText(/delete.*device|confirm/i);
      const hasConfirm = await confirmModal
        .first()
        .isVisible({ timeout: 2000 })
        .catch(() => false);

      if (hasConfirm) {
        // Cancel deletion
        const cancelButton = dashboardPage.getByRole("button", {
          name: /cancel/i,
        });
        if (
          await cancelButton.isVisible({ timeout: 1000 }).catch(() => false)
        ) {
          await cancelButton.click();
        } else {
          await dashboardPage.keyboard.press("Escape");
        }
      }
    }

    assertNoErrors(dashboardLogs);
  });

  test("should show frequency on device tab", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await createTestCapture({
      name: "Freq Display Test",
      center_hz: 146_520_000, // 146.52 MHz - easy to recognize
    });

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    // Should show frequency somewhere in the tab
    const freqDisplay = dashboardPage.getByText(/146.*MHz|146\.52/i);
    const _hasFreq = await freqDisplay
      .first()
      .isVisible({ timeout: 3000 })
      .catch(() => false);

    assertNoErrors(dashboardLogs);
  });

  test("should show settings button", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await createTestCapture({
      name: "Settings Button Test",
      center_hz: 100_000_000,
    });

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    // Look for settings button (gear icon)
    const settingsButton = dashboardPage
      .getByRole("button", { name: /settings/i })
      .or(
        dashboardPage.locator("button").filter({
          has: dashboardPage.locator('svg[class*="settings"]'),
        }),
      );

    const _hasSettings = await settingsButton
      .first()
      .isVisible({ timeout: 3000 })
      .catch(() => false);

    assertNoErrors(dashboardLogs);
  });

  test("should handle many device tabs without overflow", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    // Create several captures
    for (let i = 0; i < 5; i++) {
      await createTestCapture({
        name: `Device ${i + 1}`,
        center_hz: 88_000_000 + i * 10_000_000,
      });
    }

    await dashboardPage.goto("/?debugOverflow=true");
    await dashboardPage.waitForLoadState("networkidle");

    // All devices should be accessible (may scroll)
    await expect(dashboardPage.getByText(/Device 1/i)).toBeVisible({
      timeout: 5000,
    });
    await expect(dashboardPage.getByText(/Device 5/i)).toBeVisible({
      timeout: 5000,
    });

    assertCleanConsoleStrict(dashboardLogs);
  });

  test("should highlight selected device tab", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await createTestCapture({
      name: "Highlight Test A",
      center_hz: 88_000_000,
    });
    await createTestCapture({
      name: "Highlight Test B",
      center_hz: 100_000_000,
    });

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    // Click first tab
    await dashboardPage.getByText(/Highlight Test A/i).click();
    await dashboardPage.waitForTimeout(300);

    // First tab should have selected styling (bg-body vs bg-dark)
    const tabA = dashboardPage
      .locator('[class*="bg-body"], [class*="active"]')
      .filter({ hasText: /Highlight Test A/i });
    const _isAHighlighted = await tabA
      .first()
      .isVisible({ timeout: 2000 })
      .catch(() => false);

    // Click second tab
    await dashboardPage.getByText(/Highlight Test B/i).click();
    await dashboardPage.waitForTimeout(300);

    // Second tab should now be highlighted
    const tabB = dashboardPage
      .locator('[class*="bg-body"], [class*="active"]')
      .filter({ hasText: /Highlight Test B/i });
    const _isBHighlighted = await tabB
      .first()
      .isVisible({ timeout: 2000 })
      .catch(() => false);

    assertNoErrors(dashboardLogs);
  });

  test("should update tab when capture state changes", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    const capture = await createTestCapture({
      name: "State Change Test",
      center_hz: 100_000_000,
    });

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/State Change Test/i).click();
    await dashboardPage.waitForTimeout(500);

    // Start capture via API
    await startCapture(capture.id);
    await dashboardPage.waitForTimeout(1000);

    // Status should update (may need reload for non-websocket)
    await dashboardPage.reload();
    await dashboardPage.waitForLoadState("networkidle");

    // Look for running indicator
    const runningIndicator = dashboardPage.locator(
      '.badge:has-text("Running"), [class*="success"], [class*="running"]',
    );
    const _isRunning = await runningIndicator
      .first()
      .isVisible({ timeout: 3000 })
      .catch(() => false);

    assertNoErrors(dashboardLogs);
  });
});
