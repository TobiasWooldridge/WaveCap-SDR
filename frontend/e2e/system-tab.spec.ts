import { test, expect } from "./fixtures";
import { assertNoErrors } from "./utils/logs";
import { waitForBackend } from "./utils/debug";

/**
 * System tab e2e tests for WaveCap-SDR
 *
 * Tests for the System tab functionality:
 * - System tab navigation
 * - Metric cards display (CPU, Memory, Temperature, Captures)
 * - Log viewer functionality
 * - WebSocket connection status
 */

test.describe("System Tab", () => {
  test.beforeAll(async () => {
    const ready = await waitForBackend(15000);
    if (!ready) {
      throw new Error("Backend not available");
    }
  });

  test("should navigate to System tab", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    // Find and click the System tab
    const systemTab = dashboardPage.getByRole("button", { name: /system/i });
    await expect(systemTab).toBeVisible({ timeout: 5000 });
    await systemTab.click();

    // Wait for System panel to load
    await dashboardPage.waitForTimeout(500);

    // Verify URL updated
    await expect(dashboardPage).toHaveURL(/mode=system/);

    assertNoErrors(dashboardLogs);
  });

  test("should display metric cards", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await dashboardPage.goto("/?mode=system");
    await dashboardPage.waitForLoadState("networkidle");

    // Wait for WebSocket connection and data
    await dashboardPage.waitForTimeout(2000);

    // Check for CPU metric card
    const cpuLabel = dashboardPage.getByText(/cpu/i);
    await expect(cpuLabel.first()).toBeVisible({ timeout: 5000 });

    // Check for Memory metric card
    const memoryLabel = dashboardPage.getByText(/memory/i);
    await expect(memoryLabel.first()).toBeVisible({ timeout: 5000 });

    // Check for Captures metric card
    const capturesLabel = dashboardPage.getByText(/captures/i);
    await expect(capturesLabel.first()).toBeVisible({ timeout: 5000 });

    assertNoErrors(dashboardLogs);
  });

  test("should show WebSocket connection status", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await dashboardPage.goto("/?mode=system");
    await dashboardPage.waitForLoadState("networkidle");

    // Wait for WebSocket to connect
    await dashboardPage.waitForTimeout(2000);

    // Should show Connected status
    const connectedStatus = dashboardPage.getByText(/connected/i);
    await expect(connectedStatus.first()).toBeVisible({ timeout: 5000 });

    assertNoErrors(dashboardLogs);
  });

  test("should display log viewer", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await dashboardPage.goto("/?mode=system");
    await dashboardPage.waitForLoadState("networkidle");

    // Wait for content to load
    await dashboardPage.waitForTimeout(2000);

    // Look for Logs section header or log entries
    const logsHeader = dashboardPage.getByText(/^logs$/i);
    await expect(logsHeader.first()).toBeVisible({ timeout: 5000 });

    // Log level filter should be present
    const levelFilter = dashboardPage.getByRole("combobox");
    await expect(levelFilter.first()).toBeVisible({ timeout: 5000 });

    assertNoErrors(dashboardLogs);
  });

  test("should filter logs by level", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await dashboardPage.goto("/?mode=system");
    await dashboardPage.waitForLoadState("networkidle");

    // Wait for logs to load
    await dashboardPage.waitForTimeout(2000);

    // Find the log level filter dropdown
    const levelFilter = dashboardPage.getByRole("combobox").first();
    await expect(levelFilter).toBeVisible({ timeout: 5000 });

    // Change to WARNING level
    await levelFilter.selectOption("WARNING");

    // Wait for filter to apply
    await dashboardPage.waitForTimeout(500);

    // The filter should be applied (verify dropdown value changed)
    await expect(levelFilter).toHaveValue("WARNING");

    assertNoErrors(dashboardLogs);
  });

  test("should show per-core CPU breakdown", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await dashboardPage.goto("/?mode=system");
    await dashboardPage.waitForLoadState("networkidle");

    // Wait for metrics to load
    await dashboardPage.waitForTimeout(2000);

    // Look for CPU Cores section
    const cpuCoresSection = dashboardPage.getByText(/cpu cores/i);
    await expect(cpuCoresSection.first()).toBeVisible({ timeout: 5000 });

    // Should show at least one core indicator (C0, C1, etc.)
    const coreIndicator = dashboardPage.getByText(/^C[0-9]$/);
    await expect(coreIndicator.first()).toBeVisible({ timeout: 5000 });

    assertNoErrors(dashboardLogs);
  });

  test("should pause and resume log auto-scroll", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await dashboardPage.goto("/?mode=system");
    await dashboardPage.waitForLoadState("networkidle");

    // Wait for logs to load
    await dashboardPage.waitForTimeout(2000);

    // Find the pause button (it's a button with a Pause icon initially)
    const pauseButton = dashboardPage
      .locator("button")
      .filter({ has: dashboardPage.locator('svg[class*="lucide-pause"]') });

    // If pause button is visible, click it
    const hasPauseButton = await pauseButton
      .first()
      .isVisible({ timeout: 3000 })
      .catch(() => false);

    if (hasPauseButton) {
      await pauseButton.first().click();
      await dashboardPage.waitForTimeout(300);

      // Now it should show play button
      const playButton = dashboardPage
        .locator("button")
        .filter({ has: dashboardPage.locator('svg[class*="lucide-play"]') });
      await expect(playButton.first()).toBeVisible({ timeout: 3000 });
    }

    assertNoErrors(dashboardLogs);
  });

  test("should search/filter logs by text", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await dashboardPage.goto("/?mode=system");
    await dashboardPage.waitForLoadState("networkidle");

    // Wait for logs to load
    await dashboardPage.waitForTimeout(2000);

    // Find the search input
    const searchInput = dashboardPage.getByPlaceholder(/filter/i);
    await expect(searchInput.first()).toBeVisible({ timeout: 5000 });

    // Type a search term
    await searchInput.first().fill("test");
    await dashboardPage.waitForTimeout(500);

    // The filter should be applied (input should have value)
    await expect(searchInput.first()).toHaveValue("test");

    assertNoErrors(dashboardLogs);
  });

  test("should handle System tab without errors at various viewports", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await dashboardPage.goto("/?mode=system&debugOverflow=true");
    await dashboardPage.waitForLoadState("networkidle");

    // Wait for content to stabilize
    await dashboardPage.waitForTimeout(2000);

    // Test different viewport sizes
    const viewports = [
      { width: 1920, height: 1080 },
      { width: 1440, height: 900 },
      { width: 1280, height: 720 },
    ];

    for (const viewport of viewports) {
      await dashboardPage.setViewportSize(viewport);
      await dashboardPage.waitForTimeout(300);

      // Page should remain functional
      await expect(dashboardPage.locator("body")).toBeVisible();

      // System tab content should still be visible
      const cpuLabel = dashboardPage.getByText(/cpu/i);
      await expect(cpuLabel.first()).toBeVisible({ timeout: 3000 });
    }

    assertNoErrors(dashboardLogs);
  });

  test("should persist System tab selection after reload", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await dashboardPage.goto("/?mode=system");
    await dashboardPage.waitForLoadState("networkidle");

    // Wait for content to load
    await dashboardPage.waitForTimeout(1000);

    // Verify we're on System tab
    await expect(dashboardPage).toHaveURL(/mode=system/);

    // Reload the page
    await dashboardPage.reload();
    await dashboardPage.waitForLoadState("networkidle");

    // Should still be on System tab
    await expect(dashboardPage).toHaveURL(/mode=system/);

    // System content should be visible
    const cpuLabel = dashboardPage.getByText(/cpu/i);
    await expect(cpuLabel.first()).toBeVisible({ timeout: 5000 });

    assertNoErrors(dashboardLogs);
  });
});
