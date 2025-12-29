import { test, expect } from "./fixtures";
import { assertNoErrors, assertCleanConsoleStrict } from "./utils/logs";
import { waitForBackend, deleteAllCaptures } from "./utils/debug";

/**
 * Dashboard e2e tests for WaveCap-SDR
 *
 * Basic dashboard functionality tests:
 * - Page load and initial state
 * - Empty state display
 * - Navigation elements
 */

test.describe("Dashboard", () => {
  test.beforeAll(async () => {
    const ready = await waitForBackend(15000);
    if (!ready) {
      throw new Error("Backend not available");
    }
  });

  test.beforeEach(async () => {
    await deleteAllCaptures();
  });

  test("should load the dashboard", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    // Page should be visible
    await expect(dashboardPage.locator("body")).toBeVisible();

    // Should have some navigation
    const nav = dashboardPage.locator(
      'nav, [role="navigation"], .navbar, .nav',
    );
    await expect(nav.first()).toBeVisible({ timeout: 5000 });

    assertNoErrors(dashboardLogs);
  });

  test("should show empty state when no captures exist", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    // Should show empty state message or add button
    const emptyState = dashboardPage.getByText(
      /no radio|add radio|get started/i,
    );
    const addButton = dashboardPage.getByRole("button", {
      name: /add|new|\+/i,
    });

    const hasEmptyState = await emptyState
      .first()
      .isVisible({ timeout: 3000 })
      .catch(() => false);
    const hasAddButton = await addButton
      .first()
      .isVisible({ timeout: 3000 })
      .catch(() => false);

    // Either empty state message or add button should be visible
    expect(hasEmptyState || hasAddButton).toBeTruthy();

    assertNoErrors(dashboardLogs);
  });

  test("should have visible add capture button", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    // Add button should be visible (either + button or "Add Radio" button)
    const addButton = dashboardPage
      .getByRole("button", { name: /add|new|\+/i })
      .first();
    await expect(addButton).toBeVisible({ timeout: 5000 });

    assertNoErrors(dashboardLogs);
  });

  test("should not have console errors on initial load", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    // Wait for any async operations
    await dashboardPage.waitForTimeout(1000);

    assertNoErrors(dashboardLogs);
  });

  test("should handle page refresh", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    // Refresh the page
    await dashboardPage.reload();
    await dashboardPage.waitForLoadState("networkidle");

    // Page should still work
    await expect(dashboardPage.locator("body")).toBeVisible();

    assertNoErrors(dashboardLogs);
  });

  test("should have no visual issues at default viewport", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await dashboardPage.goto("/?debugOverflow=true");
    await dashboardPage.waitForLoadState("networkidle");

    // Wait for overflow detector to run
    await dashboardPage.waitForTimeout(1000);

    assertCleanConsoleStrict(dashboardLogs);
  });

  test("should handle viewport resize without breaking", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await dashboardPage.goto("/?debugOverflow=true");
    await dashboardPage.waitForLoadState("networkidle");

    // Test different viewport sizes
    const viewports = [
      { width: 1920, height: 1080 },
      { width: 1440, height: 900 },
      { width: 1280, height: 720 },
      { width: 1024, height: 768 },
    ];

    for (const viewport of viewports) {
      await dashboardPage.setViewportSize(viewport);
      await dashboardPage.waitForTimeout(200);

      // Page should remain functional
      await expect(dashboardPage.locator("body")).toBeVisible();
    }

    assertCleanConsoleStrict(dashboardLogs);
  });

  test("should display application title or logo", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    // Look for app name or logo
    const title = dashboardPage.getByText(/wavecap|sdr/i);
    const hasTitle = await title
      .first()
      .isVisible({ timeout: 3000 })
      .catch(() => false);

    // Title may be in header or be the page title
    const pageTitle = await dashboardPage.title();
    expect(
      hasTitle || pageTitle.toLowerCase().includes("wavecap"),
    ).toBeTruthy();

    assertNoErrors(dashboardLogs);
  });
});
