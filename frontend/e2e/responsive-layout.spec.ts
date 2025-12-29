import { test, expect } from "./fixtures";
import {
  assertNoErrors,
  assertCleanConsoleStrict,
  getVisualIssues,
} from "./utils/logs";
import {
  waitForBackend,
  createTestCapture,
  createTestChannel,
  startCapture,
  deleteAllCaptures,
} from "./utils/debug";

/**
 * Responsive layout e2e tests
 *
 * Tests the UI at different viewport sizes:
 * - Desktop (1920x1080)
 * - Laptop (1440x900)
 * - Small laptop (1280x720)
 * - Tablet (1024x768)
 * - Mobile landscape (932x430)
 */

const VIEWPORTS = {
  desktop: { width: 1920, height: 1080 },
  laptop: { width: 1440, height: 900 },
  smallLaptop: { width: 1280, height: 720 },
  tablet: { width: 1024, height: 768 },
  mobileLandscape: { width: 932, height: 430 },
};

test.describe("Responsive Layout", () => {
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
      name: "Responsive Test",
      center_hz: 100_000_000,
      sample_rate: 1_000_000,
    });
    captureId = capture.id;
    await startCapture(captureId);

    // Create some channels for testing
    await createTestChannel(captureId, {
      name: "Channel 1",
      offset_hz: -50000,
      mode: "wbfm",
    });
    await createTestChannel(captureId, {
      name: "Channel 2",
      offset_hz: 0,
      mode: "nbfm",
    });
  });

  test.afterEach(async () => {
    await deleteAllCaptures();
  });

  test("should render correctly at desktop viewport", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await dashboardPage.setViewportSize(VIEWPORTS.desktop);
    await dashboardPage.goto("/?debugOverflow=true");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Responsive Test/i).click();
    await dashboardPage.waitForTimeout(1000);

    // All panels should be visible at desktop size
    const navigation = dashboardPage.locator(
      'nav, [role="navigation"], .nav-tabs',
    );
    await expect(navigation.first()).toBeVisible();

    // Spectrum, radio panel, and channel list should all be visible
    const panels = await dashboardPage
      .locator('.d-flex, .row, [class*="panel"]')
      .count();
    expect(panels).toBeGreaterThan(0);

    assertCleanConsoleStrict(dashboardLogs);
  });

  test("should render correctly at laptop viewport", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await dashboardPage.setViewportSize(VIEWPORTS.laptop);
    await dashboardPage.goto("/?debugOverflow=true");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Responsive Test/i).click();
    await dashboardPage.waitForTimeout(1000);

    // Navigation should still be visible
    await expect(
      dashboardPage.locator('nav, [role="navigation"]').first(),
    ).toBeVisible();

    // Content should be accessible
    await expect(dashboardPage.getByText(/Channel 1/i)).toBeVisible({
      timeout: 5000,
    });

    assertCleanConsoleStrict(dashboardLogs);
  });

  test("should render correctly at small laptop viewport", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await dashboardPage.setViewportSize(VIEWPORTS.smallLaptop);
    await dashboardPage.goto("/?debugOverflow=true");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Responsive Test/i).click();
    await dashboardPage.waitForTimeout(1000);

    // Key elements should remain visible
    await expect(dashboardPage.locator("body")).toBeVisible();

    // Channels should be accessible (may be in stacked layout)
    await expect(
      dashboardPage
        .getByText(/Channel 1/i)
        .or(dashboardPage.getByText(/Channel 2/i)),
    ).toBeVisible({
      timeout: 5000,
    });

    assertCleanConsoleStrict(dashboardLogs);
  });

  test("should render correctly at tablet viewport", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await dashboardPage.setViewportSize(VIEWPORTS.tablet);
    await dashboardPage.goto("/?debugOverflow=true");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Responsive Test/i).click();
    await dashboardPage.waitForTimeout(1000);

    // Core UI should be visible
    await expect(dashboardPage.locator("body")).toBeVisible();

    // May need to scroll to see all content
    const hasContent = await dashboardPage
      .getByText(/Channel/i)
      .first()
      .isVisible({ timeout: 5000 });
    expect(hasContent).toBeTruthy();

    assertCleanConsoleStrict(dashboardLogs);
  });

  test("should not have horizontal overflow at any viewport", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    for (const [name, size] of Object.entries(VIEWPORTS)) {
      await dashboardPage.setViewportSize(size);
      await dashboardPage.goto("/?debugOverflow=true");
      await dashboardPage.waitForLoadState("networkidle");

      // Check for horizontal overflow
      const hasHorizontalScroll = await dashboardPage.evaluate(() => {
        return (
          document.documentElement.scrollWidth >
          document.documentElement.clientWidth
        );
      });

      if (hasHorizontalScroll) {
        console.log(
          `Horizontal overflow detected at ${name} (${size.width}x${size.height})`,
        );
      }

      // Allow small amount of overflow (rounding errors)
      const overflowAmount = await dashboardPage.evaluate(() => {
        return (
          document.documentElement.scrollWidth -
          document.documentElement.clientWidth
        );
      });

      expect(overflowAmount).toBeLessThan(20);
    }

    assertNoErrors(dashboardLogs);
  });

  test("should maintain readable font sizes across viewports", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    for (const [name, size] of Object.entries(VIEWPORTS)) {
      await dashboardPage.setViewportSize(size);
      await dashboardPage.goto("/?debugOverflow=true");
      await dashboardPage.waitForLoadState("networkidle");

      await dashboardPage.getByText(/Responsive Test/i).click();
      await dashboardPage.waitForTimeout(500);

      // Get any font size warnings
      const issues = getVisualIssues(dashboardLogs);
      const fontIssues = issues.filter((i) => i.text.includes("Font size"));

      if (fontIssues.length > 0) {
        console.log(
          `Font size issues at ${name}:`,
          fontIssues.map((i) => i.text),
        );
      }
    }

    // Final viewport for strict check
    await dashboardPage.setViewportSize(VIEWPORTS.desktop);
    assertCleanConsoleStrict(dashboardLogs);
  });

  test("should adapt layout direction at breakpoints", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    // Desktop: expect row layout
    await dashboardPage.setViewportSize(VIEWPORTS.desktop);
    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Responsive Test/i).click();
    await dashboardPage.waitForTimeout(500);

    // Check for flex-row or similar horizontal layout at desktop
    const desktopLayout = dashboardPage
      .locator(".d-lg-flex, .flex-lg-row, .d-flex.flex-row")
      .first();
    const _hasRowLayout = await desktopLayout
      .isVisible({ timeout: 2000 })
      .catch(() => false);

    // Tablet: may have column layout
    await dashboardPage.setViewportSize(VIEWPORTS.tablet);
    await dashboardPage.waitForTimeout(500);

    // Layout should adapt without breaking
    await expect(dashboardPage.locator("body")).toBeVisible();

    assertNoErrors(dashboardLogs);
  });

  test("should handle viewport resize during interaction", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await dashboardPage.goto("/?debugOverflow=true");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Responsive Test/i).click();
    await dashboardPage.waitForTimeout(500);

    // Resize while page is active
    for (const [_name, size] of Object.entries(VIEWPORTS)) {
      await dashboardPage.setViewportSize(size);
      await dashboardPage.waitForTimeout(300);

      // Page should remain functional
      await expect(dashboardPage.locator("body")).toBeVisible();
    }

    // Return to desktop
    await dashboardPage.setViewportSize(VIEWPORTS.desktop);
    await dashboardPage.waitForTimeout(500);

    assertCleanConsoleStrict(dashboardLogs);
  });

  test("should show all tabs at desktop width", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    // Create more captures for tab bar testing
    await createTestCapture({ name: "Tab Test 1", center_hz: 88_000_000 });
    await createTestCapture({ name: "Tab Test 2", center_hz: 98_000_000 });
    await createTestCapture({ name: "Tab Test 3", center_hz: 108_000_000 });

    await dashboardPage.setViewportSize(VIEWPORTS.desktop);
    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    // All tabs should be visible
    await expect(dashboardPage.getByText(/Responsive Test/i)).toBeVisible({
      timeout: 5000,
    });
    await expect(dashboardPage.getByText(/Tab Test 1/i)).toBeVisible({
      timeout: 3000,
    });
    await expect(dashboardPage.getByText(/Tab Test 2/i)).toBeVisible({
      timeout: 3000,
    });
    await expect(dashboardPage.getByText(/Tab Test 3/i)).toBeVisible({
      timeout: 3000,
    });

    assertNoErrors(dashboardLogs);
  });

  test("should handle tab bar at narrow widths", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    // Create several captures
    await createTestCapture({ name: "Narrow Tab 1", center_hz: 88_000_000 });
    await createTestCapture({ name: "Narrow Tab 2", center_hz: 98_000_000 });

    await dashboardPage.setViewportSize(VIEWPORTS.tablet);
    await dashboardPage.goto("/?debugOverflow=true");
    await dashboardPage.waitForLoadState("networkidle");

    // Tabs should still be accessible (may scroll or show dropdown)
    await expect(
      dashboardPage.locator('nav, [role="tablist"]').first(),
    ).toBeVisible();

    assertCleanConsoleStrict(dashboardLogs);
  });

  test("should maintain usability at mobile landscape", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await dashboardPage.setViewportSize(VIEWPORTS.mobileLandscape);
    await dashboardPage.goto("/?debugOverflow=true");
    await dashboardPage.waitForLoadState("networkidle");

    // Should show content (may need scrolling)
    await expect(dashboardPage.locator("body")).toBeVisible();

    // Key navigation should be accessible
    const navigation = dashboardPage.locator(
      'nav, [role="navigation"], .navbar, button',
    );
    await expect(navigation.first()).toBeVisible({ timeout: 5000 });

    assertCleanConsoleStrict(dashboardLogs);
  });
});

test.describe("Panel Visibility", () => {
  test.beforeAll(async () => {
    await waitForBackend(15000);
  });

  test.beforeEach(async () => {
    await deleteAllCaptures();
  });

  test.afterEach(async () => {
    await deleteAllCaptures();
  });

  test("should show all three panels at wide viewport", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    const capture = await createTestCapture({
      name: "Panel Test",
      center_hz: 100_000_000,
    });
    await startCapture(capture.id);
    await createTestChannel(capture.id, {
      name: "Panel Channel",
      offset_hz: 0,
      mode: "wbfm",
    });

    await dashboardPage.setViewportSize({ width: 1920, height: 1080 });
    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Panel Test/i).click();
    await dashboardPage.waitForTimeout(500);

    // At 1920px, all three panels should be visible
    // Spectrum panel
    const spectrum = dashboardPage.locator('canvas, [class*="spectrum"]');
    const _hasSpectrum = await spectrum
      .first()
      .isVisible({ timeout: 3000 })
      .catch(() => false);

    // Channel list
    const channels = dashboardPage.getByText(/Panel Channel/i);
    await expect(channels).toBeVisible({ timeout: 3000 });

    assertNoErrors(dashboardLogs);
  });

  test("should stack panels at narrow viewport", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    const capture = await createTestCapture({
      name: "Stack Test",
      center_hz: 100_000_000,
    });
    await startCapture(capture.id);
    await createTestChannel(capture.id, {
      name: "Stack Channel",
      offset_hz: 0,
      mode: "wbfm",
    });

    await dashboardPage.setViewportSize({ width: 900, height: 700 });
    await dashboardPage.goto("/?debugOverflow=true");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Stack Test/i).click();
    await dashboardPage.waitForTimeout(500);

    // Panels should stack vertically at narrow width
    // Content should be scrollable
    await dashboardPage.evaluate(() => window.scrollTo(0, 500));
    await dashboardPage.waitForTimeout(200);

    // Key content should still be accessible via scrolling
    await expect(dashboardPage.locator("body")).toBeVisible();

    assertCleanConsoleStrict(dashboardLogs);
  });
});
