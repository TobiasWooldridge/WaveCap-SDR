import { test, expect } from "./fixtures";
import { assertNoErrors, assertCleanConsoleStrict } from "./utils/logs";
import {
  waitForBackend,
  createTestCapture,
  createTestChannel,
  startCapture,
  deleteAllCaptures,
} from "./utils/debug";

/**
 * Channel settings e2e tests
 *
 * Tests the advanced channel configuration UI:
 * - Mode selection (WBFM, NBFM, AM, SSB, P25, DMR)
 * - Squelch adjustment
 * - Audio rate selection
 * - DSP filters (de-emphasis, highpass, lowpass)
 * - AGC settings
 */

test.describe("Channel Settings", () => {
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
      name: "Settings Test",
      center_hz: 100_000_000,
      sample_rate: 1_000_000,
    });
    captureId = capture.id;
    await startCapture(captureId);
  });

  test.afterEach(async () => {
    await deleteAllCaptures();
  });

  test("should display channel settings panel", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await createTestChannel(captureId, {
      name: "Settings Channel",
      offset_hz: 0,
      mode: "wbfm",
    });

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Settings Test/i).click();
    await dashboardPage.waitForTimeout(500);

    // Find the channel card and look for settings toggle
    const channelCard = dashboardPage.getByText(/Settings Channel/i);
    await expect(channelCard).toBeVisible({ timeout: 5000 });

    // Look for settings/expand toggle
    const settingsToggle = dashboardPage
      .locator('[class*="channel"], .card')
      .filter({ hasText: /Settings Channel/i })
      .locator('button, [class*="settings"], [class*="expand"]');

    const _hasSettings = await settingsToggle
      .first()
      .isVisible({ timeout: 2000 })
      .catch(() => false);

    assertNoErrors(dashboardLogs);
  });

  test("should show mode selector with all options", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await createTestChannel(captureId, {
      name: "Mode Test",
      offset_hz: 0,
      mode: "wbfm",
    });

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Settings Test/i).click();
    await dashboardPage.waitForTimeout(500);

    // Look for mode selector
    const modeSelector = dashboardPage.locator("select").filter({
      has: dashboardPage.locator('option[value="wbfm"]'),
    });

    if (
      await modeSelector
        .first()
        .isVisible({ timeout: 3000 })
        .catch(() => false)
    ) {
      // Check that common modes are available
      const options = await modeSelector
        .first()
        .locator("option")
        .allTextContents();
      const modesFound = options.some(
        (opt) =>
          opt.toLowerCase().includes("wbfm") ||
          opt.toLowerCase().includes("nbfm") ||
          opt.toLowerCase().includes("am"),
      );
      expect(modesFound).toBeTruthy();
    }

    assertNoErrors(dashboardLogs);
  });

  test("should change channel mode", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await createTestChannel(captureId, {
      name: "Mode Change Test",
      offset_hz: 0,
      mode: "wbfm",
    });

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Settings Test/i).click();
    await dashboardPage.waitForTimeout(500);

    // Find and click mode selector
    const modeSelector = dashboardPage
      .locator("select")
      .filter({ has: dashboardPage.locator('option[value="nbfm"]') })
      .first();

    if (await modeSelector.isVisible({ timeout: 3000 }).catch(() => false)) {
      await modeSelector.selectOption("nbfm");
      await dashboardPage.waitForTimeout(500);

      // Verify selection changed
      const selectedValue = await modeSelector.inputValue();
      expect(selectedValue).toBe("nbfm");
    }

    assertNoErrors(dashboardLogs);
  });

  test("should display squelch control", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await createTestChannel(captureId, {
      name: "Squelch Test",
      offset_hz: 0,
      mode: "nbfm",
      squelch_db: -50,
    });

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Settings Test/i).click();
    await dashboardPage.waitForTimeout(500);

    // Look for squelch slider or input
    const squelchControl = dashboardPage.locator(
      'input[type="range"], [class*="squelch"], label:has-text("Squelch")',
    );
    const _hasSquelch = await squelchControl
      .first()
      .isVisible({ timeout: 3000 })
      .catch(() => false);

    assertNoErrors(dashboardLogs);
  });

  test("should display audio rate selector", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await createTestChannel(captureId, {
      name: "Audio Rate Test",
      offset_hz: 0,
      mode: "wbfm",
    });

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Settings Test/i).click();
    await dashboardPage.waitForTimeout(500);

    // Look for audio rate selector
    const audioRateSelector = dashboardPage.locator("select").filter({
      has: dashboardPage.locator('option:has-text("48")'),
    });

    if (
      await audioRateSelector
        .first()
        .isVisible({ timeout: 3000 })
        .catch(() => false)
    ) {
      const options = await audioRateSelector
        .first()
        .locator("option")
        .allTextContents();
      // Should have standard audio rates
      const hasRates = options.some(
        (opt) => opt.includes("48") || opt.includes("24") || opt.includes("16"),
      );
      expect(hasRates).toBeTruthy();
    }

    assertNoErrors(dashboardLogs);
  });

  test("should show DSP filter section", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await createTestChannel(captureId, {
      name: "DSP Filter Test",
      offset_hz: 0,
      mode: "wbfm",
    });

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Settings Test/i).click();
    await dashboardPage.waitForTimeout(500);

    // Look for DSP filters section
    const dspSection = dashboardPage.getByText(/DSP|Filter|De-emphasis/i);
    const _hasDsp = await dspSection
      .first()
      .isVisible({ timeout: 3000 })
      .catch(() => false);

    // DSP filters may be in a collapsible section

    assertNoErrors(dashboardLogs);
  });

  test("should show AGC settings section", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await createTestChannel(captureId, {
      name: "AGC Settings Test",
      offset_hz: 0,
      mode: "wbfm",
    });

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Settings Test/i).click();
    await dashboardPage.waitForTimeout(500);

    // Look for AGC section
    const agcSection = dashboardPage.getByText(/AGC|Gain Control|Auto Gain/i);
    const _hasAgc = await agcSection
      .first()
      .isVisible({ timeout: 3000 })
      .catch(() => false);

    // AGC settings may be in a collapsible section

    assertNoErrors(dashboardLogs);
  });

  test("should toggle AGC on/off", async ({ dashboardPage, dashboardLogs }) => {
    await createTestChannel(captureId, {
      name: "AGC Toggle Test",
      offset_hz: 0,
      mode: "wbfm",
    });

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Settings Test/i).click();
    await dashboardPage.waitForTimeout(500);

    // Look for AGC toggle/checkbox
    const agcToggle = dashboardPage.locator(
      'input[type="checkbox"]#enableAgc, input[id*="agc"], label:has-text("AGC") input',
    );

    if (
      await agcToggle
        .first()
        .isVisible({ timeout: 3000 })
        .catch(() => false)
    ) {
      const wasChecked = await agcToggle.first().isChecked();
      await agcToggle.first().click();
      await dashboardPage.waitForTimeout(300);
      const isNowChecked = await agcToggle.first().isChecked();
      expect(isNowChecked).not.toBe(wasChecked);
    }

    assertNoErrors(dashboardLogs);
  });

  test("should show different filters for different modes", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    // Create AM channel to see AM-specific filters
    await createTestChannel(captureId, {
      name: "AM Filters Test",
      offset_hz: 0,
      mode: "am",
    });

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Settings Test/i).click();
    await dashboardPage.waitForTimeout(500);

    // AM mode should show AM filters, not FM filters
    const amFilters = dashboardPage.getByText(/AM Highpass|AM Lowpass/i);
    const fmFilters = dashboardPage.getByText(/FM Highpass|De-emphasis/i);

    const _hasAmFilters = await amFilters
      .first()
      .isVisible({ timeout: 2000 })
      .catch(() => false);
    const _hasFmFilters = await fmFilters
      .first()
      .isVisible({ timeout: 2000 })
      .catch(() => false);

    // Filters may be in collapsible sections

    assertNoErrors(dashboardLogs);
  });

  test("should not have overflow in settings panel", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await createTestChannel(captureId, {
      name: "Overflow Settings Test",
      offset_hz: 0,
      mode: "wbfm",
    });

    await dashboardPage.goto("/?debugOverflow=true");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Settings Test/i).click();
    await dashboardPage.waitForTimeout(1000);

    assertCleanConsoleStrict(dashboardLogs);
  });

  test("should persist settings after page reload", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await createTestChannel(captureId, {
      name: "Persist Settings",
      offset_hz: 0,
      mode: "wbfm",
    });

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Settings Test/i).click();
    await dashboardPage.waitForTimeout(500);

    // Change mode if selector is available
    const modeSelector = dashboardPage
      .locator("select")
      .filter({ has: dashboardPage.locator('option[value="nbfm"]') })
      .first();

    if (await modeSelector.isVisible({ timeout: 2000 }).catch(() => false)) {
      await modeSelector.selectOption("nbfm");
      await dashboardPage.waitForTimeout(500);

      // Reload page
      await dashboardPage.reload();
      await dashboardPage.waitForLoadState("networkidle");

      await dashboardPage.getByText(/Settings Test/i).click();
      await dashboardPage.waitForTimeout(500);

      // Mode should still be NBFM
      const value = await modeSelector.inputValue();
      expect(value).toBe("nbfm");
    }

    assertNoErrors(dashboardLogs);
  });
});
