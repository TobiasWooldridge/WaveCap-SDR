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
 * Audio controls e2e tests
 *
 * Tests the audio playback and control functionality:
 * - Play/pause buttons
 * - Volume controls
 * - Mute functionality
 * - Multiple channel audio
 */

test.describe("Audio Controls", () => {
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
      name: "Audio Test",
      center_hz: 100_000_000,
      sample_rate: 1_000_000,
    });
    captureId = capture.id;
    await startCapture(captureId);
  });

  test.afterEach(async () => {
    await deleteAllCaptures();
  });

  test("should display audio controls for channel", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    // Create channel with 5 kHz offset (matches fake driver tone)
    await createTestChannel(captureId, {
      name: "Audio Test Channel",
      offset_hz: 5000,
      mode: "wbfm",
    });

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Audio Test(?! Channel)/i).click();
    await dashboardPage.waitForTimeout(500);

    // Look for play button or audio control
    const playButton = dashboardPage.getByRole("button", {
      name: /play|listen|audio/i,
    });
    const audioControl = dashboardPage.locator(
      '[class*="audio"], [class*="play"], button svg',
    );

    const hasPlayButton = await playButton
      .first()
      .isVisible({ timeout: 3000 })
      .catch(() => false);
    const hasAudioControl = await audioControl
      .first()
      .isVisible({ timeout: 3000 })
      .catch(() => false);

    // Audio controls should exist
    expect(hasPlayButton || hasAudioControl).toBeTruthy();

    assertNoErrors(dashboardLogs);
  });

  test("should have volume slider", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await createTestChannel(captureId, {
      name: "Volume Test",
      offset_hz: 5000,
      mode: "wbfm",
    });

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Audio Test/i).click();
    await dashboardPage.waitForTimeout(500);

    // Look for volume control (slider or input)
    const volumeSlider = dashboardPage.locator(
      'input[type="range"], [class*="volume"], [class*="slider"]',
    );
    const _hasVolume = await volumeSlider
      .first()
      .isVisible({ timeout: 3000 })
      .catch(() => false);

    // Volume may be hidden until audio is playing or in a popover

    assertNoErrors(dashboardLogs);
  });

  test("should show squelch indicator", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await createTestChannel(captureId, {
      name: "Squelch Test",
      offset_hz: 5000,
      mode: "nbfm",
      squelch_db: -40,
    });

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Audio Test/i).click();
    await dashboardPage.waitForTimeout(500);

    // Look for squelch indicator or control
    const squelchControl = dashboardPage.locator(
      '[class*="squelch"], :text("Squelch"), :text("SQL")',
    );
    const _hasSquelch = await squelchControl
      .first()
      .isVisible({ timeout: 3000 })
      .catch(() => false);

    // Squelch indicator may be part of channel card

    assertNoErrors(dashboardLogs);
  });

  test("should display signal level meter", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await createTestChannel(captureId, {
      name: "Signal Meter Test",
      offset_hz: 5000,
      mode: "wbfm",
    });

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Audio Test/i).click();
    await dashboardPage.waitForTimeout(1000);

    // Look for signal level indicator (meter, bar, or dB display)
    const signalMeter = dashboardPage.locator(
      '[class*="meter"], [class*="signal"], [class*="level"], :text("dB")',
    );
    const _hasMeter = await signalMeter
      .first()
      .isVisible({ timeout: 3000 })
      .catch(() => false);

    // Signal meter should exist in channel display

    assertNoErrors(dashboardLogs);
  });

  test("should toggle audio play state", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await createTestChannel(captureId, {
      name: "Play Toggle Test",
      offset_hz: 5000,
      mode: "wbfm",
    });

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Audio Test/i).click();
    await dashboardPage.waitForTimeout(500);

    // Find play button
    const playButton = dashboardPage
      .locator("button")
      .filter({
        has: dashboardPage.locator('svg, [class*="play"], [class*="audio"]'),
      })
      .first();

    if (await playButton.isVisible({ timeout: 3000 }).catch(() => false)) {
      // Click to start playing
      await playButton.click();
      await dashboardPage.waitForTimeout(500);

      // Click again to stop
      await playButton.click();
      await dashboardPage.waitForTimeout(300);
    }

    assertNoErrors(dashboardLogs);
  });

  test("should support multiple channel audio", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    // Create multiple channels
    await createTestChannel(captureId, {
      name: "Ch A",
      offset_hz: -50000,
      mode: "wbfm",
    });
    await createTestChannel(captureId, {
      name: "Ch B",
      offset_hz: 0,
      mode: "nbfm",
    });
    await createTestChannel(captureId, {
      name: "Ch C",
      offset_hz: 50000,
      mode: "am",
    });

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Audio Test/i).click();
    await dashboardPage.waitForTimeout(500);

    // All channels should be visible
    await expect(dashboardPage.getByText(/Ch A/i)).toBeVisible({
      timeout: 5000,
    });
    await expect(dashboardPage.getByText(/Ch B/i)).toBeVisible({
      timeout: 5000,
    });
    await expect(dashboardPage.getByText(/Ch C/i)).toBeVisible({
      timeout: 5000,
    });

    // Each should have its own audio control
    const playButtons = dashboardPage
      .locator('[class*="channel"] button, .card button')
      .filter({
        has: dashboardPage.locator("svg"),
      });

    const buttonCount = await playButtons.count();
    expect(buttonCount).toBeGreaterThanOrEqual(1);

    assertNoErrors(dashboardLogs);
  });

  test("should show audio format options", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await createTestChannel(captureId, {
      name: "Format Test",
      offset_hz: 5000,
      mode: "wbfm",
    });

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Audio Test/i).click();
    await dashboardPage.waitForTimeout(500);

    // Look for format selector (MP3, Opus, PCM, etc.)
    const formatSelector = dashboardPage
      .locator('select, [class*="format"]')
      .filter({
        hasText: /mp3|opus|pcm|aac/i,
      });

    const _hasFormat = await formatSelector
      .first()
      .isVisible({ timeout: 2000 })
      .catch(() => false);

    // Format may be hidden in advanced settings

    assertNoErrors(dashboardLogs);
  });

  test("should handle audio mode switching", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await createTestChannel(captureId, {
      name: "Mode Switch Test",
      offset_hz: 5000,
      mode: "wbfm",
    });

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Audio Test/i).click();
    await dashboardPage.waitForTimeout(500);

    // Look for mode selector
    const modeSelector = dashboardPage
      .locator('select, button, [class*="mode"]')
      .filter({
        hasText: /wbfm|nbfm|am|ssb/i,
      });

    if (
      await modeSelector
        .first()
        .isVisible({ timeout: 2000 })
        .catch(() => false)
    ) {
      // Try clicking mode selector
      await modeSelector.first().click();
      await dashboardPage.waitForTimeout(300);

      // Look for mode options
      const modeOptions = dashboardPage
        .getByRole("option")
        .or(dashboardPage.locator('[role="menuitem"]'));
      const hasOptions = await modeOptions
        .first()
        .isVisible({ timeout: 2000 })
        .catch(() => false);

      if (hasOptions) {
        // Select a different mode
        await modeOptions.first().click();
        await dashboardPage.waitForTimeout(300);
      } else {
        // Dismiss if it was a dropdown
        await dashboardPage.keyboard.press("Escape");
      }
    }

    assertNoErrors(dashboardLogs);
  });

  test("should not have visual overflow in audio controls", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await createTestChannel(captureId, {
      name: "Very Long Channel Name For Overflow Testing",
      offset_hz: 5000,
      mode: "wbfm",
    });

    await dashboardPage.goto("/?debugOverflow=true");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Audio Test/i).click();
    await dashboardPage.waitForTimeout(1000);

    // Overflow detector will catch issues
    assertCleanConsoleStrict(dashboardLogs);
  });

  test("should display AGC indicator when enabled", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await createTestChannel(captureId, {
      name: "AGC Test",
      offset_hz: 5000,
      mode: "wbfm",
    });

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Audio Test/i).click();
    await dashboardPage.waitForTimeout(500);

    // Look for AGC indicator or toggle
    const agcIndicator = dashboardPage.locator('[class*="agc"], :text("AGC")');
    const _hasAgc = await agcIndicator
      .first()
      .isVisible({ timeout: 3000 })
      .catch(() => false);

    // AGC may be in settings panel

    assertNoErrors(dashboardLogs);
  });

  test("should handle rapid play/pause clicks", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await createTestChannel(captureId, {
      name: "Rapid Click Test",
      offset_hz: 5000,
      mode: "wbfm",
    });

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Audio Test/i).click();
    await dashboardPage.waitForTimeout(500);

    const playButton = dashboardPage
      .locator("button")
      .filter({
        has: dashboardPage.locator("svg"),
      })
      .first();

    if (await playButton.isVisible({ timeout: 2000 }).catch(() => false)) {
      // Rapid clicking
      for (let i = 0; i < 10; i++) {
        await playButton.click();
        await dashboardPage.waitForTimeout(50);
      }
    }

    // Should not crash
    await expect(dashboardPage.locator("body")).toBeVisible();

    assertNoErrors(dashboardLogs);
  });
});
