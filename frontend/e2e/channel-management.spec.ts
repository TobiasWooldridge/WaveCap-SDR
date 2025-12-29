import { test, expect } from "./fixtures";
import { assertNoErrors, assertCleanConsoleStrict } from "./utils/logs";
import {
  waitForBackend,
  createTestCapture,
  createTestChannel,
  deleteChannel,
  startCapture,
  updateChannel,
  getChannelsForCapture,
  deleteAllCaptures,
} from "./utils/debug";

/**
 * Channel management e2e tests
 *
 * Tests channel operations using the fake SDR driver:
 * - Channel creation with different modes
 * - Channel updates (offset, mode, squelch)
 * - Channel deletion
 * - Multiple channels per capture
 */

test.describe("Channel Management", () => {
  let captureId: string;

  test.beforeAll(async () => {
    const ready = await waitForBackend(15000);
    if (!ready) {
      throw new Error("Backend not available");
    }
  });

  test.beforeEach(async () => {
    // Clean slate and create a fresh capture for each test
    await deleteAllCaptures();
    const capture = await createTestCapture({
      name: "Channel Test Capture",
      center_hz: 100_000_000,
      sample_rate: 1_000_000,
    });
    captureId = capture.id;
    await startCapture(captureId);
  });

  test.afterEach(async () => {
    await deleteAllCaptures();
  });

  test("should create a WBFM channel", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    // Create WBFM channel at center frequency
    const channel = await createTestChannel(captureId, {
      name: "WBFM Test",
      offset_hz: 0,
      mode: "wbfm",
      squelch_db: -50,
    });

    expect(channel.id).toBeTruthy();
    expect(channel.mode).toBe("wbfm");

    // Navigate and select the capture
    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Channel Test Capture/i).click();
    await dashboardPage.waitForTimeout(500);

    // Channel should be visible in the channel list
    await expect(dashboardPage.getByText(/WBFM Test|WBFM/i)).toBeVisible({
      timeout: 5000,
    });

    assertNoErrors(dashboardLogs);
  });

  test("should create an NBFM channel", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    const channel = await createTestChannel(captureId, {
      name: "NBFM Test",
      offset_hz: 5000, // 5 kHz offset (matches fake driver tone)
      mode: "nbfm",
      squelch_db: -40,
    });

    expect(channel.mode).toBe("nbfm");

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Channel Test Capture/i).click();
    await dashboardPage.waitForTimeout(500);

    await expect(dashboardPage.getByText(/NBFM Test|NBFM/i)).toBeVisible({
      timeout: 5000,
    });

    assertNoErrors(dashboardLogs);
  });

  test("should create an AM channel", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    const channel = await createTestChannel(captureId, {
      name: "AM Test",
      offset_hz: 10000,
      mode: "am",
      squelch_db: -55,
    });

    expect(channel.mode).toBe("am");

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Channel Test Capture/i).click();
    await dashboardPage.waitForTimeout(500);

    await expect(dashboardPage.getByText(/AM Test|AM/i)).toBeVisible({
      timeout: 5000,
    });

    assertNoErrors(dashboardLogs);
  });

  test("should create multiple channels", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    // Create several channels with different offsets
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
    await createTestChannel(captureId, {
      name: "Channel 3",
      offset_hz: 50000,
      mode: "am",
    });

    // Verify via API
    const channels = await getChannelsForCapture(captureId);
    expect(channels.length).toBe(3);

    // Navigate and verify in UI
    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Channel Test Capture/i).click();
    await dashboardPage.waitForTimeout(500);

    // All channels should be visible
    await expect(dashboardPage.getByText(/Channel 1/i)).toBeVisible({
      timeout: 5000,
    });
    await expect(dashboardPage.getByText(/Channel 2/i)).toBeVisible({
      timeout: 5000,
    });
    await expect(dashboardPage.getByText(/Channel 3/i)).toBeVisible({
      timeout: 5000,
    });

    assertNoErrors(dashboardLogs);
  });

  test("should update channel offset", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    const channel = await createTestChannel(captureId, {
      name: "Offset Test",
      offset_hz: 0,
      mode: "wbfm",
    });

    // Update offset via API
    const updated = await updateChannel(channel.id, {
      offsetHz: 25000,
    });
    expect(updated.offsetHz).toBe(25000);

    // Navigate and verify
    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Channel Test Capture/i).click();
    await dashboardPage.waitForTimeout(500);

    // Channel should still be visible
    await expect(dashboardPage.getByText(/Offset Test/i)).toBeVisible({
      timeout: 5000,
    });

    assertNoErrors(dashboardLogs);
  });

  test("should update channel mode", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    const channel = await createTestChannel(captureId, {
      name: "Mode Change Test",
      offset_hz: 0,
      mode: "wbfm",
    });

    // Change mode to AM
    const updated = await updateChannel(channel.id, {
      mode: "am",
    });
    expect(updated.mode).toBe("am");

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Channel Test Capture/i).click();
    await dashboardPage.waitForTimeout(500);

    // Look for AM mode indicator
    await expect(dashboardPage.getByText(/Mode Change Test/i)).toBeVisible({
      timeout: 5000,
    });

    assertNoErrors(dashboardLogs);
  });

  test("should update channel squelch", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    const channel = await createTestChannel(captureId, {
      name: "Squelch Test",
      offset_hz: 5000,
      mode: "nbfm",
      squelch_db: -60,
    });

    // Update squelch
    const updated = await updateChannel(channel.id, {
      squelchDb: -30,
    });
    expect(updated.squelchDb).toBe(-30);

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Channel Test Capture/i).click();
    await dashboardPage.waitForTimeout(500);

    await expect(dashboardPage.getByText(/Squelch Test/i)).toBeVisible({
      timeout: 5000,
    });

    assertNoErrors(dashboardLogs);
  });

  test("should delete a channel", async ({ dashboardPage, dashboardLogs }) => {
    // Create two channels
    const _channel1 = await createTestChannel(captureId, {
      name: "Keep This",
      offset_hz: 0,
      mode: "wbfm",
    });
    const channel2 = await createTestChannel(captureId, {
      name: "Delete This",
      offset_hz: 25000,
      mode: "nbfm",
    });

    // Navigate and verify both exist
    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Channel Test Capture/i).click();
    await dashboardPage.waitForTimeout(500);

    await expect(dashboardPage.getByText(/Keep This/i)).toBeVisible({
      timeout: 5000,
    });
    await expect(dashboardPage.getByText(/Delete This/i)).toBeVisible({
      timeout: 5000,
    });

    // Delete via API
    await deleteChannel(channel2.id);

    // Refresh and verify
    await dashboardPage.reload();
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Channel Test Capture/i).click();
    await dashboardPage.waitForTimeout(500);

    await expect(dashboardPage.getByText(/Keep This/i)).toBeVisible({
      timeout: 5000,
    });
    await expect(dashboardPage.getByText(/Delete This/i)).not.toBeVisible({
      timeout: 3000,
    });

    assertNoErrors(dashboardLogs);
  });

  test("should rename a channel", async ({ dashboardPage, dashboardLogs }) => {
    const channel = await createTestChannel(captureId, {
      name: "Original Name",
      offset_hz: 0,
      mode: "wbfm",
    });

    // Update name
    await updateChannel(channel.id, {
      name: "New Name",
    });

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Channel Test Capture/i).click();
    await dashboardPage.waitForTimeout(500);

    // Should show new name
    await expect(dashboardPage.getByText(/New Name/i)).toBeVisible({
      timeout: 5000,
    });
    // Old name should not be visible
    await expect(dashboardPage.getByText("Original Name")).not.toBeVisible({
      timeout: 2000,
    });

    assertNoErrors(dashboardLogs);
  });

  test("should handle channel with visual overflow detection", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    // Create channel with a long name to test overflow handling
    await createTestChannel(captureId, {
      name: "Very Long Channel Name That Could Cause Overflow Issues In The UI",
      offset_hz: 0,
      mode: "wbfm",
    });

    await dashboardPage.goto("/?debugOverflow=true");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Channel Test Capture/i).click();
    await dashboardPage.waitForTimeout(1000);

    // Visual issues would be caught by assertCleanConsoleStrict
    assertCleanConsoleStrict(dashboardLogs);
  });
});
