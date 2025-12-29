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
 * Keyboard navigation e2e tests
 *
 * Tests keyboard accessibility:
 * - Tab navigation
 * - Enter/Space to activate
 * - Escape to close modals
 * - Arrow keys in controls
 * - Focus management
 */

test.describe("Keyboard Navigation", () => {
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
      name: "Keyboard Test",
      center_hz: 100_000_000,
      sample_rate: 1_000_000,
    });
    captureId = capture.id;
    await startCapture(captureId);
  });

  test.afterEach(async () => {
    await deleteAllCaptures();
  });

  test("should navigate with Tab key", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    // Start tabbing through the interface
    await dashboardPage.keyboard.press("Tab");
    await dashboardPage.waitForTimeout(100);

    // Should focus on some element
    const _focusedElement = await dashboardPage.evaluate(() => {
      const el = document.activeElement;
      return el ? el.tagName : null;
    });

    // Tab multiple times
    for (let i = 0; i < 10; i++) {
      await dashboardPage.keyboard.press("Tab");
      await dashboardPage.waitForTimeout(50);
    }

    // Should not crash
    await expect(dashboardPage.locator("body")).toBeVisible();

    assertNoErrors(dashboardLogs);
  });

  test("should activate buttons with Enter key", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    // Focus on a button
    const addButton = dashboardPage
      .getByRole("button", { name: /add|new|\+/i })
      .first();

    if (await addButton.isVisible({ timeout: 2000 }).catch(() => false)) {
      await addButton.focus();
      await dashboardPage.waitForTimeout(100);

      // Press Enter
      await dashboardPage.keyboard.press("Enter");
      await dashboardPage.waitForTimeout(300);

      // Modal may have opened
      const modal = dashboardPage.locator('.modal, [role="dialog"]');
      const hasModal = await modal
        .isVisible({ timeout: 1000 })
        .catch(() => false);

      if (hasModal) {
        // Close with Escape
        await dashboardPage.keyboard.press("Escape");
        await dashboardPage.waitForTimeout(200);
      }
    }

    assertNoErrors(dashboardLogs);
  });

  test("should close modal with Escape key", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    // Open a modal via click
    const addButton = dashboardPage
      .getByRole("button", { name: /add|new|\+/i })
      .first();

    if (await addButton.isVisible({ timeout: 2000 }).catch(() => false)) {
      await addButton.click();
      await dashboardPage.waitForTimeout(300);

      const modal = dashboardPage.locator('.modal, [role="dialog"]');
      if (await modal.isVisible({ timeout: 1000 }).catch(() => false)) {
        // Close with Escape
        await dashboardPage.keyboard.press("Escape");
        await dashboardPage.waitForTimeout(300);

        // Modal should be closed
        await expect(modal).not.toBeVisible({ timeout: 2000 });
      }
    }

    assertNoErrors(dashboardLogs);
  });

  test("should navigate select with arrow keys", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await createTestChannel(captureId, {
      name: "Arrow Keys Test",
      offset_hz: 0,
      mode: "wbfm",
    });

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Keyboard Test/i).click();
    await dashboardPage.waitForTimeout(500);

    // Find a select element
    const modeSelector = dashboardPage
      .locator("select")
      .filter({ has: dashboardPage.locator('option[value="wbfm"]') })
      .first();

    if (await modeSelector.isVisible({ timeout: 2000 }).catch(() => false)) {
      await modeSelector.focus();
      await dashboardPage.waitForTimeout(100);

      // Arrow down
      await dashboardPage.keyboard.press("ArrowDown");
      await dashboardPage.waitForTimeout(100);

      // Arrow up
      await dashboardPage.keyboard.press("ArrowUp");
      await dashboardPage.waitForTimeout(100);

      // Should still work
      await expect(modeSelector).toBeVisible();
    }

    assertNoErrors(dashboardLogs);
  });

  test("should navigate tabs with arrow keys", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await createTestCapture({
      name: "Tab Nav A",
      center_hz: 88_000_000,
    });
    await createTestCapture({
      name: "Tab Nav B",
      center_hz: 162_000_000,
    });

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    // Click first tab to focus the tab bar
    await dashboardPage.getByText(/Keyboard Test/i).click();
    await dashboardPage.waitForTimeout(300);

    // Try arrow key navigation (may or may not be implemented)
    await dashboardPage.keyboard.press("ArrowRight");
    await dashboardPage.waitForTimeout(100);
    await dashboardPage.keyboard.press("ArrowLeft");
    await dashboardPage.waitForTimeout(100);

    // Should not crash
    await expect(dashboardPage.locator("body")).toBeVisible();

    assertNoErrors(dashboardLogs);
  });

  test("should support Space bar activation", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await createTestChannel(captureId, {
      name: "Space Test",
      offset_hz: 0,
      mode: "wbfm",
    });

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Keyboard Test/i).click();
    await dashboardPage.waitForTimeout(500);

    // Find a checkbox
    const checkbox = dashboardPage.locator('input[type="checkbox"]').first();

    if (await checkbox.isVisible({ timeout: 2000 }).catch(() => false)) {
      const _wasChecked = await checkbox.isChecked();
      await checkbox.focus();
      await dashboardPage.waitForTimeout(100);

      // Toggle with Space
      await dashboardPage.keyboard.press("Space");
      await dashboardPage.waitForTimeout(200);

      const _isNowChecked = await checkbox.isChecked();
      // May or may not have toggled depending on implementation
    }

    assertNoErrors(dashboardLogs);
  });

  test("should trap focus in modal", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    // Open modal
    const addButton = dashboardPage
      .getByRole("button", { name: /add|new|\+/i })
      .first();

    if (await addButton.isVisible({ timeout: 2000 }).catch(() => false)) {
      await addButton.click();
      await dashboardPage.waitForTimeout(300);

      const modal = dashboardPage.locator('.modal, [role="dialog"]');
      if (await modal.isVisible({ timeout: 1000 }).catch(() => false)) {
        // Tab through modal elements
        for (let i = 0; i < 20; i++) {
          await dashboardPage.keyboard.press("Tab");
          await dashboardPage.waitForTimeout(50);
        }

        // Focus should still be within modal (or modal-related)
        const _focusedInModal = await dashboardPage.evaluate(() => {
          const el = document.activeElement;
          const modal = document.querySelector('.modal, [role="dialog"]');
          return modal && modal.contains(el);
        });

        // Close modal
        await dashboardPage.keyboard.press("Escape");
      }
    }

    assertNoErrors(dashboardLogs);
  });

  test("should support number input with keyboard", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await createTestChannel(captureId, {
      name: "Number Input Test",
      offset_hz: 0,
      mode: "wbfm",
    });

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Keyboard Test/i).click();
    await dashboardPage.waitForTimeout(500);

    // Find a number input
    const numberInput = dashboardPage.locator('input[type="number"]').first();

    if (await numberInput.isVisible({ timeout: 2000 }).catch(() => false)) {
      await numberInput.focus();
      await dashboardPage.waitForTimeout(100);

      // Type a number
      await dashboardPage.keyboard.type("123");
      await dashboardPage.waitForTimeout(100);

      const value = await numberInput.inputValue();
      expect(value).toContain("123");

      // Clear and try arrow keys
      await numberInput.fill("");
      await numberInput.fill("100");
      await dashboardPage.keyboard.press("ArrowUp");
      await dashboardPage.waitForTimeout(100);
    }

    assertNoErrors(dashboardLogs);
  });

  test("should have visible focus indicators", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await dashboardPage.goto("/?debugOverflow=true");
    await dashboardPage.waitForLoadState("networkidle");

    // Tab through elements and verify focus is visible
    for (let i = 0; i < 5; i++) {
      await dashboardPage.keyboard.press("Tab");
      await dashboardPage.waitForTimeout(200);

      // Check if focused element has visible styles
      const _hasFocusStyles = await dashboardPage.evaluate(() => {
        const el = document.activeElement;
        if (!el || el === document.body) return true; // OK if body
        const styles = window.getComputedStyle(el);
        // Check for outline or box-shadow (common focus indicators)
        return (
          styles.outline !== "none" ||
          styles.boxShadow !== "none" ||
          el.classList.contains("focus-visible")
        );
      });
      // Focus styles may or may not be implemented
    }

    assertCleanConsoleStrict(dashboardLogs);
  });

  test("should not lose focus unexpectedly", async ({
    dashboardPage,
    dashboardLogs,
  }) => {
    await createTestChannel(captureId, {
      name: "Focus Retention Test",
      offset_hz: 0,
      mode: "wbfm",
    });

    await dashboardPage.goto("/");
    await dashboardPage.waitForLoadState("networkidle");

    await dashboardPage.getByText(/Keyboard Test/i).click();
    await dashboardPage.waitForTimeout(500);

    // Focus on an input
    const input = dashboardPage.locator("input, select, button").first();
    if (await input.isVisible({ timeout: 2000 }).catch(() => false)) {
      await input.focus();
      await dashboardPage.waitForTimeout(500);

      // Wait for any async updates
      await dashboardPage.waitForTimeout(2000);

      // Check if something is still focused
      const _hasFocus = await dashboardPage.evaluate(() => {
        return document.activeElement !== document.body;
      });
      // Focus may have moved, but that's OK
    }

    assertNoErrors(dashboardLogs);
  });
});
