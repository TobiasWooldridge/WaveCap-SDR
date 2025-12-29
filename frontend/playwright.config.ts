import { defineConfig, devices } from "@playwright/test";
import { config } from "dotenv";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));

// Load .env.local for port configuration
config({ path: resolve(__dirname, ".env.local") });
config({ path: resolve(__dirname, ".env") });

// Port configuration with defaults
const FRONTEND_PORT = parseInt(process.env["FRONTEND_PORT"] || "5174", 10);
const _BACKEND_PORT = parseInt(process.env["BACKEND_PORT"] || "8088", 10);

/**
 * Playwright configuration for WaveCap-SDR e2e testing
 *
 * Tests run against:
 * - Frontend dev server on port 5174 (proxies to backend)
 * - Backend API server on port 8088
 *
 * @see https://playwright.dev/docs/test-configuration
 */
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: process.env.CI ? "github" : "list",

  use: {
    // Base URL for navigation
    baseURL: `http://localhost:${FRONTEND_PORT}`,
    // Collect trace on first retry
    trace: "on-first-retry",
    // Screenshot on failure
    screenshot: "only-on-failure",
  },

  projects: [
    // Desktop browser - primary for SDR dashboard
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
    {
      name: "firefox",
      use: { ...devices["Desktop Firefox"] },
    },
    {
      name: "webkit",
      use: { ...devices["Desktop Safari"] },
    },

    // Tablet for responsive testing
    {
      name: "tablet",
      use: { ...devices["iPad Pro 11"] },
    },
  ],

  // Start dev server before running tests
  webServer: [
    {
      command: "npm run dev -- --port 5174",
      url: `http://localhost:${FRONTEND_PORT}`,
      reuseExistingServer: !process.env.CI,
      timeout: 120 * 1000,
    },
  ],
});
