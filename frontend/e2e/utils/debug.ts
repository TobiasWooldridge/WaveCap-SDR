/**
 * Debug API utilities for e2e tests.
 *
 * These utilities allow tests to:
 * - Query backend state (captures, channels, devices)
 * - Execute code on the frontend page
 * - Inject CSS for real-time debugging
 * - Query DOM state remotely
 */

import type { Page } from "@playwright/test";
import { config } from "dotenv";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));

// Load .env from frontend root
config({ path: resolve(__dirname, "../../.env") });

// Debug API configuration - defaults to 8088 for development
const BACKEND_PORT = process.env["BACKEND_PORT"] ?? "8088";
const BACKEND_URL =
  process.env["BACKEND_URL"] ?? `http://localhost:${BACKEND_PORT}`;

/**
 * Get the DEBUG_SECRET from environment or .env file
 */
async function getDebugSecret(): Promise<string> {
  // First try environment variable
  if (process.env["DEBUG_SECRET"]) {
    return process.env["DEBUG_SECRET"];
  }

  // Try to read from .env files
  const fs = await import("fs/promises");
  const path = await import("path");

  const envPaths = [
    path.join(process.cwd(), "../backend/.env"),
    path.join(process.cwd(), ".env"),
  ];

  for (const envPath of envPaths) {
    try {
      const envContent = await fs.readFile(envPath, "utf-8");
      const match = envContent.match(/DEBUG_SECRET=(.+)/);
      if (match) {
        return match[1].trim();
      }
    } catch {
      // .env file doesn't exist or can't be read
    }
  }

  // Return empty string if no secret found (debug endpoints may not require auth)
  return "";
}

/**
 * Make an authenticated request to the debug API
 */
async function debugFetch(
  endpoint: string,
  options: RequestInit = {},
): Promise<Response> {
  const secret = await getDebugSecret();

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };

  if (secret) {
    headers["Authorization"] = `Bearer ${secret}`;
  }

  return fetch(`${BACKEND_URL}${endpoint}`, {
    ...options,
    headers: {
      ...headers,
      ...options.headers,
    },
  });
}

// ============================================================================
// Backend State Queries
// ============================================================================

/**
 * Get all captures from the backend
 */
export async function getCaptures(): Promise<unknown[]> {
  const response = await debugFetch("/api/v1/captures");
  const data = await response.json();
  return data;
}

/**
 * Get all channels from the backend
 */
export async function getChannels(): Promise<unknown[]> {
  const response = await debugFetch("/api/v1/channels");
  const data = await response.json();
  return data;
}

/**
 * Get device list
 */
export async function getDevices(): Promise<unknown[]> {
  const response = await debugFetch("/api/v1/devices");
  const data = await response.json();
  return data;
}

/**
 * Get backend server state
 */
export async function getServerState(): Promise<{
  captures: unknown[];
  channels: unknown[];
  devices: unknown[];
}> {
  const [captures, channels, devices] = await Promise.all([
    getCaptures(),
    getChannels(),
    getDevices(),
  ]);
  return { captures, channels, devices };
}

// ============================================================================
// Frontend Page Evaluation
// ============================================================================

/**
 * Execute code on the frontend page
 */
export async function evalOnPage(page: Page, code: string): Promise<unknown> {
  return page.evaluate((codeStr) => {
    // Execute the code string as a function
    return new Function(codeStr)();
  }, code);
}

/**
 * Get React component state (if using React DevTools)
 */
export async function getReactState(
  page: Page,
  selector: string,
): Promise<unknown> {
  return page.evaluate((sel) => {
    const element = document.querySelector(sel);
    if (!element) return null;

    // Try to get React fiber
    const key = Object.keys(element).find((k) => k.startsWith("__reactFiber$"));
    if (!key) return null;

    const fiber = (element as Record<string, unknown>)[key] as {
      memoizedProps?: unknown;
    };
    return fiber?.memoizedProps ?? null;
  }, selector);
}

/**
 * Query DOM elements and get their computed styles
 */
export async function getElementStyles(
  page: Page,
  selector: string,
): Promise<
  Array<{
    selector: string;
    tagName: string;
    className: string;
    bounds: DOMRect;
    styles: Record<string, string>;
  }>
> {
  return page.evaluate((sel) => {
    const elements = document.querySelectorAll(sel);
    return Array.from(elements).map((el) => {
      const htmlEl = el as HTMLElement;
      const computed = getComputedStyle(htmlEl);
      return {
        selector: sel,
        tagName: htmlEl.tagName.toLowerCase(),
        className: htmlEl.className,
        bounds: htmlEl.getBoundingClientRect().toJSON() as DOMRect,
        styles: {
          display: computed.display,
          position: computed.position,
          overflow: computed.overflow,
          width: computed.width,
          height: computed.height,
          fontSize: computed.fontSize,
          color: computed.color,
          backgroundColor: computed.backgroundColor,
        },
      };
    });
  }, selector);
}

/**
 * Check if elements are overlapping
 */
export async function checkElementOverlap(
  page: Page,
  selector1: string,
  selector2: string,
): Promise<{
  overlapping: boolean;
  overlapArea: number;
  bounds1: DOMRect;
  bounds2: DOMRect;
}> {
  return page.evaluate(
    ({ sel1, sel2 }) => {
      const el1 = document.querySelector(sel1);
      const el2 = document.querySelector(sel2);

      if (!el1 || !el2) {
        return {
          overlapping: false,
          overlapArea: 0,
          bounds1: new DOMRect(),
          bounds2: new DOMRect(),
        };
      }

      const rect1 = el1.getBoundingClientRect();
      const rect2 = el2.getBoundingClientRect();

      const overlapX = Math.max(
        0,
        Math.min(rect1.right, rect2.right) - Math.max(rect1.left, rect2.left),
      );
      const overlapY = Math.max(
        0,
        Math.min(rect1.bottom, rect2.bottom) - Math.max(rect1.top, rect2.top),
      );
      const overlapArea = overlapX * overlapY;

      return {
        overlapping: overlapArea > 0,
        overlapArea,
        bounds1: rect1.toJSON() as DOMRect,
        bounds2: rect2.toJSON() as DOMRect,
      };
    },
    { sel1: selector1, sel2: selector2 },
  );
}

// ============================================================================
// CSS Injection for Debugging
// ============================================================================

/**
 * Inject CSS into the page for debugging
 */
export async function injectCSS(page: Page, css: string): Promise<void> {
  await page.evaluate((cssContent) => {
    const styleId = "debug-injected-styles";
    let style = document.getElementById(styleId) as HTMLStyleElement | null;
    if (!style) {
      style = document.createElement("style");
      style.id = styleId;
      document.head.appendChild(style);
    }
    style.textContent = cssContent;
  }, css);
}

/**
 * Remove injected debug CSS
 */
export async function removeInjectedCSS(page: Page): Promise<void> {
  await page.evaluate(() => {
    const style = document.getElementById("debug-injected-styles");
    if (style) {
      style.remove();
    }
  });
}

/**
 * Highlight elements matching a selector
 */
export async function highlightElements(
  page: Page,
  selector: string,
  color = "rgba(255, 0, 0, 0.3)",
): Promise<void> {
  await page.evaluate(
    ({ sel, highlightColor }) => {
      const elements = document.querySelectorAll(sel);
      elements.forEach((el) => {
        const htmlEl = el as HTMLElement;
        htmlEl.style.outline = `2px solid ${highlightColor}`;
        htmlEl.style.backgroundColor = highlightColor;
      });
    },
    { sel: selector, highlightColor: color },
  );
}

// ============================================================================
// Capture & Channel Control
// ============================================================================

/**
 * Capture creation response
 */
export interface CaptureResponse {
  id: string;
  name?: string;
  autoName?: string;
  deviceId: string;
  centerHz: number;
  sampleRate: number;
  state: string;
}

/**
 * Channel creation response
 */
export interface ChannelResponse {
  id: string;
  captureId: string;
  name?: string;
  autoName?: string;
  mode: string;
  offsetHz: number;
  state: string;
}

/**
 * Create a test capture with fake driver
 */
export async function createTestCapture(options: {
  name?: string;
  center_hz?: number;
  sample_rate?: number;
  auto_start?: boolean;
}): Promise<CaptureResponse> {
  // Use fake0 device ID for the fake driver
  const response = await debugFetch("/api/v1/captures", {
    method: "POST",
    body: JSON.stringify({
      name: options.name ?? "Test Capture",
      deviceId: "fake0",
      centerHz: options.center_hz ?? 100_000_000,
      sampleRate: options.sample_rate ?? 1_000_000,
      createDefaultChannel: false,
    }),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Failed to create capture: ${response.status} ${error}`);
  }

  return response.json();
}

/**
 * Start a capture
 */
export async function startCapture(
  captureId: string,
): Promise<CaptureResponse> {
  const response = await debugFetch(`/api/v1/captures/${captureId}/start`, {
    method: "POST",
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Failed to start capture: ${response.status} ${error}`);
  }

  return response.json();
}

/**
 * Stop a capture
 */
export async function stopCapture(captureId: string): Promise<CaptureResponse> {
  const response = await debugFetch(`/api/v1/captures/${captureId}/stop`, {
    method: "POST",
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Failed to stop capture: ${response.status} ${error}`);
  }

  return response.json();
}

/**
 * Delete a capture
 */
export async function deleteCapture(captureId: string): Promise<void> {
  await debugFetch(`/api/v1/captures/${captureId}`, {
    method: "DELETE",
  });
}

/**
 * Create a test channel on a capture
 * Note: offset_hz is the offset from center frequency, not absolute frequency
 */
export async function createTestChannel(
  captureId: string,
  options: {
    name?: string;
    offset_hz?: number;
    mode?: string;
    squelch_db?: number;
  },
): Promise<ChannelResponse> {
  const response = await debugFetch(`/api/v1/captures/${captureId}/channels`, {
    method: "POST",
    body: JSON.stringify({
      name: options.name ?? "Test Channel",
      offsetHz: options.offset_hz ?? 0,
      mode: options.mode ?? "wbfm",
      squelchDb: options.squelch_db ?? -60,
    }),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Failed to create channel: ${response.status} ${error}`);
  }

  return response.json();
}

/**
 * Delete a channel
 */
export async function deleteChannel(channelId: string): Promise<void> {
  const response = await debugFetch(`/api/v1/channels/${channelId}`, {
    method: "DELETE",
  });

  if (!response.ok && response.status !== 404) {
    const error = await response.text();
    throw new Error(`Failed to delete channel: ${response.status} ${error}`);
  }
}

/**
 * Update a channel
 */
export async function updateChannel(
  channelId: string,
  updates: {
    name?: string;
    offsetHz?: number;
    mode?: string;
    squelchDb?: number;
  },
): Promise<ChannelResponse> {
  const response = await debugFetch(`/api/v1/channels/${channelId}`, {
    method: "PATCH",
    body: JSON.stringify(updates),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Failed to update channel: ${response.status} ${error}`);
  }

  return response.json();
}

/**
 * Get a specific capture
 */
export async function getCapture(
  captureId: string,
): Promise<CaptureResponse | null> {
  const response = await debugFetch(`/api/v1/captures/${captureId}`);

  if (response.status === 404) {
    return null;
  }

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Failed to get capture: ${response.status} ${error}`);
  }

  return response.json();
}

/**
 * Update a capture
 */
export async function updateCapture(
  captureId: string,
  updates: {
    centerHz?: number;
    sampleRate?: number;
    gain?: number;
    name?: string;
  },
): Promise<CaptureResponse> {
  const response = await debugFetch(`/api/v1/captures/${captureId}`, {
    method: "PATCH",
    body: JSON.stringify(updates),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Failed to update capture: ${response.status} ${error}`);
  }

  return response.json();
}

/**
 * Get channels for a capture
 */
export async function getChannelsForCapture(
  captureId: string,
): Promise<ChannelResponse[]> {
  const response = await debugFetch(`/api/v1/captures/${captureId}/channels`);

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Failed to get channels: ${response.status} ${error}`);
  }

  return response.json();
}

/**
 * Delete all captures (cleanup utility)
 */
export async function deleteAllCaptures(): Promise<void> {
  const captures = (await getCaptures()) as CaptureResponse[];
  for (const capture of captures) {
    try {
      await deleteCapture(capture.id);
    } catch {
      // Ignore errors during cleanup
    }
  }
}

/**
 * Wait for backend API to be available
 */
export async function waitForBackend(timeout = 10000): Promise<boolean> {
  const start = Date.now();
  while (Date.now() - start < timeout) {
    try {
      const response = await debugFetch("/api/v1/captures");
      if (response.ok) {
        return true;
      }
    } catch {
      await new Promise((r) => setTimeout(r, 100));
    }
  }
  return false;
}
