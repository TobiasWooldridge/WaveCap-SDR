import type { Capture } from "../types";

export interface CaptureStatusMessage {
  title: string;
  subtitle: string;
  titleColor: string;
}

/**
 * Get appropriate status message based on capture state and connection status.
 * Used by SpectrumAnalyzer and WaterfallDisplay to show consistent messages.
 */
export function getCaptureStatusMessage(
  capture: Capture,
  isConnected: boolean
): CaptureStatusMessage {
  if (capture.state === "running") {
    // Capture is running but no data yet - connecting
    return {
      title: isConnected ? "Waiting for data..." : "Connecting...",
      subtitle: "Spectrum data will appear shortly",
      titleColor: "#0d6efd", // Bootstrap primary blue
    };
  }

  if (capture.state === "starting") {
    // Capture is starting up
    return {
      title: "Starting capture...",
      subtitle: "Initializing SDR device",
      titleColor: "#ffc107", // Bootstrap warning yellow
    };
  }

  if (capture.state === "stopping") {
    // Capture is stopping
    return {
      title: "Stopping...",
      subtitle: "",
      titleColor: "#6c757d", // Bootstrap secondary gray
    };
  }

  if (capture.state === "failed") {
    // Capture failed
    return {
      title: "Capture Failed",
      subtitle: capture.errorMessage || "Check logs for details",
      titleColor: "#dc3545", // Bootstrap danger red
    };
  }

  // Default: stopped
  return {
    title: "Capture Stopped",
    subtitle: "Click Start to begin capturing",
    titleColor: "#6c757d", // Bootstrap secondary gray
  };
}

/**
 * Draw the status message on a canvas context.
 * Draws a muted grid background and centered status text.
 */
export function drawCaptureStatusOnCanvas(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  status: CaptureStatusMessage,
  options: {
    backgroundColor?: string;
    gridColor?: string;
    subtitleColor?: string;
  } = {}
): void {
  const {
    backgroundColor = "#f8f9fa",
    gridColor = "#e9ecef",
    subtitleColor = "#adb5bd",
  } = options;

  // Clear canvas with background
  ctx.fillStyle = backgroundColor;
  ctx.fillRect(0, 0, width, height);

  // Draw muted grid
  ctx.strokeStyle = gridColor;
  ctx.lineWidth = 1;

  // Horizontal grid lines
  for (let i = 0; i <= 4; i++) {
    const y = (i / 4) * height;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }

  // Vertical grid lines
  for (let i = 0; i <= 8; i++) {
    const x = (i / 8) * width;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }

  // Draw center line in very muted color
  const centerX = width / 2;
  ctx.strokeStyle = "#dee2e6";
  ctx.lineWidth = 1;
  ctx.setLineDash([5, 5]);
  ctx.beginPath();
  ctx.moveTo(centerX, 0);
  ctx.lineTo(centerX, height);
  ctx.stroke();
  ctx.setLineDash([]);

  // Draw status message
  ctx.font = "14px sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillStyle = status.titleColor;
  ctx.fillText(status.title, width / 2, height / 2 - (status.subtitle ? 10 : 0));

  if (status.subtitle) {
    ctx.font = "12px sans-serif";
    ctx.fillStyle = subtitleColor;
    ctx.fillText(status.subtitle, width / 2, height / 2 + 10);
  }
}
