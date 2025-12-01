import { useEffect } from "react";
import { AlertTriangle, Droplets, RefreshCw } from "lucide-react";
import { useErrorContextOptional } from "../context/ErrorContext";
import type { Capture } from "../types";

interface Props {
  captureId?: string;
  capture?: Capture;  // Optional capture for enhanced info (sample rate, etc.)
}

// Track if CSS has been injected
let cssInjected = false;

// Estimate samples lost per overflow event
// Ring buffer is ~1M samples, reset position is half buffer behind writer
const SAMPLES_LOST_PER_OVERFLOW = 524288;

export function ErrorStatusBar({ captureId, capture }: Props) {
  const errorCtx = useErrorContextOptional();

  // Inject CSS once on first render
  useEffect(() => {
    if (cssInjected) return;
    cssInjected = true;

    const styleSheet = document.createElement("style");
    styleSheet.textContent = `
      .spin-animation {
        animation: spin 1s linear infinite;
      }
      @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
      }
    `;
    document.head.appendChild(styleSheet);
  }, []);

  // Don't render if not within ErrorProvider
  if (!errorCtx) {
    return null;
  }

  const { stats, recentErrors, isConnected } = errorCtx;

  // Don't show anything if not connected yet
  if (!isConnected) {
    return null;
  }

  const overflowRate = stats.iq_overflow?.rate ?? 0;
  const overflowTotal = stats.iq_overflow?.total ?? 0;
  const dropRate = stats.audio_drop?.rate ?? 0;
  const hasOverflows = overflowRate > 0;
  const hasDrops = dropRate > 0;

  // Calculate estimated data loss percentage
  // Loss % = (overflows/sec * samples_lost_per_overflow) / sample_rate * 100
  let lossPercent: number | null = null;
  if (overflowRate > 0 && capture?.sampleRate) {
    const samplesLostPerSecond = overflowRate * SAMPLES_LOST_PER_OVERFLOW;
    lossPercent = (samplesLostPerSecond / capture.sampleRate) * 100;
  }

  // Find active retry event for this capture (if specified)
  const retryEvent = recentErrors.find(
    (e) => e.type === "device_retry" && (!captureId || e.capture_id === captureId)
  );

  // Extract retry details safely
  const retryAttempt = retryEvent?.details?.attempt as number | undefined;
  const retryMaxAttempts = retryEvent?.details?.max_attempts as number | undefined;

  const hasErrors = hasOverflows || hasDrops || retryEvent;

  // Always render a container to reserve space and prevent reflow
  return (
    <div
      className={`alert py-2 px-3 mb-2 d-flex align-items-center flex-wrap gap-3 ${hasErrors ? "alert-warning" : ""}`}
      role="alert"
      style={{
        fontSize: "0.875rem",
        minHeight: "38px", // Reserve consistent height
        visibility: hasErrors ? "visible" : "hidden",
        // Keep the element in the layout but invisible when no errors
        opacity: hasErrors ? 1 : 0,
        transition: "opacity 0.15s ease-in-out",
      }}
    >
      {hasOverflows && (
        <span className="d-flex align-items-center gap-1">
          <AlertTriangle size={16} />
          <strong>IQ Overflow:</strong>{" "}
          {overflowRate.toFixed(1)}/s
          {lossPercent !== null && (
            <span className="text-danger fw-bold">
              ({lossPercent < 1 ? "<1" : lossPercent.toFixed(0)}% loss)
            </span>
          )}
          {overflowTotal > 0 && (
            <span className="text-muted">
              [{overflowTotal} total]
            </span>
          )}
        </span>
      )}
      {hasDrops && (
        <span className="d-flex align-items-center gap-1">
          <Droplets size={16} />
          <strong>Audio Drops:</strong> {dropRate.toFixed(1)}/s
        </span>
      )}
      {retryEvent && (
        <span className="d-flex align-items-center gap-1">
          <RefreshCw size={16} className="spin-animation" />
          <strong>Retrying:</strong>{" "}
          {retryAttempt ?? "?"}/{retryMaxAttempts ?? "?"}
        </span>
      )}
    </div>
  );
}
