import { useEffect } from "react";
import { AlertTriangle, Droplets, RefreshCw } from "lucide-react";
import { useErrorContextOptional } from "../context/ErrorContext";

interface Props {
  captureId?: string;
}

// Track if CSS has been injected
let cssInjected = false;

export function ErrorStatusBar({ captureId }: Props) {
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
  const dropRate = stats.audio_drop?.rate ?? 0;
  const hasOverflows = overflowRate > 0;
  const hasDrops = dropRate > 0;

  // Find active retry event for this capture (if specified)
  const retryEvent = recentErrors.find(
    (e) => e.type === "device_retry" && (!captureId || e.capture_id === captureId)
  );

  // Don't render if no errors
  if (!hasOverflows && !hasDrops && !retryEvent) {
    return null;
  }

  // Extract retry details safely
  const retryAttempt = retryEvent?.details?.attempt as number | undefined;
  const retryMaxAttempts = retryEvent?.details?.max_attempts as number | undefined;

  return (
    <div
      className="alert alert-warning py-2 px-3 mb-2 d-flex align-items-center flex-wrap gap-3"
      role="alert"
      style={{ fontSize: "0.875rem" }}
    >
      {hasOverflows && (
        <span className="d-flex align-items-center gap-1">
          <AlertTriangle size={16} />
          <strong>IQ Overflow:</strong> {overflowRate.toFixed(1)}/s
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
