import Flex from "./Flex.react";

export interface SignalMeterProps {
  /** Signal power in dB (typically -60 to 0 dB range) */
  signalPowerDb: number | null;
  /** Width of the meter in pixels (default: 100) */
  width?: number;
  /** Height of the meter in pixels (default: 16) */
  height?: number;
}

export default function SignalMeter({
  signalPowerDb,
  width = 100,
  height = 16,
}: SignalMeterProps) {
  // Signal power typically ranges from -60 dB (very weak) to 0 dB (max)
  // Map this to a 0-100% scale
  const minDb = -60;
  const maxDb = 0;

  let percentage = 0;
  let color = "#6c757d"; // Gray for no signal
  let label = "N/A";

  if (signalPowerDb != null) {
    // Clamp and normalize to 0-100%
    const clampedDb = Math.max(minDb, Math.min(maxDb, signalPowerDb));
    percentage = ((clampedDb - minDb) / (maxDb - minDb)) * 100;

    // Color coding based on signal strength
    if (signalPowerDb > -20) {
      color = "#28a745"; // Strong - Green
    } else if (signalPowerDb > -40) {
      color = "#ffc107"; // Medium - Yellow
    } else {
      color = "#dc3545"; // Weak - Red
    }

    label = `${signalPowerDb.toFixed(1)} dB`;
  }

  return (
    <Flex direction="row" gap={2} align="center" style={{ width: `${width}px` }}>
      <div
        style={{
          position: "relative",
          flex: 1,
          height: `${height}px`,
          backgroundColor: "#e9ecef",
          borderRadius: "4px",
          overflow: "hidden",
          border: "2px solid #dee2e6",
          boxShadow: "inset 0 1px 2px rgba(0,0,0,0.1)",
        }}
      >
        <div
          style={{
            position: "absolute",
            left: 0,
            top: 0,
            height: "100%",
            width: `${percentage}%`,
            backgroundColor: color,
            transition: "width 0.2s ease-out, background-color 0.2s ease-out",
            boxShadow: percentage > 0 ? "0 0 4px rgba(0,0,0,0.2)" : "none",
          }}
        />
      </div>
      <span
        className="fw-semibold"
        style={{
          fontSize: "11px",
          minWidth: "50px",
          textAlign: "right",
          color: signalPowerDb != null ? color : "#6c757d"
        }}
      >
        {label}
      </span>
    </Flex>
  );
}
