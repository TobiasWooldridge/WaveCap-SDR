import { memo, useMemo } from "react";
import Flex from "./Flex.react";

export interface SMeterProps {
  /** RSSI in dBFS (typically -60 to 0 dBFS range) */
  rssiDbFs: number | null;
  /** Frequency in Hz (used to determine HF vs VHF/UHF for proper S-unit conversion) */
  frequencyHz?: number;
  /** Width of the meter in pixels (default: 200) */
  width?: number;
  /** Height of the meter in pixels (default: 24) */
  height?: number;
  /** Show peak hold indicator */
  showPeakHold?: boolean;
}

/**
 * Convert RSSI in dBFS to S-units.
 *
 * Since we're working with dBFS (relative to full scale) rather than dBm,
 * we use a relative mapping:
 * - S1 corresponds to approximately -54 dBFS
 * - S9 corresponds to approximately -6 dBFS
 * - Each S-unit is 6 dB
 * - Above S9, we measure in dB over S9
 *
 * Note: For absolute S-meter readings in dBm, device-specific calibration
 * would be required. This implementation provides relative signal strength
 * that's consistent and useful for SDR operation.
 */
function dbfsToSUnits(dbfs: number): { sValue: number; overS9Db: number; displayText: string } {
  // S-unit scale mapping (each S-unit is 6 dB)
  // S1 = -54 dBFS, S9 = -6 dBFS
  const S9_DBFS = -6;
  const S1_DBFS = -54;

  if (dbfs >= S9_DBFS) {
    // Above S9, measure in dB over S9
    const overS9 = dbfs - S9_DBFS;
    return {
      sValue: 9,
      overS9Db: overS9,
      displayText: overS9 > 0 ? `S9+${overS9.toFixed(0)}` : "S9"
    };
  } else {
    // Below S9, calculate S-unit (S1 to S9)
    const sValue = Math.max(1, Math.min(9, Math.floor((dbfs - S1_DBFS) / 6) + 1));
    return {
      sValue,
      overS9Db: 0,
      displayText: `S${sValue}`
    };
  }
}

/**
 * Get color based on signal strength
 */
function getSMeterColor(sValue: number, overS9Db: number): string {
  if (sValue === 9 && overS9Db > 20) {
    return "#dc3545"; // Red for very strong (S9+20 or more)
  } else if (sValue === 9 || sValue >= 7) {
    return "#28a745"; // Green for strong (S7-S9)
  } else if (sValue >= 5) {
    return "#ffc107"; // Yellow for medium (S5-S6)
  } else {
    return "#fd7e14"; // Orange for weak (S1-S4)
  }
}

// Pre-generate static tick marks (never changes)
const TICK_MARKS = Array.from({ length: 9 }, (_, i) => {
  const s = i + 1;
  const tickPosition = ((s - 1) / 8) * 100;
  return (
    <div
      key={s}
      style={{
        position: "absolute",
        left: `${tickPosition}%`,
        top: 0,
        bottom: 0,
        width: "1px",
        backgroundColor: s % 2 === 1 ? "#adb5bd" : "#dee2e6",
        opacity: 0.6,
      }}
    />
  );
});

// Pre-generate S-unit labels (never changes)
const S_UNIT_LABELS = [1, 3, 5, 7, 9].map((s) => (
  <span
    key={s}
    style={{
      fontSize: "8px",
      color: "#adb5bd",
      fontWeight: 600,
      textShadow: "0 1px 2px rgba(0,0,0,0.8)",
    }}
  >
    {s}
  </span>
));

function SMeterComponent({
  rssiDbFs,
  frequencyHz: _frequencyHz,
  width = 200,
  height = 24,
  showPeakHold: _showPeakHold = true,
}: SMeterProps) {
  // Memoize size-dependent styles
  const containerStyle = useMemo(() => ({ width: `${width}px` }), [width]);
  const meterStyle = useMemo(() => ({
    position: "relative" as const,
    flex: 1,
    height: `${height}px`,
    backgroundColor: "#212529",
    borderRadius: "4px",
    overflow: "hidden",
    border: "2px solid #495057",
    boxShadow: "inset 0 2px 4px rgba(0,0,0,0.3)",
  }), [height]);

  if (rssiDbFs == null) {
    return (
      <Flex direction="row" gap={2} align="center" style={containerStyle}>
        <div
          style={{
            flex: 1,
            height: `${height}px`,
            backgroundColor: "#e9ecef",
            borderRadius: "4px",
            border: "2px solid #dee2e6",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <span style={{ fontSize: "10px", color: "#6c757d", fontWeight: 600 }}>
            NO SIGNAL
          </span>
        </div>
        <span
          style={{
            fontSize: "11px",
            fontWeight: 700,
            minWidth: "50px",
            textAlign: "right",
            color: "#6c757d",
          }}
        >
          S0
        </span>
      </Flex>
    );
  }

  const { sValue, overS9Db, displayText } = dbfsToSUnits(rssiDbFs);
  const color = getSMeterColor(sValue, overS9Db);

  // Calculate percentage for the meter fill
  // S1-S9 covers 0-100% of the main scale
  // Above S9, we extend the scale
  let percentage: number;
  if (sValue < 9) {
    // S1 to S9: map to 0-100%
    percentage = ((sValue - 1) / 8) * 100;
  } else {
    // Above S9: full bar + extend based on overS9Db
    // Map 0-60dB over S9 to 100-150% (overflowing visually)
    percentage = 100 + Math.min(overS9Db / 60 * 50, 50);
  }

  return (
    <Flex direction="row" gap={2} align="center" style={containerStyle}>
      <div style={meterStyle}>
        {/* Tick marks (pre-generated) */}
        {TICK_MARKS}

        {/* Background gradient zones */}
        <div
          style={{
            position: "absolute",
            left: 0,
            top: 0,
            height: "100%",
            width: "50%",
            background: "linear-gradient(to right, rgba(253, 126, 20, 0.15), rgba(255, 193, 7, 0.15))",
          }}
        />
        <div
          style={{
            position: "absolute",
            left: "50%",
            top: 0,
            height: "100%",
            width: "37.5%",
            background: "linear-gradient(to right, rgba(255, 193, 7, 0.15), rgba(40, 167, 69, 0.15))",
          }}
        />
        <div
          style={{
            position: "absolute",
            left: "87.5%",
            top: 0,
            height: "100%",
            width: "12.5%",
            background: "rgba(40, 167, 69, 0.2)",
          }}
        />

        {/* S-unit labels (pre-generated) */}
        <div
          style={{
            position: "absolute",
            left: 0,
            right: 0,
            top: "2px",
            display: "flex",
            justifyContent: "space-between",
            padding: "0 4px",
            pointerEvents: "none",
          }}
        >
          {S_UNIT_LABELS}
        </div>

        {/* Signal fill bar */}
        <div
          style={{
            position: "absolute",
            left: 0,
            top: 0,
            height: "100%",
            width: `${Math.min(percentage, 100)}%`,
            backgroundColor: color,
            transition: "width 0.15s ease-out, background-color 0.15s ease-out",
            boxShadow: percentage > 0 ? `0 0 8px ${color}` : "none",
            borderRight: percentage > 0 && percentage < 100 ? `2px solid ${color}` : "none",
          }}
        />

        {/* Over S9 extension bar (if applicable) */}
        {percentage > 100 && (
          <div
            style={{
              position: "absolute",
              right: 0,
              top: 0,
              height: "100%",
              width: `${percentage - 100}%`,
              backgroundColor: "#dc3545",
              boxShadow: "0 0 10px #dc3545",
              opacity: 0.8,
            }}
          />
        )}
      </div>

      {/* S-unit display */}
      <span
        style={{
          fontSize: "12px",
          fontWeight: 700,
          minWidth: "55px",
          textAlign: "right",
          color: color,
          textShadow: "0 1px 2px rgba(0,0,0,0.3)",
          fontFamily: "monospace",
        }}
      >
        {displayText}
      </span>
    </Flex>
  );
}

// Memoize to prevent re-renders when parent updates but props haven't changed
const SMeter = memo(SMeterComponent);
export default SMeter;
