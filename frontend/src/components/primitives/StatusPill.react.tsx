export type StatusVariant =
  | "success"   // Green - active, running, synced
  | "warning"   // Yellow - starting, syncing, searching
  | "danger"    // Red - failed, error, lost
  | "secondary" // Gray - off, stopped, idle
  | "info"      // Blue/Cyan - informational, locked
  | "primary";  // Primary brand color

interface StatusPillProps {
  /** The text to display in the pill */
  label: string;
  /** The visual variant/color */
  variant: StatusVariant;
  /** Optional count to display after label (e.g., active calls) */
  count?: number;
  /** Whether to show pulsing animation (for active states) */
  pulsing?: boolean;
  /** Optional additional CSS classes */
  className?: string;
  /** Size of the pill */
  size?: "sm" | "md";
}

// Use inline styles to ensure colors work regardless of Bootstrap loading
const variantStyles: Record<StatusVariant, React.CSSProperties> = {
  success: { backgroundColor: "#198754", color: "#fff" },     // Bootstrap green
  warning: { backgroundColor: "#ffc107", color: "#000" },     // Bootstrap yellow
  danger: { backgroundColor: "#dc3545", color: "#fff" },      // Bootstrap red
  secondary: { backgroundColor: "#6c757d", color: "#fff" },   // Bootstrap gray
  info: { backgroundColor: "#0dcaf0", color: "#000" },        // Bootstrap cyan
  primary: { backgroundColor: "#0d6efd", color: "#fff" },     // Bootstrap blue
};

// CSS for pulsing animation (injected once)
const pulseKeyframes = `
@keyframes status-pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.6; }
}
`;

// Inject keyframes into document head if not already present
if (typeof document !== "undefined") {
  const styleId = "status-pill-animations";
  if (!document.getElementById(styleId)) {
    const style = document.createElement("style");
    style.id = styleId;
    style.textContent = pulseKeyframes;
    document.head.appendChild(style);
  }
}

/**
 * A small pill/badge component for displaying status.
 * Use this for consistent status indicators across the app.
 */
export function StatusPill({
  label,
  variant,
  count,
  pulsing = false,
  className = "",
  size = "sm",
}: StatusPillProps) {
  const sizeStyles: React.CSSProperties = size === "sm"
    ? { fontSize: "0.6rem" }
    : { fontSize: "0.75rem" };

  const animationStyles: React.CSSProperties = pulsing
    ? { animation: "status-pulse 1.5s ease-in-out infinite" }
    : {};

  // Format display text with optional count
  const displayText = count !== undefined && count > 0
    ? `${label} (${count})`
    : label;

  return (
    <span
      className={`badge ${className}`}
      style={{
        ...variantStyles[variant],
        ...sizeStyles,
        ...animationStyles,
      }}
    >
      {displayText}
    </span>
  );
}

// ============================================================================
// Helper functions for mapping states to variants
// ============================================================================

export interface StatusPillConfig {
  label: string;
  variant: StatusVariant;
  pulsing?: boolean;
}

/**
 * Map capture states to status pill props.
 * Uses more expressive labels than before.
 */
export function getCaptureStatusProps(state: string): StatusPillConfig {
  switch (state) {
    case "running":
      return { label: "Live", variant: "success" };
    case "stopped":
      return { label: "Off", variant: "secondary" };
    case "starting":
      return { label: "Starting", variant: "warning" };
    case "stopping":
      return { label: "Stopping", variant: "warning" };
    case "failed":
    case "error":
      return { label: "Failed", variant: "danger" };
    default:
      // Capitalize first letter for unknown states
      return { label: state.charAt(0).toUpperCase() + state.slice(1), variant: "secondary" };
  }
}

/**
 * Map trunking system states to status pill props.
 * Includes control channel state for more expressive status.
 *
 * @param systemState - The trunking system state (running, stopped, etc.)
 * @param ccState - The control channel state (locked, searching, lost)
 * @param activeCalls - Number of active calls (optional, shown in label)
 * @param isManuallyLocked - Whether manually locked to a frequency
 */
export function getTrunkingStatusProps(
  systemState: string,
  ccState?: string,
  activeCalls?: number,
  isManuallyLocked?: boolean
): StatusPillConfig & { count?: number } {
  // Handle non-running states first
  switch (systemState) {
    case "stopped":
      return { label: "Off", variant: "secondary" };
    case "starting":
      return { label: "Starting", variant: "warning" };
    case "failed":
      return { label: "Failed", variant: "danger" };
  }

  // For running systems, check control channel state
  if (systemState === "running" || systemState === "synced") {
    // Check if manually locked
    if (isManuallyLocked) {
      return {
        label: "Locked",
        variant: "info",
        count: activeCalls,
        pulsing: activeCalls !== undefined && activeCalls > 0,
      };
    }

    // Check control channel state
    switch (ccState) {
      case "locked":
        return {
          label: "Synced",
          variant: "success",
          count: activeCalls,
          pulsing: activeCalls !== undefined && activeCalls > 0,
        };
      case "searching":
        return { label: "Hunting", variant: "warning" };
      case "lost":
        return { label: "Lost", variant: "danger" };
      default:
        // Default for running without CC state
        return {
          label: "Synced",
          variant: "success",
          count: activeCalls,
          pulsing: activeCalls !== undefined && activeCalls > 0,
        };
    }
  }

  // Fallback
  return { label: systemState.charAt(0).toUpperCase() + systemState.slice(1), variant: "secondary" };
}

/**
 * Simple trunking status without CC state (for backwards compatibility)
 */
export function getTrunkingStatusPropsSimple(state: string): StatusPillConfig {
  switch (state) {
    case "running":
    case "synced":
      return { label: "Synced", variant: "success" };
    case "stopped":
      return { label: "Off", variant: "secondary" };
    case "starting":
      return { label: "Starting", variant: "warning" };
    case "searching":
      return { label: "Hunting", variant: "warning" };
    case "synced":
      return { label: "Synced", variant: "success" };
    case "failed":
      return { label: "Failed", variant: "danger" };
    default:
      return { label: state.charAt(0).toUpperCase() + state.slice(1), variant: "secondary" };
  }
}

/**
 * Get status props for a radio tab (capture or trunking)
 */
export function getRadioTabStatusProps(
  state: string,
  type: "capture" | "trunking"
): StatusPillConfig {
  return type === "trunking"
    ? getTrunkingStatusPropsSimple(state)
    : getCaptureStatusProps(state);
}
