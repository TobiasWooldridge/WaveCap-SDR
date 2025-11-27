import React from "react";
import { AlertTriangle, Droplets, RefreshCw } from "lucide-react";

type IndicatorType = "overflow" | "drops" | "retry";

interface Props {
  type: IndicatorType;
  rate?: number;
  count?: number;
  active?: boolean;
  className?: string;
}

const icons: Record<IndicatorType, React.ElementType> = {
  overflow: AlertTriangle,
  drops: Droplets,
  retry: RefreshCw,
};

const labels: Record<IndicatorType, string> = {
  overflow: "IQ Overflow",
  drops: "Audio Drops",
  retry: "Retrying",
};

export function ErrorIndicator({
  type,
  rate = 0,
  count = 0,
  active = false,
  className = "",
}: Props) {
  // Don't render if no active error
  if (!active && rate === 0 && count === 0) {
    return null;
  }

  const Icon = icons[type];
  const isWarning = rate > 0 || active;
  const label = labels[type];

  // Format the value to display
  let valueText = "";
  if (rate > 0) {
    valueText = ` ${rate.toFixed(0)}/s`;
  } else if (count > 0) {
    valueText = ` ${count}`;
  }

  return (
    <span
      className={`badge ${isWarning ? "bg-warning text-dark" : "bg-secondary"} ${className}`}
      title={`${label}: ${rate > 0 ? `${rate.toFixed(1)}/s` : count > 0 ? count : "active"}`}
      style={{ fontSize: "0.7rem" }}
    >
      <Icon
        size={12}
        className={type === "retry" && active ? "spin-animation" : ""}
        style={{ marginRight: valueText ? "0.25rem" : 0 }}
      />
      {valueText}
    </span>
  );
}
