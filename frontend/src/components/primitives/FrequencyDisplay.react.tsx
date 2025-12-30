import { formatFrequency, formatFrequencyMHz } from "../../utils/frequency";

type FrequencyUnit = "auto" | "MHz";

interface FrequencyDisplayProps {
  frequencyHz: number | null | undefined;
  decimals?: number;
  name?: string | null;
  unit?: FrequencyUnit;
}

export function FrequencyDisplay({
  frequencyHz,
  decimals = 4,
  name,
  unit = "auto",
}: FrequencyDisplayProps) {
  if (frequencyHz == null) {
    return <>---</>;
  }

  const label = unit === "MHz" ? formatFrequencyMHz(frequencyHz, decimals) : formatFrequency(frequencyHz);

  return (
    <>
      {label}
      {unit === "MHz" && <span className="text-muted"> MHz</span>}
      {name && (
        <span className="ms-1 text-muted" style={{ fontSize: "0.75rem" }}>
          {name}
        </span>
      )}
    </>
  );
}
