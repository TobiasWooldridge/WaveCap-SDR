import { formatFrequencyMHz } from "../../utils/frequency";

interface FrequencyDisplayProps {
  frequencyHz: number | null | undefined;
  decimals?: number;
  name?: string | null;
}

export function FrequencyDisplay({
  frequencyHz,
  decimals = 4,
  name,
}: FrequencyDisplayProps) {
  if (frequencyHz == null) {
    return <>---</>;
  }

  const label = formatFrequencyMHz(frequencyHz, decimals);

  return (
    <>
      {label} <span className="text-muted">MHz</span>
      {name && (
        <span className="ms-1 text-muted" style={{ fontSize: "0.75rem" }}>
          {name}
        </span>
      )}
    </>
  );
}
