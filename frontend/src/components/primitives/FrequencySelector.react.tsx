import NumericSelector, { type UnitConfig } from "./NumericSelector.react";

export interface FrequencySelectorProps {
  label: string;
  value: number; // Value in Hz
  min: number;
  max: number;
  step?: number;
  info?: string;
  onChange: (value: number) => void;
  disabled?: boolean;
}

// Frequency units configuration
const frequencyUnits: UnitConfig[] = [
  {
    name: "MHz",
    multiplier: 1_000_000,
    decimals: 3,
    placeValues: [
      { label: "100", value: 100_000_000 },
      { label: "10", value: 10_000_000 },
      { label: "1", value: 1_000_000 },
      { label: "0.1", value: 100_000 },
      { label: "0.01", value: 10_000 },
      { label: "0.001", value: 1_000 },
    ],
  },
  {
    name: "kHz",
    multiplier: 1_000,
    decimals: 1,
    placeValues: [
      { label: "10000", value: 10_000_000_000 },
      { label: "1000", value: 1_000_000_000 },
      { label: "100", value: 100_000_000 },
      { label: "10", value: 10_000_000 },
      { label: "1", value: 1_000_000 },
      { label: "0.1", value: 100_000 },
    ],
  },
];

export default function FrequencySelector(props: FrequencySelectorProps) {
  return <NumericSelector {...props} units={frequencyUnits} />;
}
