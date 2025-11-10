import { useState } from "react";
import { ChevronUp, ChevronDown } from "lucide-react";
import Flex from "./Flex.react";
import InfoTooltip from "./InfoTooltip.react";

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

export default function FrequencySelector({
  label,
  value,
  min,
  max,
  step = 1000,
  info,
  onChange,
  disabled = false,
}: FrequencySelectorProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [inputValue, setInputValue] = useState("");

  // Convert Hz to MHz for display
  const mhz = value / 1_000_000;

  // Split frequency into digit groups: hundreds, tens, ones, tenths, hundredths, thousandths
  const formatFrequency = (freq: number): string => {
    return freq.toFixed(3);
  };

  const clamp = (val: number) => Math.max(min, Math.min(max, val));

  // Increment/decrement by specific place value
  const adjustByPlace = (placeValue: number, direction: 1 | -1) => {
    const delta = placeValue * direction;
    onChange(clamp(value + delta));
  };

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onChange(parseFloat(e.target.value));
  };

  const handleTextFocus = () => {
    setIsEditing(true);
    setInputValue((value / 1_000_000).toFixed(3));
  };

  const handleTextBlur = () => {
    if (inputValue) {
      const parsed = parseFloat(inputValue);
      if (!isNaN(parsed)) {
        onChange(clamp(parsed * 1_000_000));
      }
    }
    setInputValue("");
    setIsEditing(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      (e.target as HTMLInputElement).blur();
    } else if (e.key === "Escape") {
      setInputValue("");
      setIsEditing(false);
      (e.target as HTMLInputElement).blur();
    }
  };

  // Place values in Hz: 100 MHz, 10 MHz, 1 MHz, 0.1 MHz, 0.01 MHz, 0.001 MHz
  const placeValues = [
    { label: "100", value: 100_000_000 },
    { label: "10", value: 10_000_000 },
    { label: "1", value: 1_000_000 },
    { label: "0.1", value: 100_000 },
    { label: "0.01", value: 10_000 },
    { label: "0.001", value: 1_000 },
  ];

  return (
    <Flex direction="column" gap={2}>
      <Flex justify="between" align="center">
        <label className="form-label mb-0 fw-semibold">
          {label}
          {info && <InfoTooltip content={info} />}
        </label>
      </Flex>

      <Flex gap={3} align="center">
        {/* Slider on the left */}
        <div style={{ flex: 1 }}>
          <input
            type="range"
            className="form-range"
            min={min}
            max={max}
            step={step}
            value={value}
            disabled={disabled}
            onChange={handleSliderChange}
          />
        </div>

        {/* Frequency display and controls on the right */}
        <Flex gap={1} align="center">
          {/* Editable text display */}
          <input
            type="text"
            className="form-control form-control-sm text-end"
            style={{ width: "85px", fontFamily: "monospace", fontSize: "0.875rem" }}
            value={isEditing ? inputValue : formatFrequency(mhz)}
            onChange={(e) => setInputValue(e.target.value)}
            onFocus={handleTextFocus}
            onBlur={handleTextBlur}
            onKeyDown={handleKeyDown}
            disabled={disabled}
          />
          <span className="small text-muted" style={{ width: "30px" }}>MHz</span>

          {/* Up/down controls for each decimal place */}
          {placeValues.map((place, idx) => (
            <Flex key={idx} direction="column" gap={0} style={{ width: "28px" }}>
              <button
                type="button"
                className="btn btn-sm btn-outline-secondary p-0"
                style={{ height: "16px", lineHeight: "1", fontSize: "10px", borderRadius: "2px 2px 0 0" }}
                onClick={() => adjustByPlace(place.value, 1)}
                disabled={disabled || value >= max}
                title={`+${place.label} MHz`}
              >
                <ChevronUp size={12} style={{ marginTop: "-2px" }} />
              </button>
              <button
                type="button"
                className="btn btn-sm btn-outline-secondary p-0"
                style={{ height: "16px", lineHeight: "1", fontSize: "10px", borderRadius: "0 0 2px 2px", borderTop: "none" }}
                onClick={() => adjustByPlace(place.value, -1)}
                disabled={disabled || value <= min}
                title={`-${place.label} MHz`}
              >
                <ChevronDown size={12} style={{ marginTop: "-2px" }} />
              </button>
            </Flex>
          ))}
        </Flex>
      </Flex>

      {/* Min/Max labels */}
      <Flex justify="between" className="small text-muted">
        <span>{formatFrequency(min / 1_000_000)} MHz</span>
        <span>{formatFrequency(max / 1_000_000)} MHz</span>
      </Flex>
    </Flex>
  );
}
