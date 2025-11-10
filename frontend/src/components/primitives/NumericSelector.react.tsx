import { useState } from "react";
import { ChevronUp, ChevronDown, Info } from "lucide-react";
import Flex from "./Flex.react";

export interface PlaceValue {
  label: string;
  value: number;
}

export interface UnitConfig {
  name: string;
  multiplier: number;
  decimals: number;
  placeValues: PlaceValue[];
}

export interface NumericSelectorProps {
  label: string;
  value: number; // Value in base units
  min: number;
  max: number;
  step?: number;
  info?: string;
  onChange: (value: number) => void;
  disabled?: boolean;
  units?: UnitConfig[]; // If not provided, uses value as-is with no unit conversion
}

export default function NumericSelector({
  label,
  value,
  min,
  max,
  step = 1,
  info,
  onChange,
  disabled = false,
  units,
}: NumericSelectorProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [inputValue, setInputValue] = useState("");
  const [currentUnitIndex, setCurrentUnitIndex] = useState(0);

  const currentUnit = units?.[currentUnitIndex];
  const displayValue = currentUnit ? value / currentUnit.multiplier : value;
  const unitMultiplier = currentUnit?.multiplier ?? 1;
  const unitName = currentUnit?.name ?? "";
  const decimals = currentUnit?.decimals ?? 0;
  const placeValues = currentUnit?.placeValues ?? [];

  // Format value in the selected unit
  const formatValue = (val: number): string => {
    return val.toFixed(decimals);
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
    setInputValue(formatValue(displayValue));
  };

  const handleTextBlur = () => {
    if (inputValue) {
      const parsed = parseFloat(inputValue);
      if (!isNaN(parsed)) {
        onChange(clamp(parsed * unitMultiplier));
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

  return (
    <Flex direction="column" gap={2}>
      <Flex justify="between" align="center">
        <label className="form-label mb-0 fw-semibold">
          <Flex align="center" gap={1}>
            <span>{label}</span>
            {info && (
              <span title={info} style={{ cursor: "help", display: "flex", alignItems: "center" }}>
                <Info size={14} className="text-muted" />
              </span>
            )}
          </Flex>
        </label>
        {units && units.length > 1 && (
          <select
            className="form-select form-select-sm"
            style={{ width: "auto" }}
            value={currentUnitIndex}
            onChange={(e) => setCurrentUnitIndex(parseInt(e.target.value))}
            disabled={disabled}
          >
            {units.map((unit, idx) => (
              <option key={idx} value={idx}>
                {unit.name}
              </option>
            ))}
          </select>
        )}
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

        {/* Value display and controls on the right */}
        <Flex gap={1} align="center">
          {/* Editable text display */}
          <input
            type="text"
            className="form-control form-control-sm text-end"
            style={{ width: "85px", fontFamily: "monospace", fontSize: "0.875rem" }}
            value={isEditing ? inputValue : formatValue(displayValue)}
            onChange={(e) => setInputValue(e.target.value)}
            onFocus={handleTextFocus}
            onBlur={handleTextBlur}
            onKeyDown={handleKeyDown}
            disabled={disabled}
          />
          <span className="small text-muted" style={{ width: "40px" }}>{unitName}</span>

          {/* Up/down controls for each place value */}
          {placeValues.map((place, idx) => (
            <Flex key={idx} align="center" gap={0}>
              <Flex direction="column" gap={0} style={{ width: "fit-content", minWidth: "28px" }}>
                {/* Place value label above buttons */}
                <div className="text-center text-muted" style={{ fontSize: "8px", lineHeight: "10px", marginBottom: "1px" }}>
                  {place.label}
                </div>
                <button
                  type="button"
                  className="btn btn-sm btn-outline-secondary p-0"
                  style={{ height: "16px", lineHeight: "1", fontSize: "10px", borderRadius: "2px 2px 0 0" }}
                  onClick={() => adjustByPlace(place.value, 1)}
                  disabled={disabled || value >= max}
                  title={`+${place.label} ${unitName}`}
                >
                  <ChevronUp size={12} style={{ marginTop: "-2px" }} />
                </button>
                <button
                  type="button"
                  className="btn btn-sm btn-outline-secondary p-0"
                  style={{ height: "16px", lineHeight: "1", fontSize: "10px", borderRadius: "0 0 2px 2px", borderTop: "none" }}
                  onClick={() => adjustByPlace(place.value, -1)}
                  disabled={disabled || value <= min}
                  title={`-${place.label} ${unitName}`}
                >
                  <ChevronDown size={12} style={{ marginTop: "-2px" }} />
                </button>
              </Flex>
              {/* Decimal point separators for better visual grouping */}
              {place.label.includes(".") && idx < placeValues.length - 1 && !placeValues[idx + 1].label.includes(".") && (
                <span className="text-muted" style={{ fontSize: "10px", marginLeft: "2px", marginRight: "2px" }}>
                  .
                </span>
              )}
            </Flex>
          ))}
        </Flex>
      </Flex>

      {/* Min/Max labels */}
      <Flex justify="between" className="small text-muted">
        <span>{formatValue(min / unitMultiplier)} {unitName}</span>
        <span>{formatValue(max / unitMultiplier)} {unitName}</span>
      </Flex>
    </Flex>
  );
}
