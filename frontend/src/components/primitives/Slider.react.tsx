import { forwardRef, useState, type InputHTMLAttributes } from "react";
import clsx from "clsx";
import { ChevronUp, ChevronDown, ChevronsUp, ChevronsDown, Info } from "lucide-react";
import Flex from "./Flex.react";
import Button from "./Button.react";

export type SliderProps = {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  coarseStep?: number;
  unit?: string;
  formatValue?: (value: number) => string;
  parseValue?: (text: string) => number | null;
  showMinMax?: boolean;
  info?: string;
  onChange: (value: number) => void;
} & Omit<InputHTMLAttributes<HTMLInputElement>, "type" | "onChange" | "value" | "min" | "max" | "step">;

const Slider = forwardRef<HTMLInputElement, SliderProps>(
  (
    {
      label,
      value,
      min,
      max,
      step = 1,
      coarseStep,
      unit = "",
      formatValue,
      parseValue,
      showMinMax = true,
      info,
      onChange,
      className,
      disabled,
      ...rest
    },
    ref,
  ) => {
    const [inputValue, setInputValue] = useState("");
    const [isEditing, setIsEditing] = useState(false);

    const displayValue = formatValue ? formatValue(value) : value.toLocaleString();
    const actualCoarseStep = coarseStep ?? step * 10;

    const clamp = (val: number) => Math.max(min, Math.min(max, val));

    const handleIncrement = (amount: number) => {
      onChange(clamp(value + amount));
    };

    const handleTextChange = (text: string) => {
      setInputValue(text);
    };

    const handleTextBlur = () => {
      if (inputValue) {
        let newValue: number | null = null;

        if (parseValue) {
          newValue = parseValue(inputValue);
        } else {
          const parsed = parseFloat(inputValue.replace(/,/g, ''));
          if (!isNaN(parsed)) {
            newValue = parsed;
          }
        }

        if (newValue !== null) {
          onChange(clamp(newValue));
        }
      }
      setInputValue("");
      setIsEditing(false);
    };

    const handleTextFocus = () => {
      setIsEditing(true);
      setInputValue(value.toString());
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
      if (e.key === "Enter") {
        (e.target as HTMLInputElement).blur();
      } else if (e.key === "Escape") {
        setInputValue("");
        setIsEditing(false);
        (e.target as HTMLInputElement).blur();
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        handleIncrement(e.shiftKey ? actualCoarseStep : step);
      } else if (e.key === "ArrowDown") {
        e.preventDefault();
        handleIncrement(e.shiftKey ? -actualCoarseStep : -step);
      }
    };

    return (
      <Flex direction="column" gap={2} className={clsx("slider-container", className)}>
        <Flex justify="between" align="center" style={{ flexWrap: "wrap", gap: "0.5rem" }}>
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

          <Flex align="center" gap={1} style={{ flexWrap: "wrap" }}>
            {/* Decrement buttons - kept together */}
            <Flex align="center" gap={1} style={{ flexShrink: 0 }}>
              <Button
                use="secondary"
                size="sm"
                appearance="outline"
                onClick={() => handleIncrement(-actualCoarseStep)}
                disabled={disabled || value <= min}
                title={`Decrease by ${actualCoarseStep} ${unit}`}
                className="px-1 py-0"
              >
                <ChevronsDown size={14} />
              </Button>
              <Button
                use="secondary"
                size="sm"
                appearance="outline"
                onClick={() => handleIncrement(-step)}
                disabled={disabled || value <= min}
                title={`Decrease by ${step} ${unit}`}
                className="px-1 py-0"
              >
                <ChevronDown size={14} />
              </Button>
            </Flex>

            {/* Text input with unit - kept together */}
            <Flex align="center" gap={1} style={{ flexShrink: 0 }}>
              <input
                type="text"
                className="form-control form-control-sm text-center"
                style={{ width: "110px", fontFamily: "monospace", fontSize: "0.875rem" }}
                value={isEditing ? inputValue : displayValue}
                onChange={(e) => handleTextChange(e.target.value)}
                onFocus={handleTextFocus}
                onBlur={handleTextBlur}
                onKeyDown={handleKeyDown}
                disabled={disabled}
                placeholder={displayValue}
              />
              <span className="small text-muted">{unit}</span>
            </Flex>

            {/* Increment buttons - kept together */}
            <Flex align="center" gap={1} style={{ flexShrink: 0 }}>
              <Button
                use="secondary"
                size="sm"
                appearance="outline"
                onClick={() => handleIncrement(step)}
                disabled={disabled || value >= max}
                title={`Increase by ${step} ${unit}`}
                className="px-1 py-0"
              >
                <ChevronUp size={14} />
              </Button>
              <Button
                use="secondary"
                size="sm"
                appearance="outline"
                onClick={() => handleIncrement(actualCoarseStep)}
                disabled={disabled || value >= max}
                title={`Increase by ${actualCoarseStep} ${unit}`}
                className="px-1 py-0"
              >
                <ChevronsUp size={14} />
              </Button>
            </Flex>
          </Flex>
        </Flex>

        <input
          {...rest}
          ref={ref}
          type="range"
          className="form-range"
          min={min}
          max={max}
          step={step}
          value={value}
          disabled={disabled}
          onChange={(e) => onChange(parseFloat(e.target.value))}
        />

        {showMinMax && (
          <Flex justify="between" className="small text-muted">
            <span>
              {formatValue ? formatValue(min) : min.toLocaleString()} {unit}
            </span>
            <span>
              {formatValue ? formatValue(max) : max.toLocaleString()} {unit}
            </span>
          </Flex>
        )}
      </Flex>
    );
  },
);

Slider.displayName = "Slider";

export default Slider;
