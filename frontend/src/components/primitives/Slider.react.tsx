import { forwardRef, type InputHTMLAttributes } from "react";
import clsx from "clsx";
import Flex from "./Flex.react";

export type SliderProps = {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  unit?: string;
  formatValue?: (value: number) => string;
  showMinMax?: boolean;
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
      unit = "",
      formatValue,
      showMinMax = true,
      onChange,
      className,
      disabled,
      ...rest
    },
    ref,
  ) => {
    const displayValue = formatValue ? formatValue(value) : value.toLocaleString();

    return (
      <Flex direction="column" gap={2} className={clsx("slider-container", className)}>
        <Flex justify="between" align="center">
          <label className="form-label mb-0 fw-semibold">{label}</label>
          <span className="badge bg-primary">
            {displayValue} {unit}
          </span>
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
