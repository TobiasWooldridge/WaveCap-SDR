import clsx from "clsx";

type SpinnerSize = "sm" | "md";
type SpinnerVariant =
  | "inherit"
  | "primary"
  | "secondary"
  | "success"
  | "danger"
  | "warning"
  | "info"
  | "light"
  | "dark"
  | "muted"
  | "white";

interface SpinnerProps {
  /**
   * Size of the spinner. Defaults to medium which matches Bootstrap's base size.
   */
  size?: SpinnerSize;
  /**
   * Visual variant for the spinner stroke. When set to `inherit`, no color class is applied.
   */
  variant?: SpinnerVariant;
  /**
   * Accessible label announced to screen readers while the spinner is visible.
   */
  label?: string;
  /**
   * Optional CSS class names applied to the spinner element.
   */
  className?: string;
}

const Spinner = ({
  size = "md",
  variant = "primary",
  label = "Loading…",
  className,
}: SpinnerProps) => {
  const sizeClass = size === "sm" ? "spinner-border-sm" : null;
  const variantClass =
    variant === "inherit" ? null : (`text-${variant}` as const);

  return (
    <span
      role="status"
      aria-live="polite"
      aria-busy="true"
      className={clsx("spinner-border", sizeClass, variantClass, className)}
    >
      <span className="visually-hidden">{label}</span>
    </span>
  );
};

export default Spinner;
