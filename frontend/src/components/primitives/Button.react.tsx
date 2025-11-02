import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from "react";
import clsx from "clsx";

export type ButtonUse =
  | "default"
  | "primary"
  | "secondary"
  | "success"
  | "create"
  | "danger"
  | "destroy"
  | "warning"
  | "light"
  | "link"
  | "close"
  | "unstyled";

export type ButtonAppearance = "filled" | "outline";

export type ButtonSize = "sm" | "md" | "lg";

type ButtonStyle = {
  filled?: string;
  outline?: string;
  includeBaseClass?: boolean;
};

const BUTTON_STYLES: Record<ButtonUse, ButtonStyle> = {
  default: { filled: "btn-secondary", outline: "btn-outline-secondary" },
  primary: { filled: "btn-primary", outline: "btn-outline-primary" },
  secondary: { filled: "btn-secondary", outline: "btn-outline-secondary" },
  success: { filled: "btn-success", outline: "btn-outline-success" },
  create: { filled: "btn-success", outline: "btn-outline-success" },
  danger: { filled: "btn-danger", outline: "btn-outline-danger" },
  destroy: { filled: "btn-danger", outline: "btn-outline-danger" },
  warning: { filled: "btn-warning", outline: "btn-outline-warning" },
  light: { filled: "btn-light", outline: "btn-outline-light" },
  link: { filled: "btn-link" },
  close: { filled: "btn-close", includeBaseClass: false },
  unstyled: { filled: "", includeBaseClass: false },
};

export type ButtonProps = {
  use?: ButtonUse;
  appearance?: ButtonAppearance;
  size?: ButtonSize;
  startContent?: ReactNode;
  endContent?: ReactNode;
  tooltip?: string;
  isContentInline?: boolean;
  isCondensed?: boolean;
} & ButtonHTMLAttributes<HTMLButtonElement>;

const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      use = "default",
      appearance = "filled",
      size = "md",
      startContent,
      endContent,
      tooltip,
      className,
      children,
      type = "button",
      isContentInline,
      isCondensed = false,
      title: htmlTitle,
      ...rest
    },
    ref,
  ) => {
    const { ["aria-label"]: ariaLabelProp, ...buttonRest } = rest;
    const style = BUTTON_STYLES[use] ?? BUTTON_STYLES.default;
    const includeBaseClass = style.includeBaseClass ?? true;
    const variantClass =
      appearance === "outline"
        ? style.outline ?? style.filled ?? ""
        : style.filled ?? style.outline ?? "";
    const shouldApplyContentLayout =
      isContentInline ?? Boolean(startContent || endContent);

    const sizeClass =
      includeBaseClass && size !== "md" ? `btn-${size}` : undefined;

    const labelText =
      typeof children === "string"
        ? children
        : typeof tooltip === "string"
          ? tooltip
          : typeof htmlTitle === "string"
            ? htmlTitle
            : undefined;

    const resolvedTitle = tooltip ?? htmlTitle ?? (isCondensed ? labelText : undefined);
    const resolvedAriaLabel = ariaLabelProp ?? (isCondensed ? labelText : undefined);

    return (
      <button
        {...buttonRest}
        ref={ref}
        type={type}
        title={resolvedTitle}
        aria-label={resolvedAriaLabel}
        className={clsx(
          includeBaseClass && "btn",
          includeBaseClass && sizeClass,
          variantClass,
          includeBaseClass && shouldApplyContentLayout &&
            "d-inline-flex align-items-center gap-2",
          isCondensed && "btn-condensed",
          className,
        )}
      >
        {startContent}
        {isCondensed
          ? children != null && (
              <span className="visually-hidden">{children}</span>
            )
          : children}
        {endContent}
      </button>
    );
  },
);

Button.displayName = "Button";

export default Button;
