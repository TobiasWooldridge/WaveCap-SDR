import { useState, useRef, useEffect, type ReactNode } from "react";
import clsx from "clsx";
import { ChevronDown } from "lucide-react";
import type { ButtonUse, ButtonSize } from "./Button.react";

export interface DropdownMenuItem {
  id: string;
  label: ReactNode;
  icon?: ReactNode;
  onClick: () => void;
  disabled?: boolean;
  use?: ButtonUse;
  hidden?: boolean;
  divider?: boolean;
  requireConfirm?: boolean;
  confirmLabel?: ReactNode;
}

interface SplitButtonDropdownProps {
  /** Main button content */
  mainLabel: ReactNode;
  /** Main button click handler */
  onMainClick: () => void;
  /** Main button disabled state */
  mainDisabled?: boolean;
  /** Menu items for dropdown */
  menuItems: DropdownMenuItem[];
  /** Button use/variant */
  use?: ButtonUse;
  /** Button size */
  size?: ButtonSize;
  /** Additional class for the container */
  className?: string;
  /** Whether any action is pending */
  isPending?: boolean;
  /** Pending spinner content */
  pendingContent?: ReactNode;
}

const BUTTON_STYLES: Record<ButtonUse, { filled: string }> = {
  default: { filled: "btn-secondary" },
  primary: { filled: "btn-primary" },
  secondary: { filled: "btn-secondary" },
  success: { filled: "btn-success" },
  create: { filled: "btn-success" },
  danger: { filled: "btn-danger" },
  destroy: { filled: "btn-danger" },
  warning: { filled: "btn-warning" },
  light: { filled: "btn-light" },
  link: { filled: "btn-link" },
  close: { filled: "btn-close" },
  unstyled: { filled: "" },
};

export default function SplitButtonDropdown({
  mainLabel,
  onMainClick,
  mainDisabled = false,
  menuItems,
  use = "secondary",
  size = "sm",
  className,
  isPending = false,
  pendingContent,
}: SplitButtonDropdownProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [confirmingId, setConfirmingId] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setIsOpen(false);
        setConfirmingId(null);
      }
    };

    if (isOpen) {
      document.addEventListener("mousedown", handleClickOutside);
      return () => document.removeEventListener("mousedown", handleClickOutside);
    }
  }, [isOpen]);

  // Reset confirm state after timeout
  useEffect(() => {
    if (confirmingId) {
      const timer = setTimeout(() => setConfirmingId(null), 3000);
      return () => clearTimeout(timer);
    }
  }, [confirmingId]);

  const handleMenuItemClick = (item: DropdownMenuItem) => {
    if (item.disabled) return;

    if (item.requireConfirm) {
      if (confirmingId === item.id) {
        // Second click - execute
        item.onClick();
        setConfirmingId(null);
        setIsOpen(false);
      } else {
        // First click - enter confirm mode
        setConfirmingId(item.id);
      }
    } else {
      item.onClick();
      setIsOpen(false);
    }
  };

  const btnClass = BUTTON_STYLES[use]?.filled ?? BUTTON_STYLES.default.filled;
  const sizeClass = size !== "md" ? `btn-${size}` : "";

  const visibleItems = menuItems.filter((item) => !item.hidden);

  return (
    <div ref={containerRef} className={clsx("btn-group", className)}>
      {/* Main button */}
      <button
        type="button"
        className={clsx("btn", btnClass, sizeClass)}
        onClick={onMainClick}
        disabled={mainDisabled || isPending}
      >
        {isPending && pendingContent ? pendingContent : mainLabel}
      </button>

      {/* Dropdown toggle */}
      <button
        type="button"
        className={clsx("btn dropdown-toggle dropdown-toggle-split", btnClass, sizeClass)}
        onClick={() => setIsOpen(!isOpen)}
        disabled={isPending}
        aria-expanded={isOpen}
      >
        <ChevronDown size={12} />
        <span className="visually-hidden">Toggle Dropdown</span>
      </button>

      {/* Dropdown menu */}
      {isOpen && visibleItems.length > 0 && (
        <ul
          className="dropdown-menu dropdown-menu-end show"
          style={{ position: "absolute", right: 0, top: "100%", zIndex: 1050 }}
        >
          {visibleItems.map((item, index) => {
            if (item.divider) {
              return <li key={`divider-${index}`}><hr className="dropdown-divider" /></li>;
            }

            const isConfirming = confirmingId === item.id;
            const itemUse = isConfirming ? "warning" : (item.use ?? "default");
            const itemClass = itemUse === "danger" ? "text-danger" :
                            itemUse === "warning" ? "text-warning" : "";

            return (
              <li key={item.id}>
                <button
                  type="button"
                  className={clsx(
                    "dropdown-item d-flex align-items-center gap-2",
                    itemClass,
                    item.disabled && "disabled"
                  )}
                  onClick={() => handleMenuItemClick(item)}
                  disabled={item.disabled}
                >
                  {item.icon}
                  <span>
                    {isConfirming && item.confirmLabel ? item.confirmLabel : item.label}
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
