import { useState, type ReactNode } from "react";
import clsx from "clsx";
import { ChevronDown, AlertTriangle } from "lucide-react";
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
  const [showMenuModal, setShowMenuModal] = useState(false);
  const [confirmingItem, setConfirmingItem] = useState<DropdownMenuItem | null>(null);

  const handleMenuItemClick = (item: DropdownMenuItem) => {
    if (item.disabled) return;

    if (item.requireConfirm) {
      setConfirmingItem(item);
      setShowMenuModal(false);
    } else {
      item.onClick();
      setShowMenuModal(false);
    }
  };

  const handleConfirm = () => {
    if (confirmingItem) {
      confirmingItem.onClick();
      setConfirmingItem(null);
    }
  };

  const handleCancelConfirm = () => {
    setConfirmingItem(null);
  };

  const btnClass = BUTTON_STYLES[use]?.filled ?? BUTTON_STYLES.default.filled;
  const sizeClass = size !== "md" ? `btn-${size}` : "";

  const visibleItems = menuItems.filter((item) => !item.hidden);

  return (
    <>
      <div className={clsx("btn-group", className)}>
        {/* Main button */}
        <button
          type="button"
          className={clsx("btn", btnClass, sizeClass)}
          onClick={onMainClick}
          disabled={mainDisabled || isPending}
        >
          {isPending && pendingContent ? pendingContent : mainLabel}
        </button>

        {/* Menu toggle button */}
        {visibleItems.length > 0 && (
          <button
            type="button"
            className={clsx("btn", btnClass, sizeClass)}
            onClick={() => setShowMenuModal(true)}
            disabled={isPending}
            aria-label="More options"
          >
            <ChevronDown size={12} />
          </button>
        )}
      </div>

      {/* Menu Modal */}
      {showMenuModal && visibleItems.length > 0 && (
        <div
          className="modal d-block"
          style={{ backgroundColor: "rgba(0,0,0,0.5)" }}
          onClick={() => setShowMenuModal(false)}
        >
          <div
            className="modal-dialog modal-dialog-centered modal-sm"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="modal-content">
              <div className="modal-header py-2">
                <h6 className="modal-title">More Actions</h6>
                <button
                  type="button"
                  className="btn-close btn-close-sm"
                  onClick={() => setShowMenuModal(false)}
                  aria-label="Close"
                />
              </div>
              <div className="modal-body d-flex flex-column gap-2 p-3">
                {visibleItems.map((item, index) => {
                  if (item.divider) {
                    return <hr key={`divider-${index}`} className="my-1" />;
                  }

                  const itemVariant = item.use === "danger" ? "btn-outline-danger" :
                                      item.use === "warning" ? "btn-outline-warning" :
                                      "btn-outline-secondary";

                  return (
                    <button
                      key={item.id}
                      type="button"
                      className={clsx(
                        "btn d-flex align-items-center gap-2",
                        itemVariant,
                        item.disabled && "disabled"
                      )}
                      onClick={() => handleMenuItemClick(item)}
                      disabled={item.disabled}
                    >
                      {item.icon}
                      <span>{item.label}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Confirmation Modal */}
      {confirmingItem && (
        <div
          className="modal d-block"
          style={{ backgroundColor: "rgba(0,0,0,0.5)" }}
          onClick={handleCancelConfirm}
        >
          <div
            className="modal-dialog modal-dialog-centered modal-sm"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="modal-content">
              <div className="modal-header py-2">
                <h6 className="modal-title d-flex align-items-center gap-2">
                  <AlertTriangle size={18} className="text-warning" />
                  Confirm Action
                </h6>
                <button
                  type="button"
                  className="btn-close btn-close-sm"
                  onClick={handleCancelConfirm}
                  aria-label="Close"
                />
              </div>
              <div className="modal-body py-3">
                <p className="mb-0">
                  Are you sure you want to <strong>{confirmingItem.label}</strong>?
                </p>
              </div>
              <div className="modal-footer py-2">
                <button
                  type="button"
                  className="btn btn-sm btn-secondary"
                  onClick={handleCancelConfirm}
                >
                  Cancel
                </button>
                <button
                  type="button"
                  className={clsx(
                    "btn btn-sm",
                    confirmingItem.use === "danger" ? "btn-danger" : "btn-warning"
                  )}
                  onClick={handleConfirm}
                >
                  {confirmingItem.confirmLabel || "Confirm"}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
