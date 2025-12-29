import { useState } from "react";
import { Radio, Antenna, Plus, X, Settings } from "lucide-react";
import type { DeviceTab } from "../types";
import { formatFrequencyMHz } from "../utils/frequency";
import Button from "./primitives/Button.react";
import {
  StatusPill,
  getCaptureStatusProps,
  getTrunkingStatusProps,
} from "./primitives/StatusPill.react";

interface DeviceTabBarProps {
  /** Array of device tabs to display */
  deviceTabs: DeviceTab[];
  /** Currently selected device ID */
  selectedDeviceId: string | null;
  /** Callback when a device is selected */
  onSelectDevice: (deviceId: string) => void;
  /** Callback to create a new capture */
  onCreateCapture: () => void;
  /** Callback to create a new trunking system */
  onCreateTrunkingSystem?: () => void;
  /** Callback to delete a capture */
  onDeleteCapture: (captureId: string) => void;
  /** Callback to delete a trunking system */
  onDeleteTrunkingSystem?: (systemId: string) => void;
  /** Callback to open device settings */
  onOpenSettings: () => void;
}

/**
 * Level 1 navigation bar showing one tab per SDR device.
 *
 * Each device tab shows:
 * - Device name and frequency
 * - Mode icons (radio/trunking) showing what's configured
 * - Dual status pills: one for Radio, one for Trunking (when both exist)
 * - Active calls count with pulsing animation
 */
export function DeviceTabBar({
  deviceTabs,
  selectedDeviceId,
  onSelectDevice,
  onCreateCapture,
  onCreateTrunkingSystem,
  onDeleteCapture,
  onDeleteTrunkingSystem,
  onOpenSettings,
}: DeviceTabBarProps) {
  const [showAddModal, setShowAddModal] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState<string | null>(null);

  if (deviceTabs.length === 0) {
    return (
      <div className="d-flex align-items-center gap-2 p-2 bg-dark">
        <Radio size={18} className="text-light" />
        <span className="text-light small">No radios configured</span>
        <Button size="sm" use="primary" onClick={onCreateCapture}>
          <Plus size={14} className="me-1" />
          Add Radio
        </Button>
        <Button size="sm" use="secondary" appearance="outline" onClick={onOpenSettings} className="ms-auto">
          <Settings size={14} />
        </Button>
      </div>
    );
  }

  const handleDeleteDevice = (tab: DeviceTab) => {
    // Delete capture if present
    if (tab.capture) {
      onDeleteCapture(tab.capture.id);
    }
    // Delete trunking if present
    if (tab.trunkingSystem && onDeleteTrunkingSystem) {
      onDeleteTrunkingSystem(tab.trunkingSystem.id);
    }
    setShowDeleteConfirm(null);
  };

  return (
    <div className="d-flex align-items-center bg-dark">
      {/* Scrollable tabs area */}
      <div className="d-flex flex-nowrap overflow-auto flex-grow-1">
        {deviceTabs.map((tab) => (
          <DeviceTabItem
            key={tab.deviceId}
            tab={tab}
            isSelected={tab.deviceId === selectedDeviceId}
            onSelect={() => onSelectDevice(tab.deviceId)}
            onDelete={() => setShowDeleteConfirm(tab.deviceId)}
          />
        ))}
      </div>

      {/* Fixed buttons area */}
      <div className="d-flex align-items-center gap-1 px-2 flex-shrink-0">
        <Button
          size="sm"
          use="light"
          appearance="outline"
          onClick={() => setShowAddModal(true)}
          title="Add radio or trunking system"
        >
          <Plus size={14} />
        </Button>
        <Button size="sm" use="secondary" appearance="outline" onClick={onOpenSettings} title="Device settings">
          <Settings size={14} />
        </Button>
      </div>

      {/* Add Radio/Trunking Modal */}
      {showAddModal && (
        <div
          className="modal d-block"
          style={{ backgroundColor: "rgba(0,0,0,0.5)" }}
          onClick={() => setShowAddModal(false)}
        >
          <div
            className="modal-dialog modal-dialog-centered modal-sm"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="modal-content">
              <div className="modal-header py-2">
                <h6 className="modal-title">Add New</h6>
                <button
                  type="button"
                  className="btn-close btn-close-sm"
                  onClick={() => setShowAddModal(false)}
                  aria-label="Close"
                />
              </div>
              <div className="modal-body d-flex flex-column gap-2">
                <button
                  className="btn btn-outline-primary d-flex align-items-center gap-2"
                  onClick={() => {
                    setShowAddModal(false);
                    onCreateCapture();
                  }}
                >
                  <Radio size={16} />
                  Add Radio
                </button>
                {onCreateTrunkingSystem && (
                  <button
                    className="btn btn-outline-secondary d-flex align-items-center gap-2"
                    onClick={() => {
                      setShowAddModal(false);
                      onCreateTrunkingSystem();
                    }}
                  >
                    <Antenna size={16} />
                    Add Trunking System
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Delete Confirmation Modal */}
      {showDeleteConfirm && (
        <div
          className="modal d-block"
          style={{ backgroundColor: "rgba(0,0,0,0.5)" }}
          onClick={() => setShowDeleteConfirm(null)}
        >
          <div
            className="modal-dialog modal-dialog-centered modal-sm"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="modal-content">
              <div className="modal-header py-2">
                <h6 className="modal-title">Delete Device</h6>
                <button
                  type="button"
                  className="btn-close btn-close-sm"
                  onClick={() => setShowDeleteConfirm(null)}
                  aria-label="Close"
                />
              </div>
              <div className="modal-body">
                {(() => {
                  const tab = deviceTabs.find((t) => t.deviceId === showDeleteConfirm);
                  if (!tab) return null;
                  return (
                    <div>
                      <p className="mb-2">Delete this device configuration?</p>
                      <ul className="small text-muted mb-0">
                        {tab.capture && <li>Radio capture and all channels</li>}
                        {tab.trunkingSystem && <li>Trunking system: {tab.trunkingSystem.name}</li>}
                      </ul>
                    </div>
                  );
                })()}
              </div>
              <div className="modal-footer py-2">
                <button
                  className="btn btn-sm btn-secondary"
                  onClick={() => setShowDeleteConfirm(null)}
                >
                  Cancel
                </button>
                <button
                  className="btn btn-sm btn-danger"
                  onClick={() => {
                    const tab = deviceTabs.find((t) => t.deviceId === showDeleteConfirm);
                    if (tab) handleDeleteDevice(tab);
                  }}
                >
                  Delete
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

interface DeviceTabItemProps {
  tab: DeviceTab;
  isSelected: boolean;
  onSelect: () => void;
  onDelete: () => void;
}

/**
 * Get tooltip text for mode indicators
 */
function getModeIndicatorTooltip(tab: DeviceTab): string {
  const modes: string[] = [];
  if (tab.hasRadio) modes.push("Radio");
  if (tab.hasTrunking) modes.push("Trunking");
  return modes.join(" + ") + " available";
}

function DeviceTabItem({ tab, isSelected, onSelect, onDelete }: DeviceTabItemProps) {
  // Get status props for radio (capture)
  const radioStatus = tab.capture
    ? getCaptureStatusProps(tab.capture.state)
    : null;

  // Get status props for trunking with expressive state
  const trunkingStatus = tab.trunkingSystem
    ? getTrunkingStatusProps(
        tab.trunkingSystem.state,
        tab.controlChannelState,
        tab.activeCalls,
        tab.isManuallyLocked
      )
    : null;

  // Determine icon colors based on running states
  const radioIconColor = tab.capture?.state === "running"
    ? "text-success"
    : tab.capture?.state === "failed" || tab.capture?.state === "error"
    ? "text-danger"
    : "text-secondary";

  const trunkingIconColor = tab.trunkingSystem?.state === "running"
    ? tab.controlChannelState === "lost"
      ? "text-danger"
      : "text-success"
    : tab.trunkingSystem?.state === "failed"
    ? "text-danger"
    : "text-secondary";

  return (
    <div
      className={`
        d-flex align-items-center gap-2 px-3 py-2
        border-end border-secondary
        ${isSelected ? "bg-body" : "bg-dark text-light"}
        ${!isSelected && "hover-lighten"}
      `}
      style={{ cursor: "pointer", minWidth: "220px", minHeight: "52px" }}
      onClick={onSelect}
    >
      {/* Mode indicators */}
      <div className="d-flex flex-column gap-1" title={getModeIndicatorTooltip(tab)}>
        {tab.hasRadio && (
          <Radio size={12} className={radioIconColor} />
        )}
        {tab.hasTrunking && (
          <Antenna size={12} className={trunkingIconColor} />
        )}
      </div>

      {/* Device info */}
      <div className="d-flex flex-column overflow-hidden">
        <span
          className={`small fw-semibold text-truncate ${isSelected ? "" : "text-light"}`}
          style={{ maxWidth: "140px" }}
        >
          {tab.deviceName}
        </span>
        <span
          className="text-truncate"
          style={{ fontSize: "0.7rem", maxWidth: "140px", color: isSelected ? "#6c757d" : "#adb5bd" }}
        >
          {tab.frequencyHz > 0 ? `${formatFrequencyMHz(tab.frequencyHz)} MHz` : "No frequency set"}
        </span>
      </div>

      {/* Status pills - inline for Radio and Trunking */}
      <div className="ms-auto d-flex align-items-center gap-1">
        {/* Radio status pill */}
        {radioStatus && (
          <StatusPill
            label={radioStatus.label}
            variant={radioStatus.variant}
            pulsing={radioStatus.pulsing}
          />
        )}

        {/* Trunking status pill */}
        {trunkingStatus && (
          <StatusPill
            label={trunkingStatus.label}
            variant={trunkingStatus.variant}
            count={trunkingStatus.count}
            pulsing={trunkingStatus.pulsing}
          />
        )}

        {/* Delete button when selected */}
        {isSelected && (
          <button
            className="btn btn-sm p-0 border-0"
            onClick={(e) => {
              e.stopPropagation();
              onDelete();
            }}
            title="Delete device configuration"
          >
            <X size={14} className="text-muted" />
          </button>
        )}
      </div>
    </div>
  );
}
