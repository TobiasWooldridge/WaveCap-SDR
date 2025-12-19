import React, { useMemo, useState } from "react";
import { Radio, Antenna, Plus, X, Settings } from "lucide-react";
import type { RadioTab, RadioTabType } from "../../types";
import { formatFrequencyMHz } from "../../utils/frequency";
import Button from "../../components/primitives/Button.react";

interface RadioTabBarProps {
  tabs: RadioTab[];
  selectedType: RadioTabType | null;
  selectedId: string | null;
  onSelectTab: (type: RadioTabType, id: string) => void;
  onCreateCapture: () => void;
  onCreateTrunkingSystem?: () => void;
  onDeleteCapture: (id: string) => void;
  onDeleteTrunkingSystem?: (id: string) => void;
  onOpenSettings: () => void;
}

// Group tabs by deviceId for visual coupling
interface DeviceGroup {
  deviceId: string;
  tabs: RadioTab[];
}

export function RadioTabBar({
  tabs,
  selectedType,
  selectedId,
  onSelectTab,
  onCreateCapture,
  onCreateTrunkingSystem,
  onDeleteCapture,
  onDeleteTrunkingSystem,
  onOpenSettings,
}: RadioTabBarProps) {
  const [showAddModal, setShowAddModal] = useState(false);

  // Group tabs by deviceId for visual coupling
  const deviceGroups = useMemo(() => {
    const groups: DeviceGroup[] = [];
    const deviceMap = new Map<string, RadioTab[]>();

    for (const tab of tabs) {
      const key = tab.deviceId || `ungrouped-${tab.id}`;
      if (!deviceMap.has(key)) {
        deviceMap.set(key, []);
      }
      deviceMap.get(key)!.push(tab);
    }

    // Convert map to array, putting captures before trunking in each group
    for (const [deviceId, groupTabs] of deviceMap) {
      // Sort: captures first, then trunking
      const sorted = [...groupTabs].sort((a, b) => {
        if (a.type === "capture" && b.type === "trunking") return -1;
        if (a.type === "trunking" && b.type === "capture") return 1;
        return 0;
      });
      groups.push({ deviceId, tabs: sorted });
    }

    return groups;
  }, [tabs]);

  if (tabs.length === 0) {
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

  return (
    <div className="d-flex align-items-center bg-dark">
      {/* Scrollable tabs area */}
      <div className="d-flex flex-nowrap overflow-auto flex-grow-1">
        {deviceGroups.map((group, groupIndex) => (
          <DeviceGroupTabs
            key={group.deviceId}
            group={group}
            selectedType={selectedType}
            selectedId={selectedId}
            onSelectTab={onSelectTab}
            onDeleteCapture={onDeleteCapture}
            onDeleteTrunkingSystem={onDeleteTrunkingSystem}
            isLastGroup={groupIndex === deviceGroups.length - 1}
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
        >
          <Plus size={14} />
        </Button>
        <Button size="sm" use="secondary" appearance="outline" onClick={onOpenSettings}>
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
    </div>
  );
}

interface DeviceGroupTabsProps {
  group: DeviceGroup;
  selectedType: RadioTabType | null;
  selectedId: string | null;
  onSelectTab: (type: RadioTabType, id: string) => void;
  onDeleteCapture: (id: string) => void;
  onDeleteTrunkingSystem?: (id: string) => void;
  isLastGroup: boolean;
}

function DeviceGroupTabs({
  group,
  selectedType,
  selectedId,
  onSelectTab,
  onDeleteCapture,
  onDeleteTrunkingSystem,
  isLastGroup,
}: DeviceGroupTabsProps) {
  const hasMultipleTabs = group.tabs.length > 1;

  return (
    <div
      className="d-flex flex-nowrap"
      style={{
        // Visually couple tabs sharing the same device with a subtle background
        backgroundColor: hasMultipleTabs ? "rgba(255, 255, 255, 0.03)" : "transparent",
        borderRight: isLastGroup ? "none" : "2px solid #343a40",
      }}
    >
      {group.tabs.map((tab, index) => {
        const isSelected = tab.type === selectedType && tab.id === selectedId;
        const isFirstInGroup = index === 0;
        const isLastInGroup = index === group.tabs.length - 1;

        return (
          <TabItem
            key={`${tab.type}:${tab.id}`}
            tab={tab}
            isSelected={isSelected}
            onSelect={() => onSelectTab(tab.type, tab.id)}
            onDelete={() => {
              if (tab.type === "capture") {
                onDeleteCapture(tab.id);
              } else if (onDeleteTrunkingSystem) {
                onDeleteTrunkingSystem(tab.id);
              }
            }}
            isFirstInGroup={isFirstInGroup}
            isLastInGroup={isLastInGroup}
            hasMultipleTabs={hasMultipleTabs}
          />
        );
      })}
    </div>
  );
}

interface TabItemProps {
  tab: RadioTab;
  isSelected: boolean;
  onSelect: () => void;
  onDelete: () => void;
  isFirstInGroup: boolean;
  isLastInGroup: boolean;
  hasMultipleTabs: boolean;
}

function TabItem({
  tab,
  isSelected,
  onSelect,
  onDelete,
  isFirstInGroup,
  isLastInGroup,
  hasMultipleTabs,
}: TabItemProps) {
  const isRunning = tab.state === "running";
  const isFailed = tab.state === "failed" || tab.state === "error";
  const isTrunking = tab.type === "trunking";

  // Trunking states have different semantics
  const trunkingStateLabel = getTrunkingStateLabel(tab.state);
  const stateLabel = isTrunking ? trunkingStateLabel : getCaptureStateLabel(tab.state);
  const stateBadgeClass = getStateBadgeClass(tab.state, isTrunking);

  const Icon = isTrunking ? Antenna : Radio;

  // When multiple tabs share a device, use a connecting visual style
  const groupStyle: React.CSSProperties = hasMultipleTabs
    ? {
        // Smaller gap between grouped tabs
        borderRight: isLastInGroup ? "none" : "1px dashed #495057",
        // Subtle left border on non-first items to show connection
        borderLeft: isFirstInGroup ? "none" : "none",
      }
    : {};

  return (
    <div
      className={`
        d-flex align-items-center gap-2 px-3 py-2
        ${!hasMultipleTabs || isLastInGroup ? "border-end border-secondary" : ""}
        ${isSelected ? "bg-body" : "bg-dark text-light"}
        ${!isSelected && "hover-lighten"}
      `}
      style={{ cursor: "pointer", minWidth: "180px", minHeight: "52px", ...groupStyle }}
      onClick={onSelect}
    >
      <Icon
        size={14}
        className={
          isFailed ? "text-danger" : isRunning ? "text-success" : "text-secondary"
        }
      />

      <div className="d-flex flex-column overflow-hidden">
        <span
          className={`small fw-semibold text-truncate ${isSelected ? "" : "text-light"}`}
          style={{ maxWidth: "200px" }}
        >
          {tab.name}
        </span>
        <span
          className="text-truncate"
          style={{ fontSize: "0.7rem", maxWidth: "200px", color: isSelected ? "#6c757d" : "#adb5bd" }}
        >
          {tab.deviceName}
          {tab.frequencyHz > 0 && ` - ${formatFrequencyMHz(tab.frequencyHz)} MHz`}
        </span>
      </div>

      <div className="ms-auto d-flex align-items-center gap-1">
        <span
          className={`badge ${stateBadgeClass}`}
          style={{ fontSize: "0.6rem" }}
        >
          {stateLabel}
        </span>

        {isSelected && (
          <button
            className="btn btn-sm p-0 border-0"
            onClick={(e) => {
              e.stopPropagation();
              onDelete();
            }}
            title={isTrunking ? "Delete trunking system" : "Delete capture"}
          >
            <X size={14} className="text-muted" />
          </button>
        )}
      </div>
    </div>
  );
}

function getCaptureStateLabel(state: string): string {
  switch (state) {
    case "running":
      return "ON";
    case "stopped":
      return "OFF";
    case "starting":
      return "STARTING";
    case "stopping":
      return "STOPPING";
    case "failed":
    case "error":
      return "FAILED";
    default:
      return state.toUpperCase();
  }
}

function getTrunkingStateLabel(state: string): string {
  switch (state) {
    case "running":
      return "SYNCED";
    case "stopped":
      return "OFF";
    case "starting":
      return "STARTING";
    case "searching":
      return "SEARCH";
    case "syncing":
      return "SYNCING";
    case "failed":
      return "FAILED";
    default:
      return state.toUpperCase();
  }
}

function getStateBadgeClass(state: string, isTrunking: boolean): string {
  if (isTrunking) {
    switch (state) {
      case "running":
        return "bg-success";
      case "stopped":
        return "bg-secondary";
      case "searching":
      case "syncing":
      case "starting":
        return "bg-warning text-dark";
      case "failed":
        return "bg-danger";
      default:
        return "bg-secondary";
    }
  } else {
    // Capture states
    switch (state) {
      case "running":
        return "bg-success";
      case "stopped":
        return "bg-secondary";
      case "starting":
      case "stopping":
        return "bg-warning text-dark";
      case "failed":
      case "error":
        return "bg-danger";
      default:
        return "bg-secondary";
    }
  }
}
