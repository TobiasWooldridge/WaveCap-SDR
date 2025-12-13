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
    <div className="d-flex align-items-center bg-dark overflow-auto">
      <div className="d-flex flex-nowrap">
        {tabs.map((tab) => {
          const isSelected = tab.type === selectedType && tab.id === selectedId;

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
            />
          );
        })}
      </div>

      <div className="d-flex align-items-center gap-1 ms-auto px-2">
        {/* Add dropdown or multiple buttons */}
        <div className="dropdown">
          <Button
            size="sm"
            use="light"
            appearance="outline"
            data-bs-toggle="dropdown"
            aria-expanded="false"
          >
            <Plus size={14} />
          </Button>
          <ul className="dropdown-menu dropdown-menu-end">
            <li>
              <button className="dropdown-item d-flex align-items-center gap-2" onClick={onCreateCapture}>
                <Radio size={14} />
                Add Radio
              </button>
            </li>
            {onCreateTrunkingSystem && (
              <li>
                <button className="dropdown-item d-flex align-items-center gap-2" onClick={onCreateTrunkingSystem}>
                  <Antenna size={14} />
                  Add Trunking System
                </button>
              </li>
            )}
          </ul>
        </div>
        <Button size="sm" use="secondary" appearance="outline" onClick={onOpenSettings}>
          <Settings size={14} />
        </Button>
      </div>
    </div>
  );
}

interface TabItemProps {
  tab: RadioTab;
  isSelected: boolean;
  onSelect: () => void;
  onDelete: () => void;
}

function TabItem({ tab, isSelected, onSelect, onDelete }: TabItemProps) {
  const isRunning = tab.state === "running";
  const isFailed = tab.state === "failed" || tab.state === "error";
  const isTrunking = tab.type === "trunking";

  // Trunking states have different semantics
  const trunkingStateLabel = getTrunkingStateLabel(tab.state);
  const stateLabel = isTrunking ? trunkingStateLabel : getCaptureStateLabel(tab.state);
  const stateBadgeClass = getStateBadgeClass(tab.state, isTrunking);

  const Icon = isTrunking ? Antenna : Radio;

  return (
    <div
      className={`
        d-flex align-items-center gap-2 px-3 py-2 border-end border-secondary
        ${isSelected ? "bg-body" : "bg-dark text-light"}
        ${!isSelected && "hover-lighten"}
      `}
      style={{ cursor: "pointer", minWidth: "140px", minHeight: "52px" }}
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
          style={{ maxWidth: "150px" }}
        >
          {tab.name}
        </span>
        <span
          className="text-truncate"
          style={{ fontSize: "0.7rem", maxWidth: "150px", color: isSelected ? "#6c757d" : "#adb5bd" }}
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
