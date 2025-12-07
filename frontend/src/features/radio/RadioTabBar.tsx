import { useMemo } from "react";
import { Radio, Plus, X, Settings } from "lucide-react";
import type { Capture, Device } from "../../types";
import { groupCapturesByDevice, findDeviceForCapture } from "../../utils/deviceId";
import { getDeviceDisplayName } from "../../utils/device";
import { formatFrequencyMHz } from "../../utils/frequency";
import Button from "../../components/primitives/Button.react";

interface RadioTabBarProps {
  captures: Capture[];
  devices: Device[];
  selectedCaptureId: string | null;
  onSelectCapture: (id: string) => void;
  onCreateCapture: () => void;
  onDeleteCapture: (id: string) => void;
  onOpenSettings: () => void;
}

export function RadioTabBar({
  captures,
  devices,
  selectedCaptureId,
  onSelectCapture,
  onCreateCapture,
  onDeleteCapture,
  onOpenSettings,
}: RadioTabBarProps) {
  // Group captures by device for organized display
  const captureGroups = useMemo(
    () => groupCapturesByDevice(captures, devices),
    [captures, devices]
  );

  if (captures.length === 0) {
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
        {captureGroups.map((group) =>
          group.captures.map((capture) => {
            const device = findDeviceForCapture(devices, capture);
            const isSelected = capture.id === selectedCaptureId;
            const isRunning = capture.state === "running";
            const isFailed = capture.state === "failed" || capture.state === "error";

            return (
              <RadioTab
                key={capture.id}
                capture={capture}
                device={device ?? null}
                isSelected={isSelected}
                isRunning={isRunning}
                isFailed={isFailed}
                onSelect={() => onSelectCapture(capture.id)}
                onDelete={() => onDeleteCapture(capture.id)}
              />
            );
          })
        )}
      </div>

      <div className="d-flex align-items-center gap-1 ms-auto px-2">
        <Button size="sm" use="light" appearance="outline" onClick={onCreateCapture}>
          <Plus size={14} />
        </Button>
        <Button size="sm" use="secondary" appearance="outline" onClick={onOpenSettings}>
          <Settings size={14} />
        </Button>
      </div>
    </div>
  );
}

interface RadioTabProps {
  capture: Capture;
  device: Device | null;
  isSelected: boolean;
  isRunning: boolean;
  isFailed: boolean;
  onSelect: () => void;
  onDelete: () => void;
}

function RadioTab({
  capture,
  device,
  isSelected,
  isRunning,
  isFailed,
  onSelect,
  onDelete,
}: RadioTabProps) {
  const displayName = capture.name || capture.autoName || formatCaptureId(capture.id);
  const deviceName = device ? getDeviceDisplayName(device) : "Unknown Device";

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
      <Radio
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
          {displayName}
        </span>
        <span
          className="text-truncate"
          style={{ fontSize: "0.7rem", maxWidth: "150px", color: isSelected ? "#6c757d" : "#adb5bd" }}
        >
          {deviceName} - {formatFrequencyMHz(capture.centerHz)} MHz
        </span>
      </div>

      <div className="ms-auto d-flex align-items-center gap-1">
        <span
          className={`badge ${
            isFailed
              ? "bg-danger"
              : isRunning
              ? "bg-success"
              : capture.state === "starting"
              ? "bg-warning text-dark"
              : "bg-secondary"
          }`}
          style={{ fontSize: "0.6rem" }}
        >
          {capture.state === "running" ? "ON" : capture.state === "stopped" ? "OFF" : capture.state.toUpperCase()}
        </span>

        {isSelected && (
          <button
            className="btn btn-sm p-0 border-0"
            onClick={(e) => {
              e.stopPropagation();
              onDelete();
            }}
            title="Delete capture"
          >
            <X size={14} className="text-muted" />
          </button>
        )}
      </div>
    </div>
  );
}

function formatCaptureId(id: string): string {
  const match = id.match(/^c(\d+)$/);
  return match ? `Radio ${match[1]}` : id;
}
