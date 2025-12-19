import { useState, useEffect } from "react";
import { Save, Zap } from "lucide-react";
import { useDevices, usePowerCycleAllUSB } from "../hooks/useDevices";
import { useUpdateDeviceNickname } from "../hooks/useDeviceNicknames";
import { useToast } from "../hooks/useToast";
import Button from "./primitives/Button.react";
import Flex from "./primitives/Flex.react";
import Spinner from "./primitives/Spinner.react";

interface DeviceSettingsModalProps {
  onClose: () => void;
}

export const DeviceSettingsModal = ({ onClose }: DeviceSettingsModalProps) => {
  const { data: devices, isLoading } = useDevices();
  const updateNickname = useUpdateDeviceNickname();
  const powerCycleAll = usePowerCycleAllUSB();
  const toast = useToast();
  const [nicknames, setNicknames] = useState<Record<string, string>>({});
  const [hasChanges, setHasChanges] = useState(false);
  const [confirmPowerCycle, setConfirmPowerCycle] = useState(false);

  // Initialize nicknames from devices
  useEffect(() => {
    if (devices) {
      const initialNicknames: Record<string, string> = {};
      devices.forEach((device) => {
        initialNicknames[device.id] = device.nickname || "";
      });
      setNicknames(initialNicknames);
    }
  }, [devices]);

  const handleNicknameChange = (deviceId: string, value: string) => {
    setNicknames((prev) => ({ ...prev, [deviceId]: value }));
    setHasChanges(true);
  };

  const handleSave = async () => {
    if (!devices) return;

    const updates = devices
      .filter((device) => {
        const newNickname = nicknames[device.id]?.trim() || null;
        const oldNickname = device.nickname || null;
        return newNickname !== oldNickname;
      })
      .map((device) => ({
        deviceId: device.id,
        nickname: nicknames[device.id]?.trim() || null,
      }));

    // Update all changed nicknames
    for (const update of updates) {
      await updateNickname.mutateAsync(update);
    }

    setHasChanges(false);
    onClose();
  };

  const getDeviceShorthand = (device: any): string => {
    // Extract a short identifier from the device label
    const driver = device.driver.toUpperCase();
    const label = device.label || "";

    // Try to extract serial number or meaningful identifier
    const serialMatch = label.match(/SN:?\s*([A-Za-z0-9]+)/i);
    if (serialMatch) {
      return `${driver} (${serialMatch[1]})`;
    }

    // Fallback to first 40 chars of label
    const shortLabel = label.length > 40 ? label.substring(0, 40) + "..." : label;
    return `${driver} - ${shortLabel}`;
  };

  const handlePowerCycleAll = () => {
    if (!confirmPowerCycle) {
      setConfirmPowerCycle(true);
      return;
    }
    setConfirmPowerCycle(false);
    powerCycleAll.mutate(undefined, {
      onSuccess: (data) => {
        toast.success(data.message);
      },
      onError: (error) => {
        toast.error(`Power cycle failed: ${error.message}`);
      },
    });
  };

  return (
    <div
      className="modal d-block"
      style={{ backgroundColor: "rgba(0,0,0,0.5)" }}
      onClick={onClose}
    >
      <div
        className="modal-dialog modal-dialog-centered modal-lg"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-content">
          <div className="modal-header">
            <h5 className="modal-title">Device Settings</h5>
            <button
              type="button"
              className="btn-close"
              onClick={onClose}
              aria-label="Close"
            />
          </div>

          <div className="modal-body">
            {isLoading ? (
              <Flex justify="center" className="py-4">
                <Spinner />
              </Flex>
            ) : devices && devices.length > 0 ? (
              <div>
                <p className="text-muted small mb-3">
                  Customize device nicknames to make them easier to identify. Leave blank to use auto-detected names.
                </p>
                <div className="table-responsive">
                  <table className="table table-hover">
                    <thead>
                      <tr>
                        <th style={{ width: "35%" }}>Device</th>
                        <th style={{ width: "40%" }}>Auto-Detected</th>
                        <th style={{ width: "25%" }}>Nickname</th>
                      </tr>
                    </thead>
                    <tbody>
                      {devices.map((device) => (
                        <tr key={device.id}>
                          <td>
                            <div className="fw-semibold">{device.driver.toUpperCase()}</div>
                            <div className="small text-muted">ID: {device.id}</div>
                          </td>
                          <td>
                            <div className="small text-muted" style={{ wordBreak: "break-word" }}>
                              {getDeviceShorthand(device)}
                            </div>
                          </td>
                          <td>
                            <input
                              type="text"
                              className="form-control form-control-sm"
                              placeholder="Custom name"
                              value={nicknames[device.id] || ""}
                              onChange={(e) =>
                                handleNicknameChange(device.id, e.target.value)
                              }
                            />
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : (
              <div className="text-center text-muted py-4">
                No devices available
              </div>
            )}
          </div>

          <div className="modal-footer justify-content-between">
            <Button
              use={confirmPowerCycle ? "danger" : "warning"}
              onClick={handlePowerCycleAll}
              disabled={powerCycleAll.isPending}
              onBlur={() => setConfirmPowerCycle(false)}
            >
              <Flex align="center" gap={1}>
                {powerCycleAll.isPending ? (
                  <Spinner size="sm" />
                ) : (
                  <Zap size={16} />
                )}
                <span>{confirmPowerCycle ? "Click to Confirm" : "Power Cycle All USB"}</span>
              </Flex>
            </Button>
            <Flex gap={2}>
              <Button use="secondary" onClick={onClose}>
                Cancel
              </Button>
              <Button
                use="primary"
                onClick={handleSave}
                disabled={!hasChanges || updateNickname.isPending}
              >
                <Flex align="center" gap={1}>
                  {updateNickname.isPending ? (
                    <Spinner size="sm" />
                  ) : (
                    <Save size={16} />
                  )}
                  <span>Save Changes</span>
                </Flex>
              </Button>
            </Flex>
          </div>
        </div>
      </div>
    </div>
  );
};
