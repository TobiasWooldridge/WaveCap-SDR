import { useEffect } from "react";
import { RotateCcw, RefreshCw, Zap } from "lucide-react";
import type { Capture, Device } from "../../types";
import {
  useUpdateCapture,
  useStartCapture,
  useStopCapture,
  useRestartCapture,
} from "../../hooks/useCaptures";
import { useDevices, useRestartSDRplayService, usePowerCycleDevice } from "../../hooks/useDevices";
import { useToast } from "../../hooks/useToast";
import { getDeviceDisplayName } from "../../utils/device";
import Button from "../../components/primitives/Button.react";
import Flex from "../../components/primitives/Flex.react";
import Spinner from "../../components/primitives/Spinner.react";
import SplitButtonDropdown from "../../components/primitives/SplitButtonDropdown.react";
import { ErrorStatusBar } from "../../components/ErrorStatusBar.react";

interface DeviceControlsProps {
  capture: Capture;
  device: Device | undefined;
}

// Content-only version for use in accordions
export function DeviceControlsContent({ capture, device: _device }: DeviceControlsProps) {
  const { data: devices } = useDevices();
  const updateCapture = useUpdateCapture();
  const startCapture = useStartCapture();
  const stopCapture = useStopCapture();
  const restartCapture = useRestartCapture();
  const restartService = useRestartSDRplayService();
  const powerCycle = usePowerCycleDevice();
  const toast = useToast();

  // Toast feedback for mutations
  useEffect(() => {
    if (updateCapture.isError) {
      toast.error(`Update failed: ${updateCapture.error?.message}`);
    }
  }, [updateCapture.isError, updateCapture.error]);

  useEffect(() => {
    if (powerCycle.isSuccess) toast.success("USB power cycle complete");
    if (powerCycle.isError) toast.error(`Power cycle failed: ${powerCycle.error?.message}`);
  }, [powerCycle.isSuccess, powerCycle.isError, powerCycle.error]);

  useEffect(() => {
    if (restartService.isSuccess) toast.success("SDRplay service restarted");
    if (restartService.isError) toast.error(`Service restart failed: ${restartService.error?.message}`);
  }, [restartService.isSuccess, restartService.isError, restartService.error]);

  const isSDRplay = capture.deviceId?.toLowerCase().includes("sdrplay");
  const isRunning = capture.state === "running";
  const isStarting = capture.state === "starting";
  const isStopping = capture.state === "stopping";
  const isFailed = capture.state === "failed";
  const isError = capture.state === "error";
  const hasError = isFailed || isError;
  const isTransitioning = isStarting || isStopping;
  const anyPending =
    startCapture.isPending ||
    stopCapture.isPending ||
    restartCapture.isPending ||
    restartService.isPending ||
    powerCycle.isPending;

  const handleDeviceChange = (deviceId: string) => {
    const newDevice = devices?.find((d) => d.id === deviceId);
    if (!newDevice) return;

    updateCapture.mutate({
      captureId: capture.id,
      request: {
        deviceId,
        sampleRate: newDevice.sampleRates[0],
      },
    });
  };

  return (
    <Flex direction="column" gap={2}>
      {/* Device Selector */}
      <div>
        <label className="form-label mb-1 small fw-semibold">Radio Device</label>
        <select
          className="form-select form-select-sm"
          value={capture.deviceId}
          onChange={(e) => handleDeviceChange(e.target.value)}
          disabled={isRunning}
        >
          {(devices || []).map((dev) => (
            <option key={dev.id} value={dev.id}>
              {getDeviceDisplayName(dev)}
            </option>
          ))}
        </select>
        {isRunning && (
          <small className="text-warning d-block mt-1" style={{ fontSize: "0.7rem" }}>
            Stop to change device
          </small>
        )}
      </div>

      {/* Control Buttons */}
      <div className="d-flex gap-2 align-items-center flex-wrap">
        <SplitButtonDropdown
          mainLabel={
            isStarting
              ? "Starting..."
              : isStopping
              ? "Stopping..."
              : isRunning
              ? "Stop Capture"
              : "Start Capture"
          }
          onMainClick={() => {
            if (isRunning) {
              stopCapture.mutate(capture.id);
            } else {
              startCapture.mutate(capture.id);
            }
          }}
          mainDisabled={anyPending || isTransitioning}
          use={isRunning || isStopping ? "danger" : isStarting ? "warning" : "success"}
          isPending={anyPending}
          pendingContent={<Spinner size="sm" />}
          menuItems={[
            {
              id: "restart",
              label: "Restart Capture",
              icon: <RotateCcw size={14} />,
              onClick: () => restartCapture.mutate(capture.id),
              disabled: restartCapture.isPending || isTransitioning,
              requireConfirm: true,
              confirmLabel: "Restart",
            },
            {
              id: "divider1",
              label: "",
              divider: true,
              onClick: () => {},
            },
            {
              id: "service",
              label: "Restart SDRplay Service",
              icon: <RefreshCw size={14} />,
              onClick: () => restartService.mutate(),
              disabled: restartService.isPending,
              hidden: !isSDRplay,
              use: "danger",
              requireConfirm: true,
              confirmLabel: "Restart Service",
            },
            {
              id: "powercycle",
              label: "Power Cycle USB",
              icon: <Zap size={14} />,
              onClick: () => powerCycle.mutate(capture.id),
              disabled: powerCycle.isPending || isTransitioning,
              use: "danger",
              requireConfirm: true,
              confirmLabel: "Power Cycle",
            },
          ]}
        />
      </div>

      {/* Error Display */}
      {hasError && (
        <div className="alert alert-danger mb-0 py-1 px-2">
          <div className="d-flex align-items-center justify-content-between flex-wrap gap-1">
            <small>
              <strong>{isError ? "Device Error:" : "Error:"}</strong>{" "}
              {capture.errorMessage || "Unknown error"}
            </small>
            <div className="d-flex gap-1">
              <Button
                use="warning"
                size="sm"
                onClick={() => restartCapture.mutate(capture.id)}
                disabled={anyPending}
              >
                Restart
              </Button>
              {isSDRplay && (
                <Button
                  use="danger"
                  size="sm"
                  onClick={() => restartService.mutate()}
                  disabled={anyPending}
                >
                  Restart Service
                </Button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Real-time Error Status */}
      {isRunning && <ErrorStatusBar captureId={capture.id} capture={capture} />}
    </Flex>
  );
}

// Legacy wrapper for backwards compatibility
export function DeviceControls({ capture, device }: DeviceControlsProps) {
  return <DeviceControlsContent capture={capture} device={device} />;
}
