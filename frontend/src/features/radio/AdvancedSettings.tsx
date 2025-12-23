import { useState } from "react";
import { Settings, ChevronDown, ChevronUp } from "lucide-react";
import type { Capture, Device } from "../../types";
import { useDebouncedMutation } from "../../hooks/useDebouncedMutation";
import { useUpdateCapture } from "../../hooks/useCaptures";
import Flex from "../../components/primitives/Flex.react";
import { SDRplaySettings, isSDRplayDevice } from "./SDRplaySettings";

interface AdvancedSettingsProps {
  capture: Capture;
  device?: Device;
}

// Content-only version for use in accordions
export function AdvancedSettingsContent({ capture, device: _device }: AdvancedSettingsProps) {
  const updateCapture = useUpdateCapture();
  const isRunning = capture.state === "running";

  // Debounced mutations for advanced settings
  const [dcOffsetAuto, setDcOffsetAuto] = useDebouncedMutation(
    capture.dcOffsetAuto ?? true,
    (value) => updateCapture.mutate({ captureId: capture.id, request: { dcOffsetAuto: value } }),
    { delay: 0 }
  );

  const [iqBalanceAuto, setIqBalanceAuto] = useDebouncedMutation(
    capture.iqBalanceAuto ?? true,
    (value) => updateCapture.mutate({ captureId: capture.id, request: { iqBalanceAuto: value } }),
    { delay: 0 }
  );

  const [streamFormat, setStreamFormat] = useDebouncedMutation(
    capture.streamFormat ?? "",
    (value) => updateCapture.mutate({ captureId: capture.id, request: { streamFormat: value || undefined } }),
    { delay: 0 }
  );

  // FFT/Spectrum settings (can be changed while running)
  const [fftFps, setFftFps] = useDebouncedMutation(
    capture.fftFps ?? 15,
    (value) => updateCapture.mutate({ captureId: capture.id, request: { fftFps: value } }),
    { delay: 300 }
  );

  const [fftSize, setFftSize] = useDebouncedMutation(
    capture.fftSize ?? 2048,
    (value) => updateCapture.mutate({ captureId: capture.id, request: { fftSize: value } }),
    { delay: 0 }
  );

  const [elementGains, setElementGains] = useDebouncedMutation(
    capture.elementGains ?? {},
    (value) => updateCapture.mutate({ captureId: capture.id, request: { elementGains: value } }),
    { delay: 100, isEqual: (a, b) => JSON.stringify(a) === JSON.stringify(b) }
  );

  const [deviceSettings, setDeviceSettings] = useDebouncedMutation(
    capture.deviceSettings ?? {},
    (value) => updateCapture.mutate({ captureId: capture.id, request: { deviceSettings: value } }),
    { delay: 100, isEqual: (a, b) => JSON.stringify(a) === JSON.stringify(b) }
  );

  const hasElementGains = Object.keys(elementGains).length > 0;
  const hasDeviceSettings = Object.keys(deviceSettings).length > 0;

  return (
    <Flex direction="column" gap={2}>
      {/* DC Offset Auto */}
      <div className="form-check">
        <input
          className="form-check-input"
          type="checkbox"
          id="dcOffsetAuto"
          checked={dcOffsetAuto}
          onChange={(e) => setDcOffsetAuto(e.target.checked)}
          disabled={isRunning}
        />
        <label className="form-check-label" htmlFor="dcOffsetAuto">
          DC Offset Auto-Correction
        </label>
        <div className="form-text">Automatically remove DC bias from IQ samples</div>
      </div>

      {/* IQ Balance Auto */}
      <div className="form-check">
        <input
          className="form-check-input"
          type="checkbox"
          id="iqBalanceAuto"
          checked={iqBalanceAuto}
          onChange={(e) => setIqBalanceAuto(e.target.checked)}
          disabled={isRunning}
        />
        <label className="form-check-label" htmlFor="iqBalanceAuto">
          IQ Balance Auto-Correction
        </label>
        <div className="form-text">Automatically correct IQ imbalance</div>
      </div>

      {/* Stream Format */}
      <Flex direction="column" gap={1}>
        <label className="form-label mb-0 small fw-semibold">Stream Format</label>
        <select
          className="form-select form-select-sm"
          value={streamFormat}
          onChange={(e) => setStreamFormat(e.target.value)}
          disabled={isRunning}
        >
          <option value="">Auto (CF32)</option>
          <option value="CF32">CF32 (Complex Float32)</option>
          <option value="CS16">CS16 (Complex Int16)</option>
          <option value="CS8">CS8 (Complex Int8)</option>
        </select>
        <small className="text-muted">Stream format affects bandwidth and precision</small>
      </Flex>

      {/* Spectrum Analyzer Settings */}
      <Flex direction="column" gap={2}>
        <label className="form-label mb-0 small fw-semibold">Spectrum Analyzer</label>

        {/* FFT FPS Slider */}
        <Flex direction="column" gap={1}>
          <Flex justify="between" align="center">
            <label className="form-label mb-0 small">Target FPS</label>
            <span className="badge bg-secondary">{fftFps} FPS</span>
          </Flex>
          <input
            type="range"
            className="form-range"
            min={5}
            max={60}
            step={5}
            value={fftFps}
            onChange={(e) => setFftFps(parseInt(e.target.value, 10))}
          />
          <small className="text-muted">
            Higher FPS uses more CPU. Adaptive: doubles when viewing spectrum.
          </small>
        </Flex>

        {/* FFT Size Dropdown */}
        <Flex direction="column" gap={1}>
          <label className="form-label mb-0 small">FFT Size (Resolution)</label>
          <select
            className="form-select form-select-sm"
            value={fftSize}
            onChange={(e) => setFftSize(parseInt(e.target.value, 10))}
          >
            <option value={512}>512 bins (fast, low resolution)</option>
            <option value={1024}>1024 bins</option>
            <option value={2048}>2048 bins (default)</option>
            <option value={4096}>4096 bins (high resolution)</option>
          </select>
          <small className="text-muted">Larger FFT = better frequency resolution but more CPU</small>
        </Flex>
      </Flex>

      {/* Element Gains */}
      {hasElementGains && (
        <Flex direction="column" gap={2}>
          <Flex direction="column" gap={0}>
            <label className="form-label mb-0 small fw-semibold">Element Gains</label>
            <small className="text-muted">
              Individual gain controls for RF stages (LNA, IF, etc.)
            </small>
          </Flex>
          {Object.entries(elementGains).map(([key, value]) => (
            <Flex key={key} direction="column" gap={1}>
              <label className="form-label mb-0 small">{key}</label>
              <input
                type="number"
                className="form-control form-control-sm"
                value={value}
                onChange={(e) =>
                  setElementGains({ ...elementGains, [key]: parseFloat(e.target.value) })
                }
                disabled={isRunning}
                step="0.1"
              />
            </Flex>
          ))}
        </Flex>
      )}

      {/* Device Settings - SDRplay-specific panel or generic */}
      {isSDRplayDevice(capture.deviceId) ? (
        <SDRplaySettings capture={capture} device={_device} />
      ) : hasDeviceSettings ? (
        <Flex direction="column" gap={2}>
          <label className="form-label mb-0 small fw-semibold">Device Settings</label>
          {Object.entries(deviceSettings).map(([key, value]) => (
            <Flex key={key} direction="column" gap={1}>
              <label className="form-label mb-0 small">{key}</label>
              <input
                type="text"
                className="form-control form-control-sm"
                value={value}
                onChange={(e) =>
                  setDeviceSettings({ ...deviceSettings, [key]: e.target.value })
                }
                disabled={isRunning}
              />
            </Flex>
          ))}
        </Flex>
      ) : null}

      {!isRunning && (
        <small className="text-muted">Start capture to apply changes</small>
      )}
    </Flex>
  );
}

// Legacy wrapper for backwards compatibility
export function AdvancedSettings({ capture, device }: AdvancedSettingsProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className="card shadow-sm">
      <div
        className="card-header bg-body-tertiary py-1 px-2"
        style={{ cursor: "pointer" }}
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <Flex align="center" gap={1}>
          <Settings size={14} />
          <small className="fw-semibold mb-0">Advanced Settings</small>
          <div className="ms-auto">
            {isExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </div>
        </Flex>
      </div>
      {isExpanded && (
        <div className="card-body" style={{ padding: "0.75rem" }}>
          <AdvancedSettingsContent capture={capture} device={device} />
        </div>
      )}
    </div>
  );
}
