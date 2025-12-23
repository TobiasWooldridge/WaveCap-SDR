import { Radio, Zap, Volume2, Settings2 } from "lucide-react";
import type { Capture, Device } from "../../types";
import { useDebouncedMutation } from "../../hooks/useDebouncedMutation";
import { useUpdateCapture } from "../../hooks/useCaptures";
import Flex from "../../components/primitives/Flex.react";
import InfoTooltip from "../../components/primitives/InfoTooltip.react";

interface SDRplaySettingsProps {
  capture: Capture;
  device?: Device;
}

/**
 * SDRplay-specific device settings panel.
 *
 * Provides UI controls for RSPdx-R2 and other SDRplay devices:
 * - rfnotch_ctrl: FM Notch filter (77-115 MHz)
 * - dabnotch_ctrl: DAB Notch filter (155-235 MHz)
 * - biasT_ctrl: Bias-T 4.7V output on Antenna B
 * - hdr_ctrl: HDR mode for frequencies below 2 MHz
 * - agc_setpoint: AGC target level in dBfs
 * - iqcorr_ctrl: IQ imbalance correction
 *
 * Note: Band pass filters are automatic based on tuned frequency.
 */
export function SDRplaySettings({ capture, device: _device }: SDRplaySettingsProps) {
  const updateCapture = useUpdateCapture();
  const isRunning = capture.state === "running";

  // Get current device settings
  const currentSettings = capture.deviceSettings ?? {};

  // Helper to update a single device setting
  const updateSetting = (key: string, value: string) => {
    const newSettings = { ...currentSettings, [key]: value };
    updateCapture.mutate({
      captureId: capture.id,
      request: { deviceSettings: newSettings },
    });
  };

  // Debounced mutations for each setting
  const [fmNotch, setFmNotch] = useDebouncedMutation(
    currentSettings["rfnotch_ctrl"] === "true",
    (value) => updateSetting("rfnotch_ctrl", value ? "true" : "false"),
    { delay: 0 }
  );

  const [dabNotch, setDabNotch] = useDebouncedMutation(
    currentSettings["dabnotch_ctrl"] === "true",
    (value) => updateSetting("dabnotch_ctrl", value ? "true" : "false"),
    { delay: 0 }
  );

  const [biasT, setBiasT] = useDebouncedMutation(
    currentSettings["biasT_ctrl"] === "true",
    (value) => updateSetting("biasT_ctrl", value ? "true" : "false"),
    { delay: 0 }
  );

  const [hdrMode, setHdrMode] = useDebouncedMutation(
    currentSettings["hdr_ctrl"] === "true",
    (value) => updateSetting("hdr_ctrl", value ? "true" : "false"),
    { delay: 0 }
  );

  const [iqCorrection, setIqCorrection] = useDebouncedMutation(
    currentSettings["iqcorr_ctrl"] === "true",
    (value) => updateSetting("iqcorr_ctrl", value ? "true" : "false"),
    { delay: 0 }
  );

  const [agcSetpoint, setAgcSetpoint] = useDebouncedMutation(
    parseInt(currentSettings["agc_setpoint"] || "-30", 10),
    (value) => updateSetting("agc_setpoint", value.toString()),
    { delay: 300 }
  );

  // Check if we're tuned to a frequency where notch filters are relevant
  const centerMHz = capture.centerHz / 1_000_000;
  const fmNotchRelevant = centerMHz >= 77 && centerMHz <= 115;
  const dabNotchRelevant = centerMHz >= 155 && centerMHz <= 235;
  const hdrRelevant = centerMHz <= 2;

  // Determine current antenna from capture
  const currentAntenna = capture.antenna || "Antenna A";
  const biasTAvailable = currentAntenna === "Antenna B";

  return (
    <Flex direction="column" gap={3}>
      {/* Section Header */}
      <Flex align="center" gap={2}>
        <Radio size={16} className="text-primary" />
        <span className="fw-semibold small">SDRplay RSPdx Settings</span>
      </Flex>

      {/* Notch Filters Section */}
      <div className="border rounded p-2">
        <Flex direction="column" gap={2}>
          <Flex align="center" gap={1}>
            <Settings2 size={14} className="text-muted" />
            <span className="small fw-semibold">Notch Filters</span>
          </Flex>

          {/* FM Notch */}
          <div className="form-check">
            <input
              className="form-check-input"
              type="checkbox"
              id="fmNotch"
              checked={fmNotch}
              onChange={(e) => setFmNotch(e.target.checked)}
              disabled={isRunning}
            />
            <label className="form-check-label small" htmlFor="fmNotch">
              <Flex align="center" gap={1}>
                FM Notch (77-115 MHz)
                <InfoTooltip content="Rejects FM broadcast interference. Enable when receiving near the FM band." />
                {fmNotchRelevant && (
                  <span className="badge bg-warning text-dark" style={{ fontSize: "0.65rem" }}>
                    In-Band
                  </span>
                )}
              </Flex>
            </label>
          </div>

          {/* DAB Notch */}
          <div className="form-check">
            <input
              className="form-check-input"
              type="checkbox"
              id="dabNotch"
              checked={dabNotch}
              onChange={(e) => setDabNotch(e.target.checked)}
              disabled={isRunning}
            />
            <label className="form-check-label small" htmlFor="dabNotch">
              <Flex align="center" gap={1}>
                DAB Notch (155-235 MHz)
                <InfoTooltip content="Rejects DAB digital broadcast interference. Enable when receiving in the VHF region." />
                {dabNotchRelevant && (
                  <span className="badge bg-warning text-dark" style={{ fontSize: "0.65rem" }}>
                    In-Band
                  </span>
                )}
              </Flex>
            </label>
          </div>
        </Flex>
      </div>

      {/* Antenna & Power Section */}
      <div className="border rounded p-2">
        <Flex direction="column" gap={2}>
          <Flex align="center" gap={1}>
            <Zap size={14} className="text-muted" />
            <span className="small fw-semibold">Antenna & Power</span>
          </Flex>

          {/* Bias-T */}
          <div className="form-check">
            <input
              className="form-check-input"
              type="checkbox"
              id="biasT"
              checked={biasT}
              onChange={(e) => setBiasT(e.target.checked)}
              disabled={isRunning || !biasTAvailable}
            />
            <label className="form-check-label small" htmlFor="biasT">
              <Flex align="center" gap={1}>
                Bias-T (4.7V on Antenna B)
                <InfoTooltip content="Provides 4.7V DC power to active antennas via Antenna B port. Only available on Antenna B." />
                {!biasTAvailable && (
                  <span className="badge bg-secondary" style={{ fontSize: "0.65rem" }}>
                    Antenna B Only
                  </span>
                )}
              </Flex>
            </label>
          </div>

          {/* HDR Mode */}
          <div className="form-check">
            <input
              className="form-check-input"
              type="checkbox"
              id="hdrMode"
              checked={hdrMode}
              onChange={(e) => setHdrMode(e.target.checked)}
              disabled={isRunning || !hdrRelevant}
            />
            <label className="form-check-label small" htmlFor="hdrMode">
              <Flex align="center" gap={1}>
                HDR Mode (below 2 MHz)
                <InfoTooltip content="High Dynamic Range mode provides improved performance for LF/MW frequencies below 2 MHz." />
                {!hdrRelevant && (
                  <span className="badge bg-secondary" style={{ fontSize: "0.65rem" }}>
                    LF/MW Only
                  </span>
                )}
              </Flex>
            </label>
          </div>
        </Flex>
      </div>

      {/* Signal Processing Section */}
      <div className="border rounded p-2">
        <Flex direction="column" gap={2}>
          <Flex align="center" gap={1}>
            <Volume2 size={14} className="text-muted" />
            <span className="small fw-semibold">Signal Processing</span>
          </Flex>

          {/* IQ Correction */}
          <div className="form-check">
            <input
              className="form-check-input"
              type="checkbox"
              id="iqCorrection"
              checked={iqCorrection}
              onChange={(e) => setIqCorrection(e.target.checked)}
              disabled={isRunning}
            />
            <label className="form-check-label small" htmlFor="iqCorrection">
              <Flex align="center" gap={1}>
                IQ Imbalance Correction
                <InfoTooltip content="Corrects gain and phase imbalance between I and Q channels. Recommended for most use cases." />
              </Flex>
            </label>
          </div>

          {/* AGC Setpoint */}
          <Flex direction="column" gap={1}>
            <Flex justify="between" align="center">
              <Flex align="center" gap={1}>
                <span className="small">AGC Setpoint</span>
                <InfoTooltip content="Target level for automatic gain control in dBfs. Default: -30 dBfs. Lower values = more headroom but potentially weaker signals." />
              </Flex>
              <span className="badge bg-secondary">{agcSetpoint} dBfs</span>
            </Flex>
            <input
              type="range"
              className="form-range"
              min={-60}
              max={0}
              step={1}
              value={agcSetpoint}
              onChange={(e) => setAgcSetpoint(parseInt(e.target.value, 10))}
              disabled={isRunning}
            />
            <Flex justify="between" className="small text-muted">
              <span>-60 dBfs</span>
              <span>0 dBfs</span>
            </Flex>
          </Flex>
        </Flex>
      </div>

      {/* Antenna Port Info */}
      <div className="alert alert-info py-1 px-2 mb-0 small">
        <strong>Current:</strong> {currentAntenna}
        <div className="mt-1 text-muted" style={{ fontSize: "0.75rem" }}>
          <strong>A:</strong> 1 kHz-2 GHz wideband (general) &middot;{" "}
          <strong>B:</strong> 1 kHz-2 GHz + Bias-T &middot;{" "}
          <strong>C:</strong> 1 kHz-200 MHz BNC (HF/VHF)
        </div>
      </div>

      {/* Band Pass Filter Info */}
      <div className="small text-muted">
        <em>
          Band pass filters are automatic: 380-420 MHz (P25 UHF) is active at {centerMHz.toFixed(1)} MHz
        </em>
      </div>

      {isRunning && (
        <small className="text-warning">Stop capture to change device settings</small>
      )}
    </Flex>
  );
}

/**
 * Check if a device is an SDRplay device.
 */
export function isSDRplayDevice(deviceId: string | undefined): boolean {
  if (!deviceId) return false;
  return deviceId.toLowerCase().includes("sdrplay");
}
