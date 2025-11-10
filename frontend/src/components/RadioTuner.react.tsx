import { useState, useEffect } from "react";
import { Radio, Settings, ChevronDown, ChevronUp } from "lucide-react";
import type { Capture, Device } from "../types";
import { useUpdateCapture, useStartCapture, useStopCapture } from "../hooks/useCaptures";
import { useDebounce } from "../hooks/useDebounce";
import { formatFrequencyMHz, formatSampleRate } from "../utils/frequency";
import Button from "./primitives/Button.react";
import Flex from "./primitives/Flex.react";
import Slider from "./primitives/Slider.react";
import FrequencySelector from "./primitives/FrequencySelector.react";
import NumericSelector, { type UnitConfig } from "./primitives/NumericSelector.react";
import Spinner from "./primitives/Spinner.react";
import { FrequencyLabel } from "./FrequencyLabel.react";

interface RadioTunerProps {
  capture: Capture;
  device: Device | undefined;
}

// Gain units configuration
const gainUnits: UnitConfig[] = [
  {
    name: "dB",
    multiplier: 1,
    decimals: 1,
    placeValues: [
      { label: "10", value: 10 },
      { label: "1", value: 1 },
      { label: "0.1", value: 0.1 },
    ],
  },
];

// Bandwidth units configuration
const bandwidthUnits: UnitConfig[] = [
  {
    name: "kHz",
    multiplier: 1_000,
    decimals: 0,
    placeValues: [
      { label: "1000", value: 1_000_000 },
      { label: "100", value: 100_000 },
      { label: "10", value: 10_000 },
      { label: "1", value: 1_000 },
    ],
  },
  {
    name: "MHz",
    multiplier: 1_000_000,
    decimals: 3,
    placeValues: [
      { label: "1", value: 1_000_000 },
      { label: "0.1", value: 100_000 },
      { label: "0.01", value: 10_000 },
      { label: "0.001", value: 1_000 },
    ],
  },
];

export const RadioTuner = ({ capture, device }: RadioTunerProps) => {
  // UI state
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Local state for immediate UI updates
  const [localFreq, setLocalFreq] = useState(capture.centerHz);
  const [localGain, setLocalGain] = useState(capture.gain ?? 0);
  const [localBandwidth, setLocalBandwidth] = useState(capture.bandwidth ?? 200000);
  const [localPpm, setLocalPpm] = useState(capture.ppm ?? 0);
  const [localSampleRate, setLocalSampleRate] = useState(capture.sampleRate);
  const [localAntenna, setLocalAntenna] = useState(capture.antenna ?? "");

  // Advanced settings state
  const [localDcOffsetAuto, setLocalDcOffsetAuto] = useState(capture.dcOffsetAuto ?? true);
  const [localIqBalanceAuto, setLocalIqBalanceAuto] = useState(capture.iqBalanceAuto ?? true);
  const [localStreamFormat, setLocalStreamFormat] = useState(capture.streamFormat ?? "");
  const [localElementGains, setLocalElementGains] = useState<Record<string, number>>(capture.elementGains ?? {});
  const [localDeviceSettings, setLocalDeviceSettings] = useState<Record<string, string>>(capture.deviceSettings ?? {});

  // Sync local state when capture updates from backend
  useEffect(() => {
    setLocalFreq(capture.centerHz);
    setLocalGain(capture.gain ?? 0);
    setLocalBandwidth(capture.bandwidth ?? 200000);
    setLocalPpm(capture.ppm ?? 0);
    setLocalSampleRate(capture.sampleRate);
    setLocalAntenna(capture.antenna ?? "");
    setLocalDcOffsetAuto(capture.dcOffsetAuto ?? true);
    setLocalIqBalanceAuto(capture.iqBalanceAuto ?? true);
    setLocalStreamFormat(capture.streamFormat ?? "");
    setLocalElementGains(capture.elementGains ?? {});
    setLocalDeviceSettings(capture.deviceSettings ?? {});
  }, [capture]);

  // Debounce values for API calls
  const debouncedFreq = useDebounce(localFreq, 100);
  const debouncedGain = useDebounce(localGain, 100);
  const debouncedBandwidth = useDebounce(localBandwidth, 100);
  const debouncedPpm = useDebounce(localPpm, 100);

  const updateMutation = useUpdateCapture();
  const startMutation = useStartCapture();
  const stopMutation = useStopCapture();

  // Update capture when debounced values change
  useEffect(() => {
    if (debouncedFreq !== capture.centerHz) {
      updateMutation.mutate({
        captureId: capture.id,
        request: { centerHz: debouncedFreq },
      });
    }
  }, [debouncedFreq]);

  useEffect(() => {
    const gainValue = debouncedGain === 0 ? undefined : debouncedGain;
    if (gainValue !== capture.gain) {
      updateMutation.mutate({
        captureId: capture.id,
        request: { gain: gainValue },
      });
    }
  }, [debouncedGain]);

  useEffect(() => {
    if (debouncedBandwidth !== capture.bandwidth) {
      updateMutation.mutate({
        captureId: capture.id,
        request: { bandwidth: debouncedBandwidth },
      });
    }
  }, [debouncedBandwidth]);

  useEffect(() => {
    if (debouncedPpm !== capture.ppm) {
      updateMutation.mutate({
        captureId: capture.id,
        request: { ppm: debouncedPpm },
      });
    }
  }, [debouncedPpm]);

  const handleSampleRateChange = (newRate: number) => {
    setLocalSampleRate(newRate);
    updateMutation.mutate({
      captureId: capture.id,
      request: { sampleRate: newRate },
    });
  };

  const handleAntennaChange = (newAntenna: string) => {
    setLocalAntenna(newAntenna);
    updateMutation.mutate({
      captureId: capture.id,
      request: { antenna: newAntenna },
    });
  };

  const handleDcOffsetAutoChange = (enabled: boolean) => {
    setLocalDcOffsetAuto(enabled);
    updateMutation.mutate({
      captureId: capture.id,
      request: { dcOffsetAuto: enabled },
    });
  };

  const handleIqBalanceAutoChange = (enabled: boolean) => {
    setLocalIqBalanceAuto(enabled);
    updateMutation.mutate({
      captureId: capture.id,
      request: { iqBalanceAuto: enabled },
    });
  };

  const handleStreamFormatChange = (format: string) => {
    setLocalStreamFormat(format);
    updateMutation.mutate({
      captureId: capture.id,
      request: { streamFormat: format || undefined },
    });
  };

  const handleStartStop = () => {
    if (capture.state === "running") {
      stopMutation.mutate(capture.id);
    } else {
      startMutation.mutate(capture.id);
    }
  };

  const isRunning = capture.state === "running";
  const isFailed = capture.state === "failed";
  const isUpdating = updateMutation.isPending;

  // Get device constraints
  const deviceFreqMin = device?.freqMinHz ?? 24_000_000;
  const deviceFreqMax = device?.freqMaxHz ?? 1_800_000_000;
  const gainMin = device?.gainMin ?? 0;
  const gainMax = device?.gainMax ?? 60;
  const bwMin = device?.bandwidthMin ?? 200_000;
  const bwMax = device?.bandwidthMax ?? 8_000_000;
  const ppmMin = device?.ppmMin ?? -100;
  const ppmMax = device?.ppmMax ?? 100;

  return (
    <div className="card shadow-sm">
      <div className="card-header bg-body-tertiary">
        <Flex justify="between" align="center">
          <Flex direction="column" gap={1}>
            <Flex align="center" gap={2}>
              <Radio size={20} />
              <h2 className="h5 mb-0">Radio Tuner</h2>
              {updateMutation.isPending && (
                <Flex align="center" gap={1}>
                  <Spinner size="sm" />
                  <span className="small text-muted">Updating...</span>
                </Flex>
              )}
            </Flex>
            <div className="small text-muted">
              <strong>Capture:</strong> {capture.id}
              {device && (
                <>
                  {" • "}
                  <strong>Device:</strong> {device.driver.toUpperCase()} - {device.label}
                </>
              )}
            </div>
          </Flex>
          <Flex align="center" gap={2}>
            {isUpdating && <Spinner size="sm" />}
            <span className={`badge bg-${isFailed ? "danger" : isRunning ? "success" : "secondary"}`}>
              {capture.state}
            </span>
          </Flex>
        </Flex>
      </div>

      <div className="card-body">
        <Flex direction="column" gap={4}>
          {/* Frequency Display */}
          <div className="text-center py-3 bg-primary bg-opacity-10 rounded">
            <div className="display-4 fw-bold text-primary">
              {formatFrequencyMHz(localFreq)} MHz
            </div>
            <div className="text-muted small">
              Center Frequency
              <FrequencyLabel frequencyHz={localFreq} />
            </div>
          </div>

          {/* Start/Stop Button */}
          <Button
            use={isRunning ? "danger" : "success"}
            size="lg"
            onClick={handleStartStop}
            disabled={startMutation.isPending || stopMutation.isPending}
          >
            {isRunning ? "Stop Capture" : "Start Capture"}
          </Button>

          {/* Error Message */}
          {isFailed && capture.errorMessage && (
            <div className="alert alert-danger mb-0">
              <strong>Error:</strong> {capture.errorMessage}
            </div>
          )}

          <hr className="my-2" />

          {/* Frequency Selector */}
          <FrequencySelector
            label="Frequency"
            value={localFreq}
            min={deviceFreqMin}
            max={deviceFreqMax}
            step={1000}
            onChange={setLocalFreq}
            disabled={!isRunning}
            info="The center frequency your SDR will tune to. All channels are offset from this frequency."
          />

          {/* Gain Selector */}
          <NumericSelector
            label="Gain"
            value={localGain}
            min={gainMin}
            max={gainMax}
            step={0.1}
            units={gainUnits}
            info="Signal amplification in decibels. Higher gain increases sensitivity but may introduce noise. Start around 20-30 dB and adjust for best signal-to-noise ratio."
            onChange={setLocalGain}
            disabled={!isRunning}
          />

          {/* Sample Rate Dropdown */}
          <Flex direction="column" gap={2}>
            <label className="form-label mb-0 fw-semibold">
              <Settings size={16} className="me-1" />
              Sample Rate
            </label>
            <select
              className="form-select"
              value={localSampleRate}
              onChange={(e) => handleSampleRateChange(parseInt(e.target.value))}
              disabled={!isRunning}
            >
              {(device?.sampleRates || []).map((rate) => (
                <option key={rate} value={rate}>
                  {formatSampleRate(rate)}
                </option>
              ))}
            </select>
            {isRunning && (
              <small className="text-warning">
                Warning: Changing sample rate will briefly interrupt the stream while the radio restarts
              </small>
            )}
            {!isRunning && (
              <small className="text-muted">
                Start capture to change sample rate
              </small>
            )}
          </Flex>

          {/* Bandwidth Selector */}
          <NumericSelector
            label="Bandwidth"
            value={localBandwidth}
            min={bwMin}
            max={bwMax}
            step={1000}
            units={bandwidthUnits}
            info="Filter bandwidth. Wider bandwidth allows more spectrum but may include unwanted signals. Match to your signal type: FM broadcast ~200 kHz, narrowband ~10-25 kHz."
            onChange={setLocalBandwidth}
            disabled={!isRunning}
          />

          {/* PPM Correction Slider */}
          <Slider
            label="PPM Correction"
            value={localPpm}
            min={ppmMin}
            max={ppmMax}
            step={0.1}
            coarseStep={1}
            unit="ppm"
            info="Corrects frequency offset in parts-per-million caused by crystal oscillator inaccuracy. If signals appear slightly off-frequency, adjust this. Most devices need 0-5 ppm correction."
            onChange={setLocalPpm}
            disabled={!isRunning}
          />

          {/* Antenna Selector */}
          {device?.antennas && device.antennas.length > 0 && (
            <Flex direction="column" gap={2}>
              <Flex direction="column" gap={0}>
                <label className="form-label mb-0 fw-semibold">Antenna</label>
                <small className="text-muted">
                  Select which antenna port to use. Different ports may have different characteristics (frequency range, impedance). Refer to your device manual for specifics.
                </small>
              </Flex>
              <select
                className="form-select"
                value={localAntenna}
                onChange={(e) => handleAntennaChange(e.target.value)}
                disabled={!isRunning}
              >
                {device.antennas.map((ant) => (
                  <option key={ant} value={ant}>
                    {ant}
                  </option>
                ))}
              </select>
              {isRunning && (
                <small className="text-warning">
                  Warning: Changing antenna will briefly interrupt the stream while the radio restarts
                </small>
              )}
              {!isRunning && (
                <small className="text-muted">
                  Start capture to change antenna
                </small>
              )}
            </Flex>
          )}

          <hr className="my-3" />

          {/* Advanced Settings Toggle */}
          <Button
            use="secondary"
            size="sm"
            onClick={() => setShowAdvanced(!showAdvanced)}
          >
            <Flex align="center" gap={1}>
              <Settings size={16} />
              <span>Advanced Settings</span>
              {showAdvanced ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
            </Flex>
          </Button>

          {/* Advanced Settings Panel */}
          {showAdvanced && (
            <Flex direction="column" gap={3} className="mt-3 p-3 bg-light rounded">
              {/* DC Offset Auto Correction */}
              <div className="form-check">
                <input
                  className="form-check-input"
                  type="checkbox"
                  id="dcOffsetAuto"
                  checked={localDcOffsetAuto}
                  onChange={(e) => handleDcOffsetAutoChange(e.target.checked)}
                  disabled={isRunning}
                />
                <label className="form-check-label" htmlFor="dcOffsetAuto">
                  DC Offset Auto-Correction
                </label>
                <div className="form-text">
                  Automatically remove DC bias from IQ samples
                </div>
              </div>

              {/* IQ Balance Auto Correction */}
              <div className="form-check">
                <input
                  className="form-check-input"
                  type="checkbox"
                  id="iqBalanceAuto"
                  checked={localIqBalanceAuto}
                  onChange={(e) => handleIqBalanceAutoChange(e.target.checked)}
                  disabled={isRunning}
                />
                <label className="form-check-label" htmlFor="iqBalanceAuto">
                  IQ Balance Auto-Correction
                </label>
                <div className="form-text">
                  Automatically correct IQ imbalance
                </div>
              </div>

              {/* Stream Format Selector */}
              <Flex direction="column" gap={2}>
                <label className="form-label mb-0 fw-semibold">Stream Format</label>
                <select
                  className="form-select form-select-sm"
                  value={localStreamFormat}
                  onChange={(e) => handleStreamFormatChange(e.target.value)}
                  disabled={isRunning}
                >
                  <option value="">Auto (CF32)</option>
                  <option value="CF32">CF32 (Complex Float32)</option>
                  <option value="CS16">CS16 (Complex Int16)</option>
                  <option value="CS8">CS8 (Complex Int8)</option>
                </select>
                <small className="text-muted">
                  Stream format affects bandwidth and precision
                </small>
              </Flex>

              {/* Element Gains */}
              {Object.keys(localElementGains).length > 0 && (
                <Flex direction="column" gap={2}>
                  <Flex direction="column" gap={0}>
                    <label className="form-label mb-0 fw-semibold">Element Gains</label>
                    <small className="text-muted">
                      Individual gain controls for specific RF stages in your SDR. LNA (Low Noise Amplifier), IF, etc. Adjust these for fine-tuned control over different signal paths.
                    </small>
                  </Flex>
                  {Object.entries(localElementGains).map(([key, value]) => (
                    <Flex key={key} direction="column" gap={1}>
                      <label className="form-label mb-0 small">{key}</label>
                      <input
                        type="number"
                        className="form-control form-control-sm"
                        value={value}
                        onChange={(e) =>
                          setLocalElementGains({
                            ...localElementGains,
                            [key]: parseFloat(e.target.value),
                          })
                        }
                        disabled={isRunning}
                        step="0.1"
                      />
                    </Flex>
                  ))}
                  <small className="text-muted">
                    Per-element gain control (LNA, VGA, TIA, etc.)
                  </small>
                </Flex>
              )}

              {/* Device Settings */}
              {Object.keys(localDeviceSettings).length > 0 && (
                <Flex direction="column" gap={2}>
                  <label className="form-label mb-0 fw-semibold">Device Settings</label>
                  {Object.entries(localDeviceSettings).map(([key, value]) => (
                    <Flex key={key} direction="column" gap={1}>
                      <label className="form-label mb-0 small">{key}</label>
                      <input
                        type="text"
                        className="form-control form-control-sm"
                        value={value}
                        onChange={(e) =>
                          setLocalDeviceSettings({
                            ...localDeviceSettings,
                            [key]: e.target.value,
                          })
                        }
                        disabled={isRunning}
                      />
                    </Flex>
                  ))}
                  <small className="text-muted">
                    Device-specific configuration settings
                  </small>
                </Flex>
              )}

              {!isRunning && (
                <small className="text-muted">
                  Start capture to apply advanced settings changes
                </small>
              )}
            </Flex>
          )}
        </Flex>
      </div>
    </div>
  );
};
