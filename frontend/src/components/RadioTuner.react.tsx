import { useState, useEffect, useCallback } from "react";
import { Radio, Settings, ChevronDown, ChevronUp, Cpu, RotateCcw, RefreshCw, Check } from "lucide-react";
import type { Capture, Device } from "../types";
import { useUpdateCapture, useStartCapture, useStopCapture, useRestartCapture } from "../hooks/useCaptures";
import { useDevices, useRestartSDRplayService } from "../hooks/useDevices";
import { useChannels, useCreateChannel } from "../hooks/useChannels";
import { useMemoryBanks } from "../hooks/useMemoryBanks";
import { useDebounce } from "../hooks/useDebounce";
import { useToast } from "../hooks/useToast";
import { formatFrequencyMHz, formatSampleRate } from "../utils/frequency";
import { getDeviceDisplayName, getDeviceNameFromId } from "../utils/device";
import Button from "./primitives/Button.react";
import Flex from "./primitives/Flex.react";
import Slider from "./primitives/Slider.react";
import FrequencySelector from "./primitives/FrequencySelector.react";
import NumericSelector, { type UnitConfig } from "./primitives/NumericSelector.react";
import Spinner from "./primitives/Spinner.react";
import { BookmarkManager } from "./BookmarkManager.react";
import ScannerControl from "./ScannerControl.react";
import { ErrorStatusBar } from "./ErrorStatusBar.react";

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
  // Fetch all available devices
  const { data: devices } = useDevices();

  // UI state
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Local state for immediate UI updates
  const [localDeviceId, setLocalDeviceId] = useState(capture.deviceId);
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
    setLocalDeviceId(capture.deviceId);
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
  const restartMutation = useRestartCapture();
  const restartServiceMutation = useRestartSDRplayService();
  const toast = useToast();

  // Show error toast when update fails
  useEffect(() => {
    if (updateMutation.isError && updateMutation.error) {
      toast.error(`Update failed: ${updateMutation.error.message}`);
    }
  }, [updateMutation.isError, updateMutation.error]);

  // Check if this is an SDRplay device
  const isSDRplayDevice = capture.deviceId?.toLowerCase().includes("sdrplay");

  // Confirm state for destructive buttons (touch-friendly two-tap pattern)
  const [confirmingAction, setConfirmingAction] = useState<"restart" | "service" | null>(null);

  // Reset confirm state after timeout
  useEffect(() => {
    if (confirmingAction) {
      const timer = setTimeout(() => setConfirmingAction(null), 3000);
      return () => clearTimeout(timer);
    }
  }, [confirmingAction]);

  // Handle confirm-required button clicks
  const handleConfirmableAction = useCallback((
    action: "restart" | "service",
    execute: () => void
  ) => {
    if (confirmingAction === action) {
      // Second tap - execute the action
      execute();
      setConfirmingAction(null);
    } else {
      // First tap - enter confirm mode
      setConfirmingAction(action);
    }
  }, [confirmingAction]);

  // Get channels for the capture
  const { data: channels } = useChannels(capture.id);

  // Get memory banks hook
  const { getMemoryBank } = useMemoryBanks();

  // Create channel hook for loading memory banks
  const createChannel = useCreateChannel();

  // Handler to load a memory bank
  const handleLoadMemoryBank = (bankId: string) => {
    const bank = getMemoryBank(bankId);
    if (!bank) return;

    // Update capture configuration
    updateMutation.mutate({
      captureId: capture.id,
      request: {
        centerHz: bank.captureConfig.centerHz,
        sampleRate: bank.captureConfig.sampleRate,
        gain: bank.captureConfig.gain ?? undefined,
        bandwidth: bank.captureConfig.bandwidth ?? undefined,
        ppm: bank.captureConfig.ppm ?? undefined,
        antenna: bank.captureConfig.antenna ?? undefined,
      },
    });

    // Recreate channels (note: this is simplified, might need to clear existing channels first)
    bank.channels.forEach((channelConfig) => {
      createChannel.mutate({
        captureId: capture.id,
        request: {
          mode: channelConfig.mode,
          offsetHz: channelConfig.offsetHz,
          audioRate: channelConfig.audioRate,
          squelchDb: channelConfig.squelchDb,
          name: channelConfig.name,
        },
      });
    });
  };

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

  const handleDeviceChange = (deviceId: string) => {
    const newDevice = devices?.find((d) => d.id === deviceId);
    if (!newDevice) return;

    // Update local state immediately for responsive UI
    setLocalDeviceId(deviceId);
    setLocalSampleRate(newDevice.sampleRates[0]);

    updateMutation.mutate({
      captureId: capture.id,
      request: {
        deviceId: deviceId,
        // Reset sample rate to first available for new device
        sampleRate: newDevice.sampleRates[0],
      },
    });
  };

  const isRunning = capture.state === "running";
  const isStarting = capture.state === "starting";
  const isStopping = capture.state === "stopping";
  const isFailed = capture.state === "failed";
  const isError = capture.state === "error";
  const hasError = isFailed || isError;
  const isTransitioning = isStarting || isStopping;

  // Track which settings are pending update
  const isFreqPending = localFreq !== capture.centerHz || debouncedFreq !== capture.centerHz;
  const isGainPending = localGain !== (capture.gain ?? 0) || debouncedGain !== (capture.gain ?? 0);
  const isBandwidthPending = localBandwidth !== (capture.bandwidth ?? 200000) || debouncedBandwidth !== (capture.bandwidth ?? 200000);
  const isPpmPending = localPpm !== (capture.ppm ?? 0) || debouncedPpm !== (capture.ppm ?? 0);
  const isSampleRatePending = localSampleRate !== capture.sampleRate;
  const isAntennaPending = localAntenna !== (capture.antenna ?? "");

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
    <Flex direction="column" gap={2}>
      {/* Compact Inline Header */}
      <div className="d-flex align-items-center gap-2 p-2 bg-light rounded border">
        <Radio size={16} className="flex-shrink-0" />
        <span className="fw-semibold small text-truncate" style={{ maxWidth: "200px" }}>
          {device ? getDeviceDisplayName(device) : getDeviceNameFromId(capture.deviceId)}
        </span>
        <span className="small text-muted">
          {formatFrequencyMHz(localFreq)} MHz
        </span>
        {updateMutation.isPending && (
          <Spinner size="sm" />
        )}
        <span className={`badge ms-auto ${
          hasError ? "bg-danger" :
          isRunning ? "bg-success" :
          isTransitioning ? "bg-warning text-dark" :
          "bg-secondary"
        }`} style={{ fontSize: "0.7rem" }}>
          {capture.state.toUpperCase()}
        </span>
      </div>

      {/* Device & Control Card */}
      <div className="card shadow-sm">
        <div className="card-header bg-body-tertiary py-1 px-2">
          <Flex align="center" gap={1}>
            <Cpu size={14} />
            <small className="fw-semibold mb-0">Device & Control</small>
          </Flex>
        </div>
        <div className="card-body" style={{ padding: "0.75rem" }}>
          <Flex direction="column" gap={2}>
            {/* Row 1: Device Selector */}
            <div>
              <label className="form-label mb-1 small fw-semibold">Radio Device</label>
              <select
                className="form-select form-select-sm"
                value={localDeviceId}
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
                  Stop to change
                </small>
              )}
            </div>

            {/* Row 2: Control Buttons + Bookmarks */}
            <div className="d-flex gap-2 align-items-center flex-wrap">
              {/* Start/Stop + Restart + Service Button Group */}
              <div className="btn-group" role="group">
                {/* Start/Stop */}
                <Button
                  use={isRunning || isStopping ? "danger" : isStarting ? "warning" : "success"}
                  size="sm"
                  onClick={() => {
                    if (isRunning) {
                      stopMutation.mutate(capture.id);
                    } else {
                      startMutation.mutate(capture.id);
                    }
                  }}
                  disabled={startMutation.isPending || stopMutation.isPending || isTransitioning || restartMutation.isPending}
                >
                  {isStarting ? "Starting..." : isStopping ? "Stopping..." : isRunning ? "Stop" : "Start"}
                </Button>
                {/* Restart - always requires confirm */}
                <Button
                  use={confirmingAction === "restart" ? "warning" : "secondary"}
                  size="sm"
                  onClick={() => handleConfirmableAction("restart", () => restartMutation.mutate(capture.id))}
                  disabled={restartMutation.isPending || isTransitioning || stopMutation.isPending || startMutation.isPending || restartServiceMutation.isPending}
                  title={confirmingAction === "restart" ? "Tap again to confirm restart" : "Restart capture (stop then start)"}
                >
                  {restartMutation.isPending ? <Spinner size="sm" /> :
                   confirmingAction === "restart" ? <Check size={14} /> : <RotateCcw size={14} />}
                </Button>
                {/* Service restart - always requires confirm */}
                {isSDRplayDevice && (
                  <Button
                    use={confirmingAction === "service" ? "warning" : "danger"}
                    size="sm"
                    onClick={() => handleConfirmableAction("service", () => restartServiceMutation.mutate())}
                    disabled={restartMutation.isPending || restartServiceMutation.isPending}
                    title={confirmingAction === "service" ? "Tap again to confirm service restart" : "Restart SDRplay API service (fixes stuck devices)"}
                  >
                    {restartServiceMutation.isPending ? <Spinner size="sm" /> :
                     confirmingAction === "service" ? <Check size={14} /> : <RefreshCw size={14} />}
                  </Button>
                )}
              </div>

              {/* Bookmark Manager */}
              <BookmarkManager
                currentFrequency={localFreq}
                onTuneToFrequency={(freq) => setLocalFreq(freq)}
                currentCapture={capture}
                currentChannels={channels}
                onLoadMemoryBank={handleLoadMemoryBank}
              />
            </div>

            {/* Error Message */}
            {hasError && (
              <div className="alert alert-danger mb-0 py-1 px-2">
                <div className="d-flex align-items-center justify-content-between">
                  <small>
                    <strong>{isError ? "Device Error:" : "Error:"}</strong>{" "}
                    {capture.errorMessage || (isError ? "No IQ samples received - device may be stuck" : "Unknown error")}
                  </small>
                  <div className="d-flex gap-1">
                    <Button
                      use="warning"
                      size="sm"
                      onClick={() => restartMutation.mutate(capture.id)}
                      disabled={restartMutation.isPending || restartServiceMutation.isPending}
                    >
                      {restartMutation.isPending ? "Restarting..." : "Restart Capture"}
                    </Button>
                    {isSDRplayDevice && (
                      <Button
                        use="danger"
                        size="sm"
                        onClick={() => restartServiceMutation.mutate()}
                        disabled={restartMutation.isPending || restartServiceMutation.isPending}
                        title="Restart the SDRplay API service if device is stuck"
                      >
                        {restartServiceMutation.isPending ? "Restarting..." : "Restart Service"}
                      </Button>
                    )}
                  </div>
                </div>
                {restartServiceMutation.isError && (
                  <small className="text-danger d-block mt-1">
                    Service restart failed: {restartServiceMutation.error?.message}
                  </small>
                )}
              </div>
            )}

            {/* Real-time Error Indicators (IQ overflows, audio drops, retries) */}
            {isRunning && <ErrorStatusBar captureId={capture.id} capture={capture} />}
          </Flex>
        </div>
      </div>

      {/* Frequency Settings Card */}
      <div className="card shadow-sm">
        <div className="card-header bg-body-tertiary py-1 px-2">
          <Flex align="center" gap={1}>
            <Settings size={14} />
            <small className="fw-semibold mb-0">Frequency Settings</small>
          </Flex>
        </div>
        <div className="card-body" style={{ padding: "0.75rem" }}>
          {/* 2-Column Grid Layout for Controls */}
          <div className="row g-2">
            {/* Frequency Selector */}
            <div className="col-12">
              <FrequencySelector
                label={isFreqPending ? "Frequency (updating...)" : "Frequency"}
                value={localFreq}
                min={deviceFreqMin}
                max={deviceFreqMax}
                step={1000}
                onChange={setLocalFreq}
                info="The center frequency your SDR will tune to. All channels are offset from this frequency."
              />
            </div>

            {/* Gain Selector */}
            <div className="col-12">
              <NumericSelector
                label={isGainPending ? "Gain (updating...)" : "Gain"}
                value={localGain}
                min={gainMin}
                max={gainMax}
                step={0.1}
                units={gainUnits}
                info="Signal amplification in decibels. Higher gain increases sensitivity but may introduce noise. Start around 20-30 dB and adjust for best signal-to-noise ratio."
                onChange={setLocalGain}
              />
              {localGain > 45 && (
                <div className="alert alert-warning py-1 px-2 mt-2 mb-0" style={{ fontSize: "0.8rem" }}>
                  <strong>Warning:</strong> High gain ({localGain.toFixed(1)} dB) may cause signal clipping and distortion.
                  Consider reducing gain to 20-40 dB for optimal performance.
                </div>
              )}
            </div>

            {/* Bandwidth Selector */}
            <div className="col-12">
              <NumericSelector
                label={isBandwidthPending ? "Bandwidth (updating...)" : "Bandwidth"}
                value={localBandwidth}
                min={bwMin}
                max={bwMax}
                step={1000}
                units={bandwidthUnits}
                info="Filter bandwidth. Wider bandwidth allows more spectrum but may include unwanted signals. Match to your signal type: FM broadcast ~200 kHz, narrowband ~10-25 kHz."
                onChange={setLocalBandwidth}
              />
              {localBandwidth > localSampleRate && (
                <div className="alert alert-warning py-1 px-2 mt-2 mb-0" style={{ fontSize: "0.8rem" }}>
                  <strong>Warning:</strong> Bandwidth ({(localBandwidth / 1e6).toFixed(2)} MHz) is higher than sample rate ({(localSampleRate / 1e6).toFixed(2)} MHz).
                  Bandwidth should be ≤ sample rate to avoid aliasing.
                </div>
              )}
              {localBandwidth < 150_000 && localSampleRate >= 200_000 && (
                <div className="alert alert-info py-1 px-2 mt-2 mb-0" style={{ fontSize: "0.8rem" }}>
                  <strong>Note:</strong> Bandwidth ({(localBandwidth / 1e3).toFixed(0)} kHz) may be too narrow for FM broadcast reception.
                  Recommended: 150-220 kHz for WBFM, 10-25 kHz for NBFM, 10 kHz for AM.
                </div>
              )}
            </div>

            {/* Sample Rate Dropdown */}
            <div className="col-12">
              <Flex direction="column" gap={2}>
                <label className="form-label mb-0 fw-semibold">
                  <Settings size={16} className="me-1" />
                  {isSampleRatePending ? "Sample Rate (updating...)" : "Sample Rate"}
                </label>
                <select
                  className="form-select"
                  value={localSampleRate}
                  onChange={(e) => handleSampleRateChange(parseInt(e.target.value))}
                >
                  {(device?.sampleRates || []).map((rate) => (
                    <option key={rate} value={rate}>
                      {formatSampleRate(rate)}
                    </option>
                  ))}
                </select>
                {isRunning && (
                  <small className="text-warning">
                    Changing sample rate will briefly interrupt the stream
                  </small>
                )}
                {device?.sampleRates && device.sampleRates.length > 0 && !device.sampleRates.includes(localSampleRate) && (
                  <div className="alert alert-warning py-1 px-2 mt-2 mb-0" style={{ fontSize: "0.8rem" }}>
                    <strong>Warning:</strong> Current sample rate ({formatSampleRate(localSampleRate)}) is not supported by this device.
                    Select a valid rate from the dropdown above.
                  </div>
                )}
                {localSampleRate < 200_000 && (
                  <div className="alert alert-info py-1 px-2 mt-2 mb-0" style={{ fontSize: "0.8rem" }}>
                    <strong>Note:</strong> Sample rate ({(localSampleRate / 1e3).toFixed(0)} kHz) is below 200 kHz.
                    FM broadcast reception requires ≥200 kHz for optimal quality. Consider increasing sample rate if tuning to FM stations.
                  </div>
                )}
              </Flex>
            </div>

            {/* PPM Correction Slider */}
            <div className="col-12">
              <Slider
                label={isPpmPending ? "PPM Correction (updating...)" : "PPM Correction"}
                value={localPpm}
                min={ppmMin}
                max={ppmMax}
                step={0.1}
                coarseStep={1}
                unit="ppm"
                info="Corrects frequency offset in parts-per-million caused by crystal oscillator inaccuracy. If signals appear slightly off-frequency, adjust this. Most devices need 0-5 ppm correction."
                onChange={setLocalPpm}
              />
            </div>

            {/* Antenna Selector */}
            {device?.antennas && device.antennas.length > 0 && (
              <div className="col-12">
                <Flex direction="column" gap={2}>
                  <label className="form-label mb-0 fw-semibold">
                    {isAntennaPending ? "Antenna (updating...)" : "Antenna"}
                  </label>
                  <select
                    className="form-select"
                    value={localAntenna}
                    onChange={(e) => handleAntennaChange(e.target.value)}
                  >
                    {device.antennas.map((ant) => (
                      <option key={ant} value={ant}>
                        {ant}
                      </option>
                    ))}
                  </select>
                  {isRunning && (
                    <small className="text-warning">
                      Changing antenna will briefly interrupt the stream
                    </small>
                  )}
                </Flex>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Advanced Settings Card */}
      <div className="card shadow-sm">
        <div className="card-header bg-body-tertiary py-1 px-2" style={{ cursor: "pointer" }} onClick={() => setShowAdvanced(!showAdvanced)}>
          <Flex align="center" gap={1}>
            <Settings size={14} />
            <small className="fw-semibold mb-0">Advanced Settings</small>
            <div className="ms-auto">
              {showAdvanced ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            </div>
          </Flex>
        </div>
        {showAdvanced && (
          <div className="card-body" style={{ padding: "0.75rem" }}>
            {/* Advanced Settings Panel */}
            <Flex direction="column" gap={2}>
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
          </div>
        )}

        {/* Scanner Control */}
        <ScannerControl captureId={capture.id} />
      </div>
    </Flex>
  );
};
