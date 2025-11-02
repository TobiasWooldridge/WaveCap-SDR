import { useState, useEffect } from "react";
import { Radio, Settings } from "lucide-react";
import type { Capture, Device } from "../types";
import { useUpdateCapture, useStartCapture, useStopCapture } from "../hooks/useCaptures";
import { useDebounce } from "../hooks/useDebounce";
import { formatFrequencyMHz, formatSampleRate } from "../utils/frequency";
import Button from "./primitives/Button.react";
import Flex from "./primitives/Flex.react";
import Slider from "./primitives/Slider.react";
import Spinner from "./primitives/Spinner.react";

interface RadioTunerProps {
  capture: Capture;
  device: Device | undefined;
}

export const RadioTuner = ({ capture, device }: RadioTunerProps) => {
  // Local state for immediate UI updates
  const [localFreq, setLocalFreq] = useState(capture.centerHz);
  const [localGain, setLocalGain] = useState(capture.gain ?? 0);
  const [localBandwidth, setLocalBandwidth] = useState(capture.bandwidth ?? 200000);
  const [localPpm, setLocalPpm] = useState(capture.ppm ?? 0);
  const [localSampleRate, setLocalSampleRate] = useState(capture.sampleRate);
  const [localAntenna, setLocalAntenna] = useState(capture.antenna ?? "");

  // Sync local state when capture updates from backend
  useEffect(() => {
    setLocalFreq(capture.centerHz);
    setLocalGain(capture.gain ?? 0);
    setLocalBandwidth(capture.bandwidth ?? 200000);
    setLocalPpm(capture.ppm ?? 0);
    setLocalSampleRate(capture.sampleRate);
    setLocalAntenna(capture.antenna ?? "");
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
  const freqMin = device?.freqMinHz ?? 24_000_000;
  const freqMax = device?.freqMaxHz ?? 1_800_000_000;
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
            <div className="text-muted small">Center Frequency</div>
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

          {/* Frequency Slider */}
          <Slider
            label="Frequency"
            value={localFreq}
            min={freqMin}
            max={freqMax}
            step={10000}
            unit="MHz"
            formatValue={(hz) => formatFrequencyMHz(hz)}
            onChange={setLocalFreq}
            disabled={!isRunning}
          />

          {/* Gain Slider */}
          <Slider
            label="Gain"
            value={localGain}
            min={gainMin}
            max={gainMax}
            step={0.1}
            unit="dB"
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
              disabled={isRunning}
            >
              {(device?.sampleRates || []).map((rate) => (
                <option key={rate} value={rate}>
                  {formatSampleRate(rate)}
                </option>
              ))}
            </select>
            {isRunning && (
              <small className="text-warning">
                Stop capture to change sample rate
              </small>
            )}
          </Flex>

          {/* Bandwidth Slider */}
          <Slider
            label="Bandwidth"
            value={localBandwidth}
            min={bwMin}
            max={bwMax}
            step={10000}
            unit="Hz"
            formatValue={(hz) => `${(hz / 1000).toFixed(0)} k`}
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
            unit="ppm"
            onChange={setLocalPpm}
            disabled={!isRunning}
          />

          {/* Antenna Selector */}
          {device?.antennas && device.antennas.length > 0 && (
            <Flex direction="column" gap={2}>
              <label className="form-label mb-0 fw-semibold">Antenna</label>
              <select
                className="form-select"
                value={localAntenna}
                onChange={(e) => handleAntennaChange(e.target.value)}
                disabled={isRunning}
              >
                {device.antennas.map((ant) => (
                  <option key={ant} value={ant}>
                    {ant}
                  </option>
                ))}
              </select>
              {isRunning && (
                <small className="text-warning">
                  Stop capture to change antenna
                </small>
              )}
            </Flex>
          )}
        </Flex>
      </div>
    </div>
  );
};
