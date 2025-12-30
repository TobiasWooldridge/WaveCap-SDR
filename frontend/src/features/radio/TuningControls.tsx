import { Lock } from "lucide-react";
import type { Capture, Device } from "../../types";
import type { TrunkingSystem } from "../../types/trunking";
import { useDebouncedMutation } from "../../hooks/useDebouncedMutation";
import { useUpdateCapture } from "../../hooks/useCaptures";
import { formatSampleRate, formatBandwidth } from "../../utils/frequency";
import { FrequencyDisplay } from "../../components/primitives/FrequencyDisplay.react";
import FrequencySelector from "../../components/primitives/FrequencySelector.react";
import NumericSelector, { type UnitConfig } from "../../components/primitives/NumericSelector.react";
import Slider from "../../components/primitives/Slider.react";
import Flex from "../../components/primitives/Flex.react";
import { SimpleAccordion } from "../../components/primitives/Accordion.react";

interface TuningControlsProps {
  capture: Capture;
  device: Device | undefined;
  /** Active trunking system that may be managing frequency */
  trunkingSystem?: TrunkingSystem | null;
}

const GAIN_UNITS: UnitConfig[] = [
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

const BANDWIDTH_UNITS: UnitConfig[] = [
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

// Content-only version for use in accordions
export function TuningControlsContent({ capture, device }: TuningControlsProps) {
  const updateCapture = useUpdateCapture();

  // Use debounced mutations for all tuning parameters
  const [freq, setFreq, freqPending] = useDebouncedMutation(
    capture.centerHz,
    (value) => updateCapture.mutate({ captureId: capture.id, request: { centerHz: value } }),
    { delay: 100 }
  );

  const [gain, setGain, gainPending] = useDebouncedMutation(
    capture.gain ?? 0,
    (value) => updateCapture.mutate({ captureId: capture.id, request: { gain: value === 0 ? undefined : value } }),
    { delay: 100 }
  );

  const [bandwidth, setBandwidth, bandwidthPending] = useDebouncedMutation(
    capture.bandwidth ?? 200000,
    (value) => updateCapture.mutate({ captureId: capture.id, request: { bandwidth: value } }),
    { delay: 100 }
  );

  const [ppm, setPpm, ppmPending] = useDebouncedMutation(
    capture.ppm ?? 0,
    (value) => updateCapture.mutate({ captureId: capture.id, request: { ppm: value } }),
    { delay: 100 }
  );

  const [sampleRate, setSampleRate] = useDebouncedMutation(
    capture.sampleRate,
    (value) => updateCapture.mutate({ captureId: capture.id, request: { sampleRate: value } }),
    { delay: 0 } // Immediate for dropdown
  );

  const [antenna, setAntenna] = useDebouncedMutation(
    capture.antenna ?? "",
    (value) => updateCapture.mutate({ captureId: capture.id, request: { antenna: value } }),
    { delay: 0 } // Immediate for dropdown
  );

  // Device constraints
  const freqMin = device?.freqMinHz ?? 24_000_000;
  const freqMax = device?.freqMaxHz ?? 1_800_000_000;
  const gainMin = device?.gainMin ?? 0;
  const gainMax = device?.gainMax ?? 60;
  const bwMin = device?.bandwidthMin ?? 200_000;
  const bwMax = device?.bandwidthMax ?? 8_000_000;
  const ppmMin = device?.ppmMin ?? -100;
  const ppmMax = device?.ppmMax ?? 100;

  const isRunning = capture.state === "running";

  return (
    <div className="row g-2">
      {/* Frequency Selector */}
      <div className="col-12">
        <FrequencySelector
          label={freqPending ? "Frequency (updating...)" : "Frequency"}
          value={freq}
          min={freqMin}
          max={freqMax}
          step={1000}
          onChange={setFreq}
          info="The center frequency your SDR will tune to. All channels are offset from this frequency."
        />
      </div>

      {/* Gain Selector */}
      <div className="col-12">
        <NumericSelector
          label={gainPending ? "Gain (updating...)" : "Gain"}
          value={gain}
          min={gainMin}
          max={gainMax}
          step={0.1}
          units={GAIN_UNITS}
          info="Signal amplification in decibels. Start around 20-30 dB."
          onChange={setGain}
        />
        {gain > 45 && (
          <div className="alert alert-warning py-1 px-2 mt-2 mb-0" style={{ fontSize: "0.8rem" }}>
            <strong>Warning:</strong> High gain ({gain.toFixed(1)} dB) may cause clipping.
          </div>
        )}
      </div>

      {/* Bandwidth Selector */}
      <div className="col-12">
        <NumericSelector
          label={bandwidthPending ? "Bandwidth (updating...)" : "Bandwidth"}
          value={bandwidth}
          min={bwMin}
          max={bwMax}
          step={1000}
          units={BANDWIDTH_UNITS}
          info="Filter bandwidth. FM broadcast ~200 kHz, narrowband ~10-25 kHz."
          onChange={setBandwidth}
        />
        {bandwidth > sampleRate && (
          <div className="alert alert-warning py-1 px-2 mt-2 mb-0" style={{ fontSize: "0.8rem" }}>
            <strong>Warning:</strong> Bandwidth exceeds sample rate. May cause aliasing.
          </div>
        )}
      </div>

      {/* Sample Rate */}
      <div className="col-12">
        <Flex direction="column" gap={1}>
          <label className="form-label mb-0 small fw-semibold">Sample Rate</label>
          <select
            className="form-select form-select-sm"
            value={sampleRate}
            onChange={(e) => setSampleRate(parseInt(e.target.value))}
          >
            {(device?.sampleRates || []).map((rate) => (
              <option key={rate} value={rate}>
                {formatSampleRate(rate)}
              </option>
            ))}
          </select>
          {isRunning && (
            <small className="text-warning" style={{ fontSize: "0.7rem" }}>
              Changing sample rate will briefly interrupt the stream
            </small>
          )}
        </Flex>
      </div>

      {/* PPM Correction */}
      <div className="col-12">
        <Slider
          label={ppmPending ? "PPM Correction (updating...)" : "PPM Correction"}
          value={ppm}
          min={ppmMin}
          max={ppmMax}
          step={0.1}
          coarseStep={1}
          unit="ppm"
          info="Corrects frequency offset. Most devices need 0-5 ppm."
          onChange={setPpm}
        />
      </div>

      {/* Antenna */}
      {device?.antennas && device.antennas.length > 0 && (
        <div className="col-12">
          <Flex direction="column" gap={1}>
            <label className="form-label mb-0 small fw-semibold">Antenna</label>
            <select
              className="form-select form-select-sm"
              value={antenna}
              onChange={(e) => setAntenna(e.target.value)}
            >
              {device.antennas.map((ant) => (
                <option key={ant} value={ant}>
                  {ant}
                </option>
              ))}
            </select>
          </Flex>
        </div>
      )}
    </div>
  );
}

// Individual accordions for each tuning parameter
export function TuningAccordions({ capture, device, trunkingSystem }: TuningControlsProps) {
  const updateCapture = useUpdateCapture();

  // Check if trunking is actively managing the frequency
  const isTrunkingManaged = trunkingSystem &&
    trunkingSystem.state !== "stopped" &&
    trunkingSystem.state !== "failed";

  // Debounced mutations
  const [freq, setFreq, freqPending] = useDebouncedMutation(
    capture.centerHz,
    (value) => updateCapture.mutate({ captureId: capture.id, request: { centerHz: value } }),
    { delay: 100 }
  );

  const [gain, setGain, gainPending] = useDebouncedMutation(
    capture.gain ?? 0,
    (value) => updateCapture.mutate({ captureId: capture.id, request: { gain: value === 0 ? undefined : value } }),
    { delay: 100 }
  );

  const [bandwidth, setBandwidth, bandwidthPending] = useDebouncedMutation(
    capture.bandwidth ?? 200000,
    (value) => updateCapture.mutate({ captureId: capture.id, request: { bandwidth: value } }),
    { delay: 100 }
  );

  const [ppm, setPpm, ppmPending] = useDebouncedMutation(
    capture.ppm ?? 0,
    (value) => updateCapture.mutate({ captureId: capture.id, request: { ppm: value } }),
    { delay: 100 }
  );

  const [sampleRate, setSampleRate] = useDebouncedMutation(
    capture.sampleRate,
    (value) => updateCapture.mutate({ captureId: capture.id, request: { sampleRate: value } }),
    { delay: 0 }
  );

  const [antenna, setAntenna] = useDebouncedMutation(
    capture.antenna ?? "",
    (value) => updateCapture.mutate({ captureId: capture.id, request: { antenna: value } }),
    { delay: 0 }
  );

  // Device constraints
  const freqMin = device?.freqMinHz ?? 24_000_000;
  const freqMax = device?.freqMaxHz ?? 1_800_000_000;
  const gainMin = device?.gainMin ?? 0;
  const gainMax = device?.gainMax ?? 60;
  const bwMin = device?.bandwidthMin ?? 200_000;
  const bwMax = device?.bandwidthMax ?? 8_000_000;
  const ppmMin = device?.ppmMin ?? -100;
  const ppmMax = device?.ppmMax ?? 100;

  const isRunning = capture.state === "running";

  return (
    <Flex direction="column" gap={1}>
      {/* Frequency - expanded by default */}
      <SimpleAccordion
        defaultOpen
        header={
          <span className="small">
            <span className="fw-semibold">Frequency:</span>{" "}
            <span className={freqPending ? "text-warning" : ""}>
              <FrequencyDisplay frequencyHz={freq} decimals={4} />
            </span>
            {isTrunkingManaged && (
              <span className="ms-1 text-info">
                <Lock size={12} className="me-1" />
                <span style={{ fontSize: "0.7rem" }}>Trunking</span>
              </span>
            )}
          </span>
        }
      >
        {isTrunkingManaged ? (
          <div className="alert alert-info py-2 mb-0" style={{ fontSize: "0.8rem" }}>
            <div className="d-flex align-items-center gap-2">
              <Lock size={16} />
              <div>
                <strong>Managed by Trunking System</strong>
                <div className="text-muted" style={{ fontSize: "0.75rem" }}>
                  {trunkingSystem?.name} is controlling the SDR frequency.
                  Stop the trunking system to manually tune.
                </div>
              </div>
            </div>
          </div>
        ) : (
          <FrequencySelector
            label="Center Frequency"
            value={freq}
            min={freqMin}
            max={freqMax}
            step={1000}
            onChange={setFreq}
            info="The center frequency your SDR will tune to. All channels are offset from this frequency."
          />
        )}
      </SimpleAccordion>

      {/* Gain */}
      <SimpleAccordion
        header={
          <span className="small">
            <span className="fw-semibold">Gain:</span>{" "}
            <span className={gainPending ? "text-warning" : ""}>
              {gain === 0 ? "auto" : `${gain.toFixed(1)} dB`}
            </span>
            {gain > 45 && <span className="text-warning ms-1">(high)</span>}
          </span>
        }
      >
        <NumericSelector
          label="Gain"
          value={gain}
          min={gainMin}
          max={gainMax}
          step={0.1}
          units={GAIN_UNITS}
          info="Signal amplification in decibels. Start around 20-30 dB."
          onChange={setGain}
        />
        {gain > 45 && (
          <div className="alert alert-warning py-1 px-2 mt-2 mb-0" style={{ fontSize: "0.8rem" }}>
            <strong>Warning:</strong> High gain may cause clipping.
          </div>
        )}
      </SimpleAccordion>

      {/* Bandwidth */}
      <SimpleAccordion
        header={
          <span className="small">
            <span className="fw-semibold">Bandwidth:</span>{" "}
            <span className={bandwidthPending ? "text-warning" : ""}>
              {formatBandwidth(bandwidth)}
            </span>
            {bandwidth > sampleRate && <span className="text-warning ms-1">(exceeds rate)</span>}
          </span>
        }
      >
        <NumericSelector
          label="Bandwidth"
          value={bandwidth}
          min={bwMin}
          max={bwMax}
          step={1000}
          units={BANDWIDTH_UNITS}
          info="Filter bandwidth. FM broadcast ~200 kHz, narrowband ~10-25 kHz."
          onChange={setBandwidth}
        />
        {bandwidth > sampleRate && (
          <div className="alert alert-warning py-1 px-2 mt-2 mb-0" style={{ fontSize: "0.8rem" }}>
            <strong>Warning:</strong> Bandwidth exceeds sample rate. May cause aliasing.
          </div>
        )}
      </SimpleAccordion>

      {/* Sample Rate */}
      <SimpleAccordion
        header={
          <span className="small">
            <span className="fw-semibold">Sample Rate:</span> {formatSampleRate(sampleRate)}
          </span>
        }
      >
        <Flex direction="column" gap={1}>
          <label className="form-label mb-0 small fw-semibold">Sample Rate</label>
          <select
            className="form-select form-select-sm"
            value={sampleRate}
            onChange={(e) => setSampleRate(parseInt(e.target.value))}
          >
            {(device?.sampleRates || []).map((rate) => (
              <option key={rate} value={rate}>
                {formatSampleRate(rate)}
              </option>
            ))}
          </select>
          {isRunning && (
            <small className="text-warning" style={{ fontSize: "0.7rem" }}>
              Changing sample rate will briefly interrupt the stream
            </small>
          )}
        </Flex>
      </SimpleAccordion>

      {/* PPM Correction */}
      <SimpleAccordion
        header={
          <span className="small">
            <span className="fw-semibold">PPM:</span>{" "}
            <span className={ppmPending ? "text-warning" : ""}>{ppm.toFixed(1)} ppm</span>
          </span>
        }
      >
        <Slider
          label="PPM Correction"
          value={ppm}
          min={ppmMin}
          max={ppmMax}
          step={0.1}
          coarseStep={1}
          unit="ppm"
          info="Corrects frequency offset. Most devices need 0-5 ppm."
          onChange={setPpm}
        />
      </SimpleAccordion>

      {/* Antenna */}
      {device?.antennas && device.antennas.length > 1 && (
        <SimpleAccordion
          header={
            <span className="small">
              <span className="fw-semibold">Antenna:</span> {antenna || device.antennas[0]}
            </span>
          }
        >
          <Flex direction="column" gap={1}>
            <label className="form-label mb-0 small fw-semibold">Antenna</label>
            <select
              className="form-select form-select-sm"
              value={antenna}
              onChange={(e) => setAntenna(e.target.value)}
            >
              {device.antennas.map((ant) => (
                <option key={ant} value={ant}>
                  {ant}
                </option>
              ))}
            </select>
          </Flex>
        </SimpleAccordion>
      )}
    </Flex>
  );
}

// Legacy wrapper for backwards compatibility
export function TuningControls({ capture, device }: TuningControlsProps) {
  return (
    <div className="card shadow-sm">
      <div className="card-header bg-body-tertiary py-1 px-2">
        <Flex align="center" gap={1}>
          <small className="fw-semibold mb-0">Tuning</small>
        </Flex>
      </div>
      <div className="card-body" style={{ padding: "0.75rem" }}>
        <TuningControlsContent capture={capture} device={device} />
      </div>
    </div>
  );
}
