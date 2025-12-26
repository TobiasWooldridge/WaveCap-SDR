import { useState, useEffect } from "react";
import { Radio, Activity, Signal, Lock, Unlock, Volume2, VolumeX, Info } from "lucide-react";
import type { TrunkingSystem } from "../../types/trunking";
import Flex from "../../components/primitives/Flex.react";

interface SystemStatusPanelProps {
  system: TrunkingSystem;
  onStart: () => void;
  onStop: () => void;
  isStarting?: boolean;
  isStopping?: boolean;
  // Audio controls
  isPlayingAudio?: boolean;
  onPlayAudio?: () => void;
  onStopAudio?: () => void;
}

function formatFrequency(hz: number | null): string {
  if (hz === null) return "---";
  return (hz / 1_000_000).toFixed(4) + " MHz";
}

function getStateColor(state: string): string {
  switch (state) {
    case "running":
      return "success";
    case "syncing":
    case "searching":
    case "starting":
      return "warning";
    case "failed":
      return "danger";
    default:
      return "secondary";
  }
}

function getControlChannelIcon(state: string) {
  switch (state) {
    case "locked":
      return <Lock size={14} className="text-success" />;
    case "searching":
      return <Activity size={14} className="text-warning" />;
    case "lost":
      return <Unlock size={14} className="text-danger" />;
    default:
      return <Unlock size={14} className="text-muted" />;
  }
}

// Get the 1-based index of the current control channel in the scanner measurements
function getChannelIndex(system: TrunkingSystem): number {
  const scanner = system.stats.cc_scanner;
  if (!scanner || !scanner.current_channel_hz || !scanner.measurements) {
    return 1;
  }

  // Get sorted list of measured frequencies
  const freqs = Object.keys(scanner.measurements)
    .map(key => parseFloat(key.replace("_MHz", "")) * 1e6)
    .sort((a, b) => a - b);

  const currentHz = scanner.current_channel_hz;
  const index = freqs.findIndex(f => Math.abs(f - currentHz) < 1000); // 1 kHz tolerance
  return index >= 0 ? index + 1 : 1;
}

// Get SNR for current control channel
function getChannelSnr(system: TrunkingSystem): number | null {
  const scanner = system.stats.cc_scanner;
  if (!scanner || !scanner.current_channel_hz || !scanner.measurements) {
    return null;
  }

  // Find measurement matching current channel
  const currentMHz = (scanner.current_channel_hz / 1e6).toFixed(4);
  const key = `${currentMHz}_MHz`;
  const measurement = scanner.measurements[key];

  return measurement ? measurement.snr_db : null;
}

// Channel hunt timeout in seconds (should match backend config)
const HUNT_TIMEOUT_SECONDS = 5;

export function SystemStatusPanel({
  system,
  onStart,
  onStop,
  isStarting,
  isStopping,
  isPlayingAudio,
  onPlayAudio,
  onStopAudio,
}: SystemStatusPanelProps) {
  // Countdown timer for channel hunting
  const [countdown, setCountdown] = useState(HUNT_TIMEOUT_SECONDS);
  const [showInfo, setShowInfo] = useState(false);

  // Reset countdown when control channel changes
  useEffect(() => {
    if (system.controlChannelState === "searching") {
      setCountdown(HUNT_TIMEOUT_SECONDS);
      const timer = setInterval(() => {
        setCountdown((prev) => Math.max(0, prev - 1));
      }, 1000);
      return () => clearInterval(timer);
    }
  }, [system.controlChannelFreqHz, system.controlChannelState]);

  // Show Stop button for any active state
  const isRunning = system.state !== "stopped" && system.state !== "failed";
  const isStopped = system.state === "stopped" || system.state === "failed";
  const isBusy = isStarting || isStopping;
  // Show Listen button when system is actively decoding
  const canPlayAudio = isRunning && system.state !== "starting" && onPlayAudio && onStopAudio;
  const isSearching = system.controlChannelState === "searching";

  return (
    <div className="card">
      <div className="card-header d-flex align-items-center justify-content-between py-2">
        <Flex align="center" gap={2}>
          <Radio size={18} className="text-primary" />
          <span className="fw-semibold">{system.name}</span>
          <span
            className={`badge bg-${getStateColor(system.state)}`}
            style={{ fontSize: "0.65rem" }}
          >
            {system.state.toUpperCase()}
          </span>
        </Flex>
        <Flex gap={1}>
          {/* Audio play/stop button */}
          {canPlayAudio && (
            <button
              className={`btn btn-sm ${isPlayingAudio ? "btn-warning" : "btn-success"}`}
              onClick={isPlayingAudio ? onStopAudio : onPlayAudio}
              title={isPlayingAudio ? "Stop audio" : "Play all audio"}
            >
              <Flex align="center" gap={1}>
                {isPlayingAudio ? <VolumeX size={14} /> : <Volume2 size={14} />}
                <span style={{ fontSize: "11px" }}>
                  {isPlayingAudio ? "Stop" : "Listen"}
                </span>
              </Flex>
            </button>
          )}
          {isStopped && (
            <button
              className="btn btn-sm btn-success"
              onClick={onStart}
              disabled={isBusy}
            >
              {isStarting ? "Starting..." : "Start"}
            </button>
          )}
          {isRunning && (
            <button
              className="btn btn-sm btn-danger"
              onClick={onStop}
              disabled={isBusy}
            >
              {isStopping ? "Stopping..." : "Stop"}
            </button>
          )}
        </Flex>
      </div>

      <div className="card-body py-2">
        {/* Status Grid */}
        <div className="row g-2">
          {/* Control Channel */}
          <div className="col-6 col-md-3">
            <div className="bg-body-secondary rounded p-2 text-center position-relative">
              <div className="d-flex align-items-center justify-content-center gap-1 mb-1">
                {getControlChannelIcon(system.controlChannelState)}
                <small className="text-muted">Control</small>
                {system.stats.cc_scanner && system.stats.cc_scanner.channels_configured > 1 && (
                  <>
                    <small className="text-muted">
                      ({getChannelIndex(system)} of {system.stats.cc_scanner.channels_configured})
                    </small>
                    <button
                      className="btn btn-link p-0 border-0"
                      onClick={() => setShowInfo(!showInfo)}
                      title="What do these numbers mean?"
                      style={{ lineHeight: 1 }}
                    >
                      <Info size={12} className="text-muted" />
                    </button>
                  </>
                )}
              </div>
              <div className="fw-semibold" style={{ fontSize: "0.9rem" }}>
                {formatFrequency(system.controlChannelFreqHz)}
              </div>
              <div className="d-flex align-items-center justify-content-center gap-1">
                <small className="text-muted">
                  {system.controlChannelState.toUpperCase()}
                </small>
                {isSearching && (
                  <small className="text-warning">
                    ({countdown}s)
                  </small>
                )}
                {system.stats.cc_scanner && getChannelSnr(system) !== null && (
                  <small className="text-success">
                    {getChannelSnr(system)!.toFixed(1)} dB
                  </small>
                )}
              </div>
              {/* Info tooltip */}
              {showInfo && (
                <div
                  className="position-absolute bg-dark text-white rounded p-2 shadow"
                  style={{
                    top: "100%",
                    left: "50%",
                    transform: "translateX(-50%)",
                    zIndex: 1000,
                    width: "220px",
                    fontSize: "0.75rem",
                    textAlign: "left",
                    marginTop: "4px",
                  }}
                >
                  <strong>Control Channel Hunting</strong>
                  <div className="mt-1">
                    <strong>{getChannelIndex(system)}</strong> = Current channel being tried
                  </div>
                  <div>
                    <strong>{system.stats.cc_scanner?.channels_configured}</strong> = Total configured control channels
                  </div>
                  <div className="mt-1 text-muted">
                    The system tries each channel for {HUNT_TIMEOUT_SECONDS}s until it finds
                    a valid P25 control channel signal.
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* NAC */}
          <div className="col-6 col-md-3">
            <div className="bg-body-secondary rounded p-2 text-center">
              <small className="text-muted d-block mb-1">NAC</small>
              <div className="fw-semibold" style={{ fontSize: "1.1rem" }}>
                {system.nac !== null
                  ? system.nac.toString(16).toUpperCase().padStart(3, "0")
                  : "---"}
              </div>
            </div>
          </div>

          {/* Site ID */}
          <div className="col-6 col-md-3">
            <div className="bg-body-secondary rounded p-2 text-center">
              <small className="text-muted d-block mb-1">Site</small>
              <div className="fw-semibold" style={{ fontSize: "1.1rem" }}>
                {system.siteId !== null ? `${system.siteId}` : "---"}
              </div>
            </div>
          </div>

          {/* Decode Rate */}
          <div className="col-6 col-md-3">
            <div className="bg-body-secondary rounded p-2 text-center">
              <div className="d-flex align-items-center justify-content-center gap-1 mb-1">
                <Signal size={14} />
                <small className="text-muted">Decode</small>
              </div>
              <div className="fw-semibold" style={{ fontSize: "1.1rem" }}>
                {system.decodeRate.toFixed(1)}
                <small className="text-muted"> fps</small>
              </div>
            </div>
          </div>
        </div>

        {/* Stats Row */}
        <div className="d-flex gap-3 mt-2 small text-muted flex-wrap">
          <span>
            <strong>{system.stats.tsbk_count}</strong> TSBKs
          </span>
          <span>
            <strong>{system.stats.grant_count}</strong> Grants
          </span>
          <span>
            <strong>{system.stats.calls_total}</strong> Calls
          </span>
          <span>
            <strong>{system.stats.recorders_active}</strong>/
            {system.stats.recorders_active + system.stats.recorders_idle}{" "}
            Recorders
          </span>
          {system.stats.initial_scan_complete === false && (
            <span className="text-warning">
              <Activity size={12} className="me-1" style={{ animation: "pulse 1s infinite" }} />
              Scanning channels...
            </span>
          )}
        </div>

        {/* Protocol Badge */}
        <div className="mt-2">
          <span
            className={`badge ${
              system.protocol === "p25_phase2"
                ? "bg-info text-dark"
                : "bg-primary"
            }`}
          >
            {system.protocol === "p25_phase2" ? "P25 Phase II" : "P25 Phase I"}
          </span>
        </div>
      </div>
    </div>
  );
}
