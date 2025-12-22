import { Radio, Activity, Signal, Lock, Unlock, Volume2, VolumeX } from "lucide-react";
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
  const isRunning = system.state === "running" || system.state === "searching" || system.state === "syncing";
  const isStopped = system.state === "stopped";
  const isBusy = isStarting || isStopping;
  const canPlayAudio = isRunning && onPlayAudio && onStopAudio;

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
            <div className="bg-body-secondary rounded p-2 text-center">
              <div className="d-flex align-items-center justify-content-center gap-1 mb-1">
                {getControlChannelIcon(system.controlChannelState)}
                <small className="text-muted">Control</small>
              </div>
              <div className="fw-semibold" style={{ fontSize: "0.9rem" }}>
                {formatFrequency(system.controlChannelFreqHz)}
              </div>
              <small className="text-muted">
                {system.controlChannelState.toUpperCase()}
              </small>
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
        <div className="d-flex gap-3 mt-2 small text-muted">
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
