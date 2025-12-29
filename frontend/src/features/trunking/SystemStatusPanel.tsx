import { useState, useEffect } from "react";
import { Radio, Volume2, VolumeX, Activity } from "lucide-react";
import type { TrunkingSystem } from "../../types/trunking";
import Flex from "../../components/primitives/Flex.react";
import { formatFrequencyMHz } from "../../utils/frequency";
import { formatHex } from "../../utils/formatting";
import { getUnifiedSystemStatus, getChannelSnr } from "../../utils/trunkingStatus";

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

// Channel hunt timeout in seconds (should match backend config)
const HUNT_TIMEOUT_SECONDS = 5;

interface StatBoxProps {
  label: string;
  value: string | number;
  unit?: string;
  highlight?: boolean;
}

function StatBox({ label, value, unit, highlight }: StatBoxProps) {
  return (
    <div className={`rounded p-2 text-center ${highlight ? "bg-success bg-opacity-25" : "bg-body-tertiary"}`}>
      <small className="text-muted d-block" style={{ fontSize: "0.65rem" }}>
        {label}
      </small>
      <div className="fw-semibold" style={{ fontSize: "0.95rem" }}>
        {value}
        {unit && <small className="text-muted ms-1">{unit}</small>}
      </div>
    </div>
  );
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
  // Countdown timer for channel hunting
  const [countdown, setCountdown] = useState(HUNT_TIMEOUT_SECONDS);

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

  const status = getUnifiedSystemStatus(system);
  const snr = getChannelSnr(system);

  return (
    <div className="card">
      <div className="card-header d-flex align-items-center justify-content-between py-2">
        <Flex align="center" gap={2}>
          <Radio size={18} className="text-primary" />
          <span className="fw-semibold">{system.name}</span>
          <span
            className={`badge bg-${status.color}`}
            style={{ fontSize: "0.7rem" }}
            title={status.description}
          >
            {status.label.toUpperCase()}
            {isSearching && ` (${countdown}s)`}
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
        {/* Stats Grid - All stats in consistent boxes */}
        <div className="row g-2">
          {/* Control Channel Frequency */}
          <div className="col-6 col-md-2">
            <StatBox
              label="Control Freq"
              value={system.controlChannelFreqHz ? formatFrequencyMHz(system.controlChannelFreqHz, 4) : "---"}
              unit="MHz"
              highlight={system.controlChannelState === "locked"}
            />
          </div>

          {/* SNR */}
          <div className="col-6 col-md-2">
            <StatBox
              label="SNR"
              value={snr !== null ? snr.toFixed(1) : "---"}
              unit={snr !== null ? "dB" : undefined}
            />
          </div>

          {/* NAC */}
          <div className="col-6 col-md-2">
            <StatBox
              label="NAC"
              value={formatHex(system.nac, 3, false)}
            />
          </div>

          {/* Site ID */}
          <div className="col-6 col-md-2">
            <StatBox
              label="Site"
              value={system.siteId !== null ? system.siteId : "---"}
            />
          </div>

          {/* Decode Rate */}
          <div className="col-6 col-md-2">
            <StatBox
              label="Decode"
              value={system.decodeRate.toFixed(1)}
              unit="fps"
            />
          </div>

          {/* Active Calls */}
          <div className="col-6 col-md-2">
            <StatBox
              label="Active"
              value={system.activeCalls}
              unit="calls"
              highlight={system.activeCalls > 0}
            />
          </div>
        </div>

        {/* Secondary Stats Row - smaller, less prominent */}
        <div className="d-flex gap-3 mt-2 flex-wrap" style={{ fontSize: "0.75rem" }}>
          <span className="text-muted">
            <strong className="text-body">{system.stats.tsbk_count}</strong> TSBKs
          </span>
          <span className="text-muted">
            <strong className="text-body">{system.stats.grant_count}</strong> Grants
          </span>
          <span className="text-muted">
            <strong className="text-body">{system.stats.calls_total}</strong> Total calls
          </span>
          <span className="text-muted">
            <strong className="text-body">{system.stats.recorders_active}</strong>/
            {system.stats.recorders_active + system.stats.recorders_idle} Recorders
          </span>
          {system.stats.initial_scan_complete === false && (
            <span className="text-warning">
              <Activity size={12} className="me-1" style={{ animation: "pulse 1s infinite" }} />
              Scanning channels...
            </span>
          )}
        </div>

        {/* Protocol and Channel Info */}
        <div className="d-flex align-items-center gap-2 mt-2">
          <span
            className={`badge ${
              system.protocol === "p25_phase2"
                ? "bg-info text-dark"
                : "bg-primary"
            }`}
            style={{ fontSize: "0.65rem" }}
          >
            {system.protocol === "p25_phase2" ? "P25 Phase II" : "P25 Phase I"}
          </span>
          {system.stats.cc_scanner && system.stats.cc_scanner.channels_configured > 1 && (
            <span className="text-muted" style={{ fontSize: "0.7rem" }}>
              {system.controlChannels.filter(c => c.enabled).length} of{" "}
              {system.stats.cc_scanner.channels_configured} control channels enabled
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
