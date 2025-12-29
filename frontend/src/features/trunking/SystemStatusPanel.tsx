import { useState, useEffect } from "react";
import { Radio, Volume2, VolumeX, Activity } from "lucide-react";
import type { TrunkingSystem } from "../../types/trunking";
import Flex from "../../components/primitives/Flex.react";
import InfoTooltip from "../../components/primitives/InfoTooltip.react";
import { formatFrequencyMHz } from "../../utils/frequency";
import { formatHex } from "../../utils/formatting";
import {
  getUnifiedSystemStatus,
  getChannelSnr,
} from "../../utils/trunkingStatus";

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

/**
 * Get background class based on SNR value
 * Excellent: > 15 dB → green
 * Good: 10-15 dB → light green
 * Fair: 5-10 dB → yellow/warning
 * Poor: < 5 dB → red/danger
 */
function getSnrBackgroundClass(snrDb: number | null): string {
  if (snrDb === null) return "bg-body-tertiary";
  if (snrDb >= 15) return "bg-success bg-opacity-25";
  if (snrDb >= 10) return "bg-success bg-opacity-10";
  if (snrDb >= 5) return "bg-warning bg-opacity-25";
  return "bg-danger bg-opacity-25";
}

interface StatBoxProps {
  label: string;
  value: string | number;
  unit?: string;
  highlight?: boolean;
  bgClass?: string; // Custom background class (overrides highlight)
  info?: string;
}

function StatBox({
  label,
  value,
  unit,
  highlight,
  bgClass,
  info,
}: StatBoxProps) {
  const defaultBg = highlight ? "bg-success bg-opacity-25" : "bg-body-tertiary";
  return (
    <div className={`rounded p-2 text-center ${bgClass || defaultBg}`}>
      <small className="text-muted d-block" style={{ fontSize: "0.65rem" }}>
        {label}
        {info && <InfoTooltip content={info} />}
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
  const canPlayAudio =
    isRunning && system.state !== "starting" && onPlayAudio && onStopAudio;
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
              value={
                system.controlChannelFreqHz
                  ? formatFrequencyMHz(system.controlChannelFreqHz, 4)
                  : "---"
              }
              unit="MHz"
              highlight={system.controlChannelState === "locked"}
              info="The current control channel frequency. P25 systems broadcast call setup and system info on this channel."
            />
          </div>

          {/* SNR */}
          <div className="col-6 col-md-2">
            <StatBox
              label="SNR"
              value={snr !== null ? snr.toFixed(1) : "---"}
              unit={snr !== null ? "dB" : undefined}
              bgClass={getSnrBackgroundClass(snr)}
              info="Signal-to-Noise Ratio in decibels. Higher is better. Above 10 dB is good, above 15 dB is excellent."
            />
          </div>

          {/* NAC */}
          <div className="col-6 col-md-2">
            <StatBox
              label="NAC"
              value={formatHex(system.nac, 3, false)}
              info="Network Access Code - a 3-digit hex identifier unique to this P25 system. Used to distinguish between overlapping systems."
            />
          </div>

          {/* Site ID */}
          <div className="col-6 col-md-2">
            <StatBox
              label="Site"
              value={system.siteId !== null ? system.siteId : "---"}
              info="Site ID within the trunking system. Large systems have multiple sites (towers) for coverage."
            />
          </div>

          {/* Decode Rate */}
          <div className="col-6 col-md-2">
            <StatBox
              label="Decode"
              value={system.decodeRate.toFixed(1)}
              unit="fps"
              info="Control channel decode rate in frames per second. Expect 10,000-20,000 fps when locked to a strong signal. Low values indicate weak signal or interference."
            />
          </div>

          {/* Active Calls */}
          <div className="col-6 col-md-2">
            <StatBox
              label="Active"
              value={system.activeCalls}
              unit="calls"
              highlight={system.activeCalls > 0}
              info="Number of voice calls currently in progress on this system."
            />
          </div>
        </div>

        {/* Secondary Stats Row - smaller, less prominent */}
        <div
          className="d-flex gap-3 mt-2 flex-wrap"
          style={{ fontSize: "0.75rem" }}
        >
          <span className="text-muted">
            <strong className="text-body">{system.stats.tsbk_count}</strong>{" "}
            TSBKs
            <InfoTooltip content="Trunking Signaling Blocks - control channel messages received. Includes channel grants, affiliations, and system info." />
          </span>
          <span className="text-muted">
            <strong className="text-body">{system.stats.grant_count}</strong>{" "}
            Grants
            <InfoTooltip content="Voice channel grants - messages that assign a talkgroup to a voice frequency for a call." />
          </span>
          <span className="text-muted">
            <strong className="text-body">{system.stats.calls_total}</strong>{" "}
            Total calls
            <InfoTooltip content="Total voice calls detected since the system started." />
          </span>
          <span className="text-muted">
            <strong className="text-body">
              {system.stats.recorders_active}
            </strong>
            /{system.stats.recorders_active + system.stats.recorders_idle}{" "}
            Recorders
            <InfoTooltip content="Voice recorders: active/total. Each recorder can capture one voice call at a time." />
          </span>
          {system.stats.initial_scan_complete === false && (
            <span className="text-warning">
              <Activity
                size={12}
                className="me-1"
                style={{ animation: "pulse 1s infinite" }}
              />
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
          {system.stats.cc_scanner &&
            system.stats.cc_scanner.channels_configured > 1 && (
              <span className="text-muted" style={{ fontSize: "0.7rem" }}>
                {system.controlChannels.filter((c) => c.enabled).length} of{" "}
                {system.stats.cc_scanner.channels_configured} control channels
                enabled
              </span>
            )}
        </div>
      </div>
    </div>
  );
}
