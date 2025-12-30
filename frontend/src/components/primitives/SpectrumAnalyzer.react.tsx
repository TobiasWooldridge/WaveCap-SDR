import { useEffect, useRef, useState } from "react";
import { ChevronUp, ChevronDown, Minus, Plus } from "lucide-react";
import type { Capture, Channel } from "../../types";
import { useSpectrumData } from "../../hooks/useSpectrumData";
import { formatBandwidth, formatFrequencyWithUnit } from "../../utils/frequency";
import {
  getCaptureStatusMessage,
  drawCaptureStatusOnCanvas,
} from "../../utils/captureStatus";

export interface SpectrumAnalyzerProps {
  capture: Capture;
  channels?: Channel[];
  height?: number;
  onFrequencyClick?: (frequencyHz: number) => void;
}

const MIN_HEIGHT = 100;
const MAX_HEIGHT = 600;
const HEIGHT_STEP = 25;
const STORAGE_KEY = "spectrumAnalyzerHeight";

// Common frequency band plans with labels
interface FrequencyBand {
  name: string;
  startHz: number;
  endHz: number;
  color: string; // Fill color (0.2 alpha)
  strokeColor: string; // Border color (0.5 alpha)
  textColor: string; // Label text color (solid)
}

/**
 * Fast min/max calculation using single-pass loop.
 * Avoids spread operator overhead on large arrays (1000+ elements).
 */
function getMinMax(arr: number[]): { min: number; max: number } {
  if (arr.length === 0) return { min: 0, max: 0 };
  let min = arr[0];
  let max = arr[0];
  for (let i = 1; i < arr.length; i++) {
    const val = arr[i];
    if (val < min) min = val;
    if (val > max) max = val;
  }
  return { min, max };
}

// Pre-computed color variants to avoid string manipulation every frame
const BAND_PLAN: FrequencyBand[] = [
  // AM Broadcast
  {
    name: "AM Broadcast",
    startHz: 530e3,
    endHz: 1700e3,
    color: "rgba(255, 193, 7, 0.2)",
    strokeColor: "rgba(255, 193, 7, 0.5)",
    textColor: "rgb(255, 193, 7)",
  },
  // Shortwave Broadcast
  {
    name: "Shortwave",
    startHz: 3e6,
    endHz: 30e6,
    color: "rgba(111, 66, 193, 0.2)",
    strokeColor: "rgba(111, 66, 193, 0.5)",
    textColor: "rgb(111, 66, 193)",
  },
  // Citizens Band (CB)
  {
    name: "CB Radio",
    startHz: 26.965e6,
    endHz: 27.405e6,
    color: "rgba(255, 87, 34, 0.2)",
    strokeColor: "rgba(255, 87, 34, 0.5)",
    textColor: "rgb(255, 87, 34)",
  },
  // 10m Ham Band
  {
    name: "10m Ham",
    startHz: 28e6,
    endHz: 29.7e6,
    color: "rgba(0, 150, 136, 0.2)",
    strokeColor: "rgba(0, 150, 136, 0.5)",
    textColor: "rgb(0, 150, 136)",
  },
  // 6m Ham Band
  {
    name: "6m Ham",
    startHz: 50e6,
    endHz: 54e6,
    color: "rgba(0, 150, 136, 0.2)",
    strokeColor: "rgba(0, 150, 136, 0.5)",
    textColor: "rgb(0, 150, 136)",
  },
  // FM Broadcast
  {
    name: "FM Broadcast",
    startHz: 88e6,
    endHz: 108e6,
    color: "rgba(33, 150, 243, 0.2)",
    strokeColor: "rgba(33, 150, 243, 0.5)",
    textColor: "rgb(33, 150, 243)",
  },
  // Aircraft
  {
    name: "Aircraft",
    startHz: 108e6,
    endHz: 137e6,
    color: "rgba(76, 175, 80, 0.2)",
    strokeColor: "rgba(76, 175, 80, 0.5)",
    textColor: "rgb(76, 175, 80)",
  },
  // 2m Ham Band
  {
    name: "2m Ham",
    startHz: 144e6,
    endHz: 148e6,
    color: "rgba(0, 150, 136, 0.2)",
    strokeColor: "rgba(0, 150, 136, 0.5)",
    textColor: "rgb(0, 150, 136)",
  },
  // Marine VHF
  {
    name: "Marine VHF",
    startHz: 156e6,
    endHz: 163e6,
    color: "rgba(3, 169, 244, 0.2)",
    strokeColor: "rgba(3, 169, 244, 0.5)",
    textColor: "rgb(3, 169, 244)",
  },
  // Weather Radio (NOAA)
  {
    name: "NOAA Weather",
    startHz: 162.4e6,
    endHz: 162.55e6,
    color: "rgba(255, 152, 0, 0.2)",
    strokeColor: "rgba(255, 152, 0, 0.5)",
    textColor: "rgb(255, 152, 0)",
  },
  // Railroad
  {
    name: "Railroad",
    startHz: 159.81e6,
    endHz: 161.565e6,
    color: "rgba(121, 85, 72, 0.2)",
    strokeColor: "rgba(121, 85, 72, 0.5)",
    textColor: "rgb(121, 85, 72)",
  },
  // Business/Public Safety
  {
    name: "Business",
    startHz: 150e6,
    endHz: 156e6,
    color: "rgba(158, 158, 158, 0.2)",
    strokeColor: "rgba(158, 158, 158, 0.5)",
    textColor: "rgb(158, 158, 158)",
  },
  // 1.25m Ham Band
  {
    name: "1.25m Ham",
    startHz: 222e6,
    endHz: 225e6,
    color: "rgba(0, 150, 136, 0.2)",
    strokeColor: "rgba(0, 150, 136, 0.5)",
    textColor: "rgb(0, 150, 136)",
  },
  // 70cm Ham Band
  {
    name: "70cm Ham",
    startHz: 420e6,
    endHz: 450e6,
    color: "rgba(0, 150, 136, 0.2)",
    strokeColor: "rgba(0, 150, 136, 0.5)",
    textColor: "rgb(0, 150, 136)",
  },
  // FRS/GMRS
  {
    name: "FRS/GMRS",
    startHz: 462.5625e6,
    endHz: 467.7125e6,
    color: "rgba(233, 30, 99, 0.2)",
    strokeColor: "rgba(233, 30, 99, 0.5)",
    textColor: "rgb(233, 30, 99)",
  },
  // Weather Satellites (NOAA APT)
  {
    name: "Weather Sats",
    startHz: 137e6,
    endHz: 138e6,
    color: "rgba(103, 58, 183, 0.2)",
    strokeColor: "rgba(103, 58, 183, 0.5)",
    textColor: "rgb(103, 58, 183)",
  },
  // ISS/Amateur Satellites
  {
    name: "Sat Downlink",
    startHz: 145.8e6,
    endHz: 146e6,
    color: "rgba(63, 81, 181, 0.2)",
    strokeColor: "rgba(63, 81, 181, 0.5)",
    textColor: "rgb(63, 81, 181)",
  },
];

export default function SpectrumAnalyzer({
  capture,
  channels = [],
  height: initialHeight = 200,
  onFrequencyClick,
}: SpectrumAnalyzerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(800);
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [hoverFrequency, setHoverFrequency] = useState<number | null>(null);
  const [hoverPosition, setHoverPosition] = useState<{
    x: number;
    y: number;
  } | null>(null);

  // Load height from localStorage or use initial value
  const [height, setHeight] = useState(() => {
    const stored = localStorage.getItem(STORAGE_KEY);
    return stored ? parseInt(stored, 10) : initialHeight;
  });

  // Peak hold and averaging modes
  const [peakHoldEnabled, setPeakHoldEnabled] = useState(false);
  const [averagingEnabled, setAveragingEnabled] = useState(false);
  const [bandPlanEnabled, setBandPlanEnabled] = useState(false);
  const peakHoldData = useRef<number[]>([]);
  const avgHistory = useRef<number[][]>([]);
  const PEAK_DECAY_RATE = 0.98; // Decay factor per frame
  const AVG_FRAMES = 4; // Number of frames to average

  // Save height to localStorage when it changes
  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, height.toString());
  }, [height]);

  const increaseHeight = () => {
    setHeight((prev) => Math.min(MAX_HEIGHT, prev + HEIGHT_STEP));
  };

  const decreaseHeight = () => {
    setHeight((prev) => Math.max(MIN_HEIGHT, prev - HEIGHT_STEP));
  };

  // Use shared WebSocket connection - paused when collapsed
  const { spectrumData, isConnected, isIdle } = useSpectrumData(
    capture,
    isCollapsed,
  );

  // Update canvas width when container resizes
  useEffect(() => {
    const updateWidth = () => {
      if (containerRef.current) {
        const containerWidth = containerRef.current.offsetWidth;
        setWidth(containerWidth - 16); // Subtract padding
      }
    };

    updateWidth();
    window.addEventListener("resize", updateWidth);
    return () => window.removeEventListener("resize", updateWidth);
  }, []);

  // Clear peak hold and averaging history when capture (radio) changes
  useEffect(() => {
    peakHoldData.current = [];
    avgHistory.current = [];
  }, [capture.id]);

  // Draw spectrum on canvas
  useEffect(() => {
    if (!canvasRef.current) {
      return;
    }

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }

    // Clear canvas with light background
    ctx.fillStyle = "#f8f9fa";
    ctx.fillRect(0, 0, width, height);

    // If no spectrum data, show appropriate message based on capture state
    if (!spectrumData) {
      const status = getCaptureStatusMessage(capture, isConnected);
      drawCaptureStatusOnCanvas(ctx, width, height, status);
      return;
    }

    const { power, freqs, centerHz } = spectrumData;
    if (power.length === 0) {
      return;
    }

    // Calculate frequency range for use throughout rendering
    const freqMin = centerHz + freqs[0];
    const freqMax = centerHz + freqs[freqs.length - 1];
    const freqMid = centerHz;

    // Apply averaging if enabled
    let displayPower = power;
    if (averagingEnabled) {
      // Add current frame to history
      avgHistory.current.push([...power]);
      // Keep only last N frames
      if (avgHistory.current.length > AVG_FRAMES) {
        avgHistory.current.shift();
      }
      // Calculate average
      if (avgHistory.current.length > 0) {
        displayPower = power.map((_, i) => {
          const sum = avgHistory.current.reduce(
            (acc, frame) => acc + frame[i],
            0,
          );
          return sum / avgHistory.current.length;
        });
      }
    } else {
      // Reset averaging history when disabled
      avgHistory.current = [];
    }

    // Update peak hold data
    if (peakHoldEnabled) {
      if (peakHoldData.current.length !== power.length) {
        peakHoldData.current = [...power];
      } else {
        // Update peaks and apply decay
        peakHoldData.current = peakHoldData.current.map((peak, i) => {
          const decayed = peak * PEAK_DECAY_RATE;
          return Math.max(decayed, power[i]);
        });
      }
    } else {
      // Reset peak hold when disabled
      peakHoldData.current = [];
    }

    // Find min/max for scaling (single-pass, avoids spread operator overhead)
    const { min: minPower, max: maxPower } = getMinMax(displayPower);
    const powerRange = maxPower - minPower;

    // Draw spectrum with primary blue color
    ctx.strokeStyle = "#0d6efd";
    ctx.lineWidth = 2;
    ctx.beginPath();

    for (let i = 0; i < displayPower.length; i++) {
      const x = (i / displayPower.length) * width;
      // Normalize power to canvas height (invert y-axis)
      const normalized = (displayPower[i] - minPower) / (powerRange || 1);
      const y = height - normalized * height;

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }

    ctx.stroke();

    // Draw peak hold overlay if enabled
    if (peakHoldEnabled && peakHoldData.current.length > 0) {
      ctx.strokeStyle = "#dc3545";
      ctx.lineWidth = 1;
      ctx.globalAlpha = 0.6;
      ctx.beginPath();

      for (let i = 0; i < peakHoldData.current.length; i++) {
        const x = (i / peakHoldData.current.length) * width;
        const normalized =
          (peakHoldData.current[i] - minPower) / (powerRange || 1);
        const y = height - normalized * height;

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }

      ctx.stroke();
      ctx.globalAlpha = 1.0;
    }

    // Draw grid lines with subtle color
    ctx.strokeStyle = "#dee2e6";
    ctx.lineWidth = 1;

    // Horizontal grid lines (power levels)
    for (let i = 0; i <= 4; i++) {
      const y = (i / 4) * height;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Vertical grid lines (frequency divisions)
    for (let i = 0; i <= 8; i++) {
      const x = (i / 8) * width;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }

    // Draw center frequency marker with danger/red color
    const centerX = width / 2;
    ctx.strokeStyle = "#dc3545";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(centerX, 0);
    ctx.lineTo(centerX, height);
    ctx.stroke();

    // Draw band plan overlay if enabled
    if (bandPlanEnabled) {
      const freqSpan = freqMax - freqMin;

      BAND_PLAN.forEach((band) => {
        // Check if band overlaps with visible spectrum
        if (band.endHz >= freqMin && band.startHz <= freqMax) {
          // Calculate visible portion of band
          const bandStart = Math.max(band.startHz, freqMin);
          const bandEnd = Math.min(band.endHz, freqMax);

          // Convert to canvas coordinates
          const x1 = ((bandStart - freqMin) / freqSpan) * width;
          const x2 = ((bandEnd - freqMin) / freqSpan) * width;
          const bandWidth = x2 - x1;

          // Only draw if band is visible (at least 5 pixels wide)
          if (bandWidth >= 5) {
            // Draw colored overlay
            ctx.fillStyle = band.color;
            ctx.fillRect(x1, 0, bandWidth, height);

            // Draw border (use pre-computed strokeColor)
            ctx.strokeStyle = band.strokeColor;
            ctx.lineWidth = 1;
            ctx.strokeRect(x1, 0, bandWidth, height);

            // Draw label if there's enough space (at least 40 pixels)
            if (bandWidth >= 40) {
              ctx.font = "bold 10px sans-serif";
              ctx.textAlign = "center";
              ctx.textBaseline = "top";

              // Draw label background
              const labelText = band.name;
              const labelWidth = ctx.measureText(labelText).width;
              const labelX = x1 + bandWidth / 2;

              ctx.fillStyle = "rgba(255, 255, 255, 0.85)";
              ctx.fillRect(labelX - labelWidth / 2 - 3, 5, labelWidth + 6, 14);

              // Draw label text
              ctx.fillStyle = "#000";
              ctx.fillText(labelText, labelX, 8);
            }
          }
        }
      });
    }

    // Draw frequency labels with backgrounds
    ctx.font = "10px monospace";
    const freqMinText = formatFrequencyWithUnit(freqMin, 3);
    const freqMidText = formatFrequencyWithUnit(freqMid, 3);
    const freqMaxText = formatFrequencyWithUnit(freqMax, 3);

    // Draw backgrounds for frequency labels
    ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
    const freqMinWidth = ctx.measureText(freqMinText).width;
    const freqMidWidth = ctx.measureText(freqMidText).width;
    const freqMaxWidth = ctx.measureText(freqMaxText).width;
    ctx.fillRect(3, height - 16, freqMinWidth + 4, 13);
    ctx.fillRect(width / 2 - 32, height - 16, freqMidWidth + 4, 13);
    ctx.fillRect(width - 72, height - 16, freqMaxWidth + 4, 13);

    // Draw frequency label text
    ctx.fillStyle = "#6c757d";
    ctx.fillText(freqMinText, 5, height - 5);
    ctx.fillText(freqMidText, width / 2 - 30, height - 5);
    ctx.fillText(freqMaxText, width - 70, height - 5);

    // Draw power labels with backgrounds
    const maxPowerText = `${maxPower.toFixed(1)} dB`;
    const minPowerText = `${minPower.toFixed(1)} dB`;
    const maxPowerWidth = ctx.measureText(maxPowerText).width;
    const minPowerWidth = ctx.measureText(minPowerText).width;

    ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
    ctx.fillRect(3, 1, maxPowerWidth + 4, 13);
    ctx.fillRect(3, height - 26, minPowerWidth + 4, 13);

    ctx.fillStyle = "#6c757d";
    ctx.fillText(maxPowerText, 5, 12);
    ctx.fillText(minPowerText, 5, height - 15);

    // Helper function to format rate values
    const formatRate = (rate: number): string => {
      if (rate >= 1e9) return `${(rate / 1e9).toFixed(2)} GS/s`;
      if (rate >= 1e6) return `${(rate / 1e6).toFixed(2)} MS/s`;
      if (rate >= 1e3) return `${(rate / 1e3).toFixed(2)} kS/s`;
      return `${rate.toFixed(0)} S/s`;
    };

    // Draw Sample Rate overlay (full width)
    const sampleRateY = height * 0.1;
    const overlayHeight = 15;

    ctx.fillStyle = "rgba(0, 123, 255, 0.15)";
    ctx.fillRect(0, sampleRateY, width, overlayHeight);

    ctx.strokeStyle = "rgba(0, 123, 255, 0.5)";
    ctx.lineWidth = 1;
    ctx.strokeRect(0, sampleRateY, width, overlayHeight);

    ctx.font = "bold 9px monospace";
    const sampleRateText = `Sample Rate: ${formatRate(capture.sampleRate)}`;
    const sampleRateTextWidth = ctx.measureText(sampleRateText).width;
    const sampleRateTextX = (width - sampleRateTextWidth) / 2;

    // Draw white background for text
    ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
    ctx.fillRect(
      sampleRateTextX - 2,
      sampleRateY + 2,
      sampleRateTextWidth + 4,
      11,
    );

    // Draw text
    ctx.fillStyle = "#007bff";
    ctx.fillText(sampleRateText, sampleRateTextX, sampleRateY + 10);

    // Draw Bandwidth overlay (if available and centered)
    if (capture.bandwidth !== null && capture.bandwidth !== undefined) {
      const bandwidthY = height * 0.18;
      const freqSpan = freqMax - freqMin;

      // Calculate bandwidth width relative to spectrum span
      const bandwidthWidth = (capture.bandwidth / freqSpan) * width;
      const bandwidthX = (width - bandwidthWidth) / 2; // Center it

      ctx.fillStyle = "rgba(40, 167, 69, 0.15)";
      ctx.fillRect(bandwidthX, bandwidthY, bandwidthWidth, overlayHeight);

      ctx.strokeStyle = "rgba(40, 167, 69, 0.5)";
      ctx.lineWidth = 1;
      ctx.strokeRect(bandwidthX, bandwidthY, bandwidthWidth, overlayHeight);

      ctx.font = "bold 9px monospace";
      const bandwidthText = `Bandwidth: ${formatBandwidth(capture.bandwidth)}`;
      const bandwidthTextWidth = ctx.measureText(bandwidthText).width;
      const bandwidthTextX = (width - bandwidthTextWidth) / 2;

      // Draw white background for text
      ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
      ctx.fillRect(
        bandwidthTextX - 2,
        bandwidthY + 2,
        bandwidthTextWidth + 4,
        11,
      );

      // Draw text
      ctx.fillStyle = "#28a745";
      ctx.fillText(bandwidthText, bandwidthTextX, bandwidthY + 10);
    }

    // Draw channel markers
    if (channels && channels.length > 0) {
      const freqSpan = freqMax - freqMin;

      channels.forEach((channel, idx) => {
        // Calculate channel's absolute frequency
        const channelFreq = centerHz + channel.offsetHz;

        // Calculate x position on spectrum
        const freqOffset = channelFreq - freqMin;
        const x = (freqOffset / freqSpan) * width;

        // Only draw if channel is within visible spectrum
        if (x >= 0 && x <= width) {
          // Draw channel marker line with success/secondary color
          ctx.strokeStyle = channel.state === "running" ? "#198754" : "#adb5bd";
          ctx.lineWidth = 2;
          ctx.setLineDash([5, 3]);
          ctx.beginPath();
          ctx.moveTo(x, 0);
          ctx.lineTo(x, height);
          ctx.stroke();
          ctx.setLineDash([]);

          // Draw channel label at top with background
          ctx.font = "bold 10px monospace";
          const label = `CH${idx + 1}`;
          const labelWidth = ctx.measureText(label).width;

          // Draw background
          ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
          ctx.fillRect(x - labelWidth / 2 - 2, 0, labelWidth + 4, 12);

          // Draw text
          ctx.fillStyle = channel.state === "running" ? "#198754" : "#6c757d";
          ctx.fillText(label, x - labelWidth / 2, 10);
        }
      });
    }
  }, [
    spectrumData,
    width,
    height,
    channels,
    capture.bandwidth,
    capture.sampleRate,
    capture,
    capture.state,
    capture.errorMessage,
    peakHoldEnabled,
    averagingEnabled,
    bandPlanEnabled,
    isConnected,
  ]);

  // Determine badge status
  const getBadgeStatus = () => {
    if (isCollapsed) {
      return { text: "PAUSED", className: "bg-secondary" };
    } else if (isIdle && capture.state === "running") {
      return { text: "PAUSED (IDLE)", className: "bg-warning" };
    } else if (isConnected) {
      return { text: "LIVE", className: "bg-success" };
    } else {
      return { text: "OFFLINE", className: "bg-secondary" };
    }
  };

  const badgeStatus = getBadgeStatus();

  // Handle canvas click to tune to frequency
  const handleCanvasClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!onFrequencyClick || !spectrumData) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;

    // Calculate frequency from click position
    const { freqs, centerHz } = spectrumData;
    const freqMin = centerHz + freqs[0];
    const freqMax = centerHz + freqs[freqs.length - 1];
    const freqSpan = freqMax - freqMin;

    // Convert X position to frequency
    const clickedFrequency = freqMin + (x / width) * freqSpan;

    onFrequencyClick(Math.round(clickedFrequency));
  };

  // Handle mouse move to show frequency tooltip
  const handleCanvasMouseMove = (
    event: React.MouseEvent<HTMLCanvasElement>,
  ) => {
    if (!spectrumData) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;

    // Calculate frequency from mouse position
    const { freqs, centerHz } = spectrumData;
    const freqMin = centerHz + freqs[0];
    const freqMax = centerHz + freqs[freqs.length - 1];
    const freqSpan = freqMax - freqMin;

    // Convert X position to frequency
    const frequency = freqMin + (x / width) * freqSpan;

    setHoverFrequency(frequency);
    setHoverPosition({ x: event.clientX, y: event.clientY });
  };

  // Handle mouse leave to hide tooltip
  const handleCanvasMouseLeave = () => {
    setHoverFrequency(null);
    setHoverPosition(null);
  };

  return (
    <div className="card shadow-sm">
      <div className="card-header bg-body-tertiary py-1 px-2">
        <div className="d-flex justify-content-between align-items-center">
          <small className="fw-semibold mb-0">Spectrum Analyzer</small>
          <div className="d-flex align-items-center gap-2">
            <span
              className={`badge ${badgeStatus.className} text-white`}
              style={{ fontSize: "8px", padding: "2px 6px" }}
            >
              {badgeStatus.text}
            </span>
            {spectrumData?.actualFps !== undefined &&
              isConnected &&
              !isIdle && (
                <span
                  className="badge bg-info text-white"
                  style={{
                    fontSize: "8px",
                    padding: "2px 6px",
                    minWidth: "42px",
                    textAlign: "center",
                    fontVariantNumeric: "tabular-nums",
                  }}
                  title={`FFT: ${spectrumData.fftSize || "N/A"} bins`}
                >
                  {spectrumData.actualFps.toFixed(0)} FPS
                </span>
              )}
            {!isCollapsed && (
              <>
                <div
                  className="d-flex align-items-center gap-1"
                  style={{ fontSize: "10px" }}
                >
                  <label
                    className="d-flex align-items-center gap-1 mb-0"
                    style={{ cursor: "pointer" }}
                  >
                    <input
                      type="checkbox"
                      checked={peakHoldEnabled}
                      onChange={(e) => setPeakHoldEnabled(e.target.checked)}
                      style={{ width: "12px", height: "12px" }}
                    />
                    <span className="text-muted">Peak</span>
                  </label>
                  <label
                    className="d-flex align-items-center gap-1 mb-0"
                    style={{ cursor: "pointer" }}
                  >
                    <input
                      type="checkbox"
                      checked={averagingEnabled}
                      onChange={(e) => setAveragingEnabled(e.target.checked)}
                      style={{ width: "12px", height: "12px" }}
                    />
                    <span className="text-muted">Avg</span>
                  </label>
                  <label
                    className="d-flex align-items-center gap-1 mb-0"
                    style={{ cursor: "pointer" }}
                  >
                    <input
                      type="checkbox"
                      checked={bandPlanEnabled}
                      onChange={(e) => setBandPlanEnabled(e.target.checked)}
                      style={{ width: "12px", height: "12px" }}
                    />
                    <span className="text-muted">Bands</span>
                  </label>
                </div>
                <button
                  className="btn btn-sm btn-outline-secondary p-0"
                  style={{ width: "20px", height: "20px", lineHeight: 1 }}
                  onClick={decreaseHeight}
                  disabled={height <= MIN_HEIGHT}
                  title="Decrease height"
                >
                  <Minus size={12} />
                </button>
                <span
                  className="small text-muted"
                  style={{
                    fontSize: "10px",
                    minWidth: "35px",
                    textAlign: "center",
                  }}
                >
                  {height}px
                </span>
                <button
                  className="btn btn-sm btn-outline-secondary p-0"
                  style={{ width: "20px", height: "20px", lineHeight: 1 }}
                  onClick={increaseHeight}
                  disabled={height >= MAX_HEIGHT}
                  title="Increase height"
                >
                  <Plus size={12} />
                </button>
              </>
            )}
            <button
              className="btn btn-sm btn-outline-secondary p-0"
              style={{ width: "20px", height: "20px", lineHeight: 1 }}
              onClick={() => setIsCollapsed(!isCollapsed)}
              title={isCollapsed ? "Expand" : "Collapse"}
            >
              {isCollapsed ? (
                <ChevronDown size={14} />
              ) : (
                <ChevronUp size={14} />
              )}
            </button>
          </div>
        </div>
      </div>
      <div
        className="card-body"
        ref={containerRef}
        style={{
          padding: "0.5rem",
          position: "relative",
          display: isCollapsed ? "none" : "block",
        }}
      >
        <div style={{ position: "relative" }}>
          <canvas
            ref={canvasRef}
            width={width}
            height={height}
            onClick={handleCanvasClick}
            onMouseMove={handleCanvasMouseMove}
            onMouseLeave={handleCanvasMouseLeave}
            style={{
              border: "1px solid #dee2e6",
              borderRadius: "4px",
              display: "block",
              width: "100%",
              cursor: onFrequencyClick ? "crosshair" : "default",
              filter: isIdle ? "grayscale(80%) opacity(0.6)" : "none",
              transition: "filter 0.3s ease",
            }}
          />
          {/* Paused overlay */}
          {isIdle && (
            <div
              style={{
                position: "absolute",
                top: "50%",
                left: "50%",
                transform: "translate(-50%, -50%)",
                backgroundColor: "rgba(0, 0, 0, 0.7)",
                color: "#ffc107",
                padding: "8px 16px",
                borderRadius: "4px",
                fontSize: "14px",
                fontWeight: 700,
                textTransform: "uppercase",
                letterSpacing: "1px",
                pointerEvents: "none",
                zIndex: 10,
              }}
            >
              Paused
            </div>
          )}
        </div>
        {/* Frequency tooltip */}
        {hoverFrequency !== null && hoverPosition !== null && !isIdle && (
          <div
            style={{
              position: "fixed",
              left: `${hoverPosition.x + 10}px`,
              top: `${hoverPosition.y - 30}px`,
              backgroundColor: "rgba(0, 0, 0, 0.85)",
              color: "white",
              padding: "4px 8px",
              borderRadius: "4px",
              fontSize: "12px",
              fontWeight: 600,
              fontFamily: "monospace",
              pointerEvents: "none",
              zIndex: 1000,
              whiteSpace: "nowrap",
              boxShadow: "0 2px 8px rgba(0,0,0,0.3)",
            }}
          >
            {formatFrequencyWithUnit(hoverFrequency, 4)}
            {onFrequencyClick && (
              <div style={{ fontSize: "10px", opacity: 0.8, marginTop: "2px" }}>
                Click to tune
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
