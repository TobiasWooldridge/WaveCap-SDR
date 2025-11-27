import { useEffect, useRef, useState, useCallback } from "react";
import { ChevronUp, ChevronDown, Minus, Plus } from "lucide-react";
import type { Capture, Channel } from "../../types";
import { useSpectrumData } from "../../hooks/useSpectrumData";
import { getCaptureStatusMessage, drawCaptureStatusOnCanvas } from "../../utils/captureStatus";

export interface WaterfallDisplayProps {
  capture: Capture;
  channels?: Channel[];
  height?: number;
  timeSpanSeconds?: number;
  colorScheme?: "heat" | "grayscale" | "viridis";
  intensity?: number;
}

const MIN_HEIGHT = 100;
const MAX_HEIGHT = 600;
const HEIGHT_STEP = 25;
const STORAGE_KEY = "waterfallDisplayHeight";

// Color schemes for waterfall
const COLOR_SCHEMES = {
  heat: [
    { pos: 0.0, r: 0, g: 0, b: 128 },      // Dark blue (low power)
    { pos: 0.25, r: 0, g: 0, b: 255 },    // Blue
    { pos: 0.5, r: 0, g: 255, b: 255 },   // Cyan
    { pos: 0.75, r: 255, g: 255, b: 0 },  // Yellow
    { pos: 1.0, r: 255, g: 0, b: 0 },     // Red (high power)
  ],
  grayscale: [
    { pos: 0.0, r: 0, g: 0, b: 0 },       // Black (low power)
    { pos: 1.0, r: 255, g: 255, b: 255 }, // White (high power)
  ],
  viridis: [
    { pos: 0.0, r: 68, g: 1, b: 84 },     // Dark purple
    { pos: 0.25, r: 59, g: 82, b: 139 },  // Blue
    { pos: 0.5, r: 33, g: 145, b: 140 },  // Teal
    { pos: 0.75, r: 94, g: 201, b: 98 },  // Green
    { pos: 1.0, r: 253, g: 231, b: 37 },  // Yellow
  ],
};

export default function WaterfallDisplay({
  capture,
  channels = [],
  height: initialHeight = 300,
  timeSpanSeconds = 30,
  colorScheme = "heat",
  intensity = 1.0,
}: WaterfallDisplayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(800);
  const [isCollapsed, setIsCollapsed] = useState(false);

  // Load height from localStorage or use initial value
  const [height, setHeight] = useState(() => {
    const stored = localStorage.getItem(STORAGE_KEY);
    return stored ? parseInt(stored, 10) : initialHeight;
  });

  // Save height to localStorage when it changes
  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, height.toString());
  }, [height]);

  const increaseHeight = () => {
    setHeight(prev => Math.min(MAX_HEIGHT, prev + HEIGHT_STEP));
  };

  const decreaseHeight = () => {
    setHeight(prev => Math.max(MIN_HEIGHT, prev - HEIGHT_STEP));
  };

  // Use shared WebSocket connection - paused when collapsed
  const { spectrumData, isConnected, isIdle } = useSpectrumData(capture, isCollapsed);

  // Waterfall history buffer (circular buffer)
  const historyRef = useRef<number[][]>([]);
  const maxHistoryLines = height; // One line per pixel
  const [spectrumInfo, setSpectrumInfo] = useState<{
    centerHz: number;
    freqs: number[];
    minPower: number;
    maxPower: number;
  } | null>(null);

  const renderRequestRef = useRef<number | null>(null);
  const needsRenderRef = useRef(false);

  // Update canvas width when container resizes
  useEffect(() => {
    const updateWidth = () => {
      if (containerRef.current) {
        const containerWidth = containerRef.current.offsetWidth;
        setWidth(containerWidth - 16);
      }
    };

    updateWidth();
    window.addEventListener("resize", updateWidth);
    return () => window.removeEventListener("resize", updateWidth);
  }, []);

  // Handle incoming spectrum data
  useEffect(() => {
    if (!spectrumData) {
      if (capture.state !== "running") {
        historyRef.current = []; // Clear history when stopped
        setSpectrumInfo(null);
        needsRenderRef.current = true;
      }
      return;
    }

    // Add new FFT line to history (circular buffer)
    historyRef.current.push(spectrumData.power);
    if (historyRef.current.length > maxHistoryLines) {
      historyRef.current.shift(); // Remove oldest line
    }

    // Update spectrum info for frequency labels
    const minPower = Math.min(...spectrumData.power);
    const maxPower = Math.max(...spectrumData.power);
    setSpectrumInfo({
      centerHz: spectrumData.centerHz,
      freqs: spectrumData.freqs,
      minPower,
      maxPower,
    });

    needsRenderRef.current = true;
  }, [spectrumData, capture.state, maxHistoryLines]);

  // Helper function to interpolate color from gradient - memoized for performance
  const getColor = useCallback((normalizedValue: number): [number, number, number] => {
    // Apply intensity adjustment
    const adjustedValue = Math.pow(normalizedValue, 1 / intensity);
    const clampedValue = Math.max(0, Math.min(1, adjustedValue));

    const gradient = COLOR_SCHEMES[colorScheme];

    // Find the two color stops to interpolate between
    let lowerStop = gradient[0];
    let upperStop = gradient[gradient.length - 1];

    for (let i = 0; i < gradient.length - 1; i++) {
      if (clampedValue >= gradient[i].pos && clampedValue <= gradient[i + 1].pos) {
        lowerStop = gradient[i];
        upperStop = gradient[i + 1];
        break;
      }
    }

    // Interpolate between the two stops
    const range = upperStop.pos - lowerStop.pos;
    const t = range === 0 ? 0 : (clampedValue - lowerStop.pos) / range;

    const r = Math.round(lowerStop.r + t * (upperStop.r - lowerStop.r));
    const g = Math.round(lowerStop.g + t * (upperStop.g - lowerStop.g));
    const b = Math.round(lowerStop.b + t * (upperStop.b - lowerStop.b));

    return [r, g, b];
  }, [colorScheme, intensity]);

  // Render loop using requestAnimationFrame for smooth updates
  useEffect(() => {
    const render = () => {
      if (!needsRenderRef.current || !canvasRef.current) {
        renderRequestRef.current = requestAnimationFrame(render);
        return;
      }

      needsRenderRef.current = false;

      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        renderRequestRef.current = requestAnimationFrame(render);
        return;
      }

      // Clear canvas with dark background
      ctx.fillStyle = "#1a1a1a";
      ctx.fillRect(0, 0, width, height);

      const history = historyRef.current;

      if (history.length === 0) {
        // Draw status message using shared utility
        const status = getCaptureStatusMessage(capture, isConnected);
        drawCaptureStatusOnCanvas(ctx, width, height, status, {
          backgroundColor: "#1a1a1a",
          gridColor: "#333333",
          subtitleColor: "#6c757d",
        });

        renderRequestRef.current = requestAnimationFrame(render);
        return;
      }

      // Find global min/max for color scaling across all history
      let globalMin = Infinity;
      let globalMax = -Infinity;
      history.forEach((powerArray) => {
        const min = Math.min(...powerArray);
        const max = Math.max(...powerArray);
        if (min < globalMin) globalMin = min;
        if (max > globalMax) globalMax = max;
      });

      const powerRange = globalMax - globalMin || 1;

      // Create image data for efficient pixel manipulation
      const imageData = ctx.createImageData(width, height);
      const data = imageData.data;

      // Draw waterfall lines from bottom to top (newest at bottom)
      for (let lineIdx = 0; lineIdx < history.length; lineIdx++) {
        const powerArray = history[lineIdx];
        const y = height - 1 - lineIdx; // Draw from bottom up

        for (let x = 0; x < width; x++) {
          // Map x position to frequency bin
          const binIdx = Math.floor((x / width) * powerArray.length);
          const power = powerArray[binIdx];

          // Normalize power to 0-1 range
          const normalized = (power - globalMin) / powerRange;

          // Get color from gradient
          const [r, g, b] = getColor(normalized);

          // Set pixel in image data
          const pixelIdx = (y * width + x) * 4;
          data[pixelIdx] = r;
          data[pixelIdx + 1] = g;
          data[pixelIdx + 2] = b;
          data[pixelIdx + 3] = 255; // Alpha
        }
      }

      // Draw the image data to canvas
      ctx.putImageData(imageData, 0, 0);

      // Helper functions to format rate values
      const formatRate = (rate: number): string => {
        if (rate >= 1e9) return `${(rate / 1e9).toFixed(2)} GS/s`;
        if (rate >= 1e6) return `${(rate / 1e6).toFixed(2)} MS/s`;
        if (rate >= 1e3) return `${(rate / 1e3).toFixed(2)} kS/s`;
        return `${rate.toFixed(0)} S/s`;
      };

      const formatBandwidth = (bw: number): string => {
        if (bw >= 1e9) return `${(bw / 1e9).toFixed(2)} GHz`;
        if (bw >= 1e6) return `${(bw / 1e6).toFixed(2)} MHz`;
        if (bw >= 1e3) return `${(bw / 1e3).toFixed(2)} kHz`;
        return `${bw.toFixed(0)} Hz`;
      };

      // Draw Sample Rate overlay (full width) at top
      if (spectrumInfo && spectrumData) {
        const sampleRateY = height * 0.05;
        const overlayHeight = 15;

        ctx.fillStyle = "rgba(0, 123, 255, 0.15)";
        ctx.fillRect(0, sampleRateY, width, overlayHeight);

        ctx.strokeStyle = "rgba(0, 123, 255, 0.5)";
        ctx.lineWidth = 1;
        ctx.strokeRect(0, sampleRateY, width, overlayHeight);

        ctx.font = "bold 9px monospace";
        const sampleRateText = `Sample Rate: ${formatRate(spectrumData.sampleRate)}`;
        const sampleRateTextWidth = ctx.measureText(sampleRateText).width;
        const sampleRateTextX = (width - sampleRateTextWidth) / 2;

        // Draw white background for text
        ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
        ctx.fillRect(sampleRateTextX - 2, sampleRateY + 2, sampleRateTextWidth + 4, 11);

        // Draw text
        ctx.fillStyle = "#007bff";
        ctx.fillText(sampleRateText, sampleRateTextX, sampleRateY + 10);

        // Draw Bandwidth overlay (if available and centered)
        if (capture.bandwidth !== null && capture.bandwidth !== undefined) {
          const bandwidthY = height * 0.13;
          const { centerHz, freqs } = spectrumInfo;
          const freqMin = centerHz + freqs[0];
          const freqMax = centerHz + freqs[freqs.length - 1];
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
          ctx.fillRect(bandwidthTextX - 2, bandwidthY + 2, bandwidthTextWidth + 4, 11);

          // Draw text
          ctx.fillStyle = "#28a745";
          ctx.fillText(bandwidthText, bandwidthTextX, bandwidthY + 10);
        }
      }

      // Draw channel markers if we have spectrum info
      if (spectrumInfo && channels && channels.length > 0) {
        const { centerHz, freqs } = spectrumInfo;
        const freqMin = centerHz + freqs[0];
        const freqMax = centerHz + freqs[freqs.length - 1];
        const freqSpan = freqMax - freqMin;

        channels.forEach((channel, idx) => {
          const channelFreq = centerHz + channel.offsetHz;
          const freqOffset = channelFreq - freqMin;
          const x = (freqOffset / freqSpan) * width;

          if (x >= 0 && x <= width) {
            // Draw channel marker line
            ctx.strokeStyle = channel.state === "running" ? "#00ff00" : "#808080";
            ctx.lineWidth = 1;
            ctx.setLineDash([3, 2]);
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();
            ctx.setLineDash([]);

            // Draw channel label
            ctx.font = "bold 9px monospace";
            const label = `CH${idx + 1}`;
            const labelWidth = ctx.measureText(label).width;

            // Draw white label background
            ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
            ctx.fillRect(x - labelWidth / 2 - 2, 2, labelWidth + 4, 12);

            // Draw label text
            ctx.fillStyle = channel.state === "running" ? "#198754" : "#6c757d";
            ctx.fillText(label, x - labelWidth / 2, 11);
          }
        });
      }

      // Draw center frequency marker
      const centerX = width / 2;
      ctx.strokeStyle = "rgba(255, 0, 0, 0.5)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(centerX, 0);
      ctx.lineTo(centerX, height);
      ctx.stroke();

      // Draw frequency labels at bottom
      if (spectrumInfo) {
        const { centerHz, freqs } = spectrumInfo;
        const freqMin = centerHz + freqs[0];
        const freqMax = centerHz + freqs[freqs.length - 1];

        ctx.font = "10px monospace";
        const freqMinText = `${(freqMin / 1e6).toFixed(3)} MHz`;
        const freqMidText = `${(centerHz / 1e6).toFixed(3)} MHz`;
        const freqMaxText = `${(freqMax / 1e6).toFixed(3)} MHz`;

        // Measure text widths
        const freqMinWidth = ctx.measureText(freqMinText).width;
        const freqMidWidth = ctx.measureText(freqMidText).width;
        const freqMaxWidth = ctx.measureText(freqMaxText).width;

        // Draw white backgrounds for frequency labels
        ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
        ctx.fillRect(3, height - 16, freqMinWidth + 4, 13);
        ctx.fillRect(width / 2 - 32, height - 16, freqMidWidth + 4, 13);
        ctx.fillRect(width - 72, height - 16, freqMaxWidth + 4, 13);

        // Draw frequency label text
        ctx.fillStyle = "#000000";
        ctx.fillText(freqMinText, 5, height - 5);
        ctx.fillText(freqMidText, width / 2 - 30, height - 5);
        ctx.fillText(freqMaxText, width - 70, height - 5);
      }

      renderRequestRef.current = requestAnimationFrame(render);
    };

    renderRequestRef.current = requestAnimationFrame(render);

    return () => {
      if (renderRequestRef.current !== null) {
        cancelAnimationFrame(renderRequestRef.current);
      }
    };
  }, [width, height, channels, spectrumInfo, colorScheme, intensity, capture.state, capture.bandwidth, getColor]);

  // Determine badge status
  const getBadgeStatus = () => {
    if (isIdle && capture.state === "running") {
      return { text: "PAUSED (IDLE)", className: "bg-warning" };
    } else if (isConnected) {
      return { text: "LIVE", className: "bg-success" };
    } else {
      return { text: "OFFLINE", className: "bg-secondary" };
    }
  };

  const badgeStatus = getBadgeStatus();

  return (
    <div className="card shadow-sm">
      <div className="card-header bg-body-tertiary py-1 px-2">
        <div className="d-flex justify-content-between align-items-center">
          <small className="fw-semibold mb-0">Waterfall Display ({timeSpanSeconds}s)</small>
          <div className="d-flex align-items-center gap-2">
            <span
              className={`badge ${badgeStatus.className} text-white`}
              style={{ fontSize: "8px", padding: "2px 6px" }}
            >
              {badgeStatus.text}
            </span>
            {!isCollapsed && (
              <>
                <button
                  className="btn btn-sm btn-outline-secondary p-0"
                  style={{ width: "20px", height: "20px", lineHeight: 1 }}
                  onClick={decreaseHeight}
                  disabled={height <= MIN_HEIGHT}
                  title="Decrease height"
                >
                  <Minus size={12} />
                </button>
                <span className="small text-muted" style={{ fontSize: "10px", minWidth: "35px", textAlign: "center" }}>
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
              {isCollapsed ? <ChevronDown size={14} /> : <ChevronUp size={14} />}
            </button>
          </div>
        </div>
      </div>
      {!isCollapsed && (
        <div className="card-body" ref={containerRef} style={{ padding: "0.5rem" }}>
          <canvas
            ref={canvasRef}
            width={width}
            height={height}
            style={{
              border: "1px solid #dee2e6",
              borderRadius: "4px",
              display: "block",
              width: "100%",
              imageRendering: "pixelated", // Crisp pixels for waterfall
            }}
          />
        </div>
      )}
    </div>
  );
}
