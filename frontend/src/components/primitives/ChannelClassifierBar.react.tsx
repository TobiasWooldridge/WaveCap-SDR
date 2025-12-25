import { useEffect, useRef, useState, useCallback } from "react";
import { Radio, RotateCcw, ChevronDown, ChevronUp } from "lucide-react";
import type { Capture } from "../../types";
import { useSpectrumData } from "../../hooks/useSpectrumData";

interface ChannelClassifierBarProps {
  capture: Capture;
  height?: number;
}

interface FrequencyBinStats {
  sum: number;
  sumSq: number;
  count: number;
  min: number;
  max: number;
}

interface ClassifiedChannel {
  freqHz: number;
  power: number;
  variance: number;
  type: "control" | "voice" | "variable" | "noise";
}

const MIN_COLLECTION_SECONDS = 60; // Need at least 60 seconds of data before classifying
const MIN_SAMPLES_PER_BIN = 50; // Minimum samples per bin for reliable stats
const NOISE_THRESHOLD_DB = -50; // Below this is noise
const CONTROL_VARIANCE_THRESHOLD = 4; // Low variance = control channel
const VOICE_VARIANCE_THRESHOLD = 10; // High variance = voice channel

export default function ChannelClassifierBar({
  capture,
  height = 50,
}: ChannelClassifierBarProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(800);

  // Track statistics per bin
  const binStatsRef = useRef<Map<number, FrequencyBinStats>>(new Map());
  const sampleCountRef = useRef(0);
  const collectionStartRef = useRef<number | null>(null);

  // Track capture parameters to detect changes
  const prevCaptureParamsRef = useRef<string>("");

  // Classified channels for display
  const [classifiedChannels, setClassifiedChannels] = useState<ClassifiedChannel[]>([]);
  const [isCollecting, setIsCollecting] = useState(true);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  // Hover state for tooltip
  const [hoveredChannel, setHoveredChannel] = useState<ClassifiedChannel | null>(null);
  const [mousePos, setMousePos] = useState<{ x: number; y: number } | null>(null);

  // Expanded list view
  const [isListExpanded, setIsListExpanded] = useState(false);

  // Use shared WebSocket connection
  const { spectrumData } = useSpectrumData(capture, false);

  // Spectrum info for frequency mapping
  const [spectrumInfo, setSpectrumInfo] = useState<{
    centerHz: number;
    freqs: number[];
    sampleRate: number;
  } | null>(null);

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

  // Reset when capture parameters change
  useEffect(() => {
    const currentParams = `${capture.id}:${capture.centerHz}:${capture.sampleRate}`;
    if (prevCaptureParamsRef.current !== currentParams) {
      // Parameters changed - reset statistics
      binStatsRef.current = new Map();
      sampleCountRef.current = 0;
      collectionStartRef.current = null;
      setClassifiedChannels([]);
      setIsCollecting(true);
      setElapsedSeconds(0);
      prevCaptureParamsRef.current = currentParams;
    }
  }, [capture.id, capture.centerHz, capture.sampleRate]);

  // Classify channels based on accumulated statistics
  const classifyChannels = useCallback(() => {
    if (!spectrumInfo) return;

    const { centerHz, freqs } = spectrumInfo;

    // Calculate noise floor from 20th percentile of averages
    const averages: number[] = [];
    binStatsRef.current.forEach((stats) => {
      if (stats.count > 0) {
        averages.push(stats.sum / stats.count);
      }
    });
    averages.sort((a, b) => a - b);
    const noiseFloor = averages[Math.floor(averages.length * 0.2)] ?? -60;
    const signalThreshold = noiseFloor + 10;

    // Find peaks and classify them
    const classified: ClassifiedChannel[] = [];
    const visited = new Set<number>();

    binStatsRef.current.forEach((stats, binIdx) => {
      if (visited.has(binIdx) || stats.count < MIN_SAMPLES_PER_BIN) return;

      const avg = stats.sum / stats.count;
      const variance = (stats.sumSq / stats.count) - (avg * avg);
      const stdDev = Math.sqrt(Math.max(0, variance));

      // Only consider signals above threshold
      if (avg < signalThreshold) return;

      // Check if this is a local peak
      const prevStats = binStatsRef.current.get(binIdx - 1);
      const nextStats = binStatsRef.current.get(binIdx + 1);
      const prevAvg = prevStats ? prevStats.sum / prevStats.count : -Infinity;
      const nextAvg = nextStats ? nextStats.sum / nextStats.count : -Infinity;

      if (avg <= prevAvg || avg <= nextAvg) return;

      // Mark nearby bins as visited
      for (let offset = -3; offset <= 3; offset++) {
        visited.add(binIdx + offset);
      }

      // Calculate frequency
      const freqHz = centerHz + (freqs[binIdx] ?? 0);

      // Classify based on variance
      let type: "control" | "voice" | "variable" | "noise";
      if (avg < noiseFloor + 5) {
        type = "noise";
      } else if (stdDev < CONTROL_VARIANCE_THRESHOLD) {
        type = "control";
      } else if (stdDev > VOICE_VARIANCE_THRESHOLD) {
        type = "voice";
      } else {
        type = "variable";
      }

      classified.push({
        freqHz,
        power: avg,
        variance: stdDev,
        type,
      });
    });

    // Sort by power (strongest first)
    classified.sort((a, b) => b.power - a.power);
    setClassifiedChannels(classified);
  }, [spectrumInfo]);

  // Handle incoming spectrum data
  useEffect(() => {
    if (!spectrumData || capture.state !== "running") {
      return;
    }

    const power = spectrumData.power;
    if (!power || power.length === 0) return;

    // Start collection timer on first sample
    if (collectionStartRef.current === null) {
      collectionStartRef.current = Date.now();
    }

    // Update spectrum info
    setSpectrumInfo({
      centerHz: spectrumData.centerHz,
      freqs: spectrumData.freqs,
      sampleRate: spectrumData.sampleRate,
    });

    // Update statistics for each bin
    for (let i = 0; i < power.length; i++) {
      const p = power[i];
      let stats = binStatsRef.current.get(i);

      if (!stats) {
        stats = { sum: 0, sumSq: 0, count: 0, min: Infinity, max: -Infinity };
        binStatsRef.current.set(i, stats);
      }

      stats.sum += p;
      stats.sumSq += p * p;
      stats.count += 1;
      if (p < stats.min) stats.min = p;
      if (p > stats.max) stats.max = p;
    }

    sampleCountRef.current += 1;

    // Calculate elapsed time
    const elapsed = (Date.now() - collectionStartRef.current) / 1000;
    const hasEnoughData = elapsed >= MIN_COLLECTION_SECONDS;

    // Update elapsed time for display (every second)
    const newElapsedSeconds = Math.floor(elapsed);
    setElapsedSeconds(newElapsedSeconds);

    // Classify channels periodically (every 10 samples) once we have enough time
    if (sampleCountRef.current % 10 === 0 && hasEnoughData) {
      classifyChannels();
    }

    // Update collecting state
    setIsCollecting(!hasEnoughData);
  }, [spectrumData, capture.state, classifyChannels]);

  // Manual reset
  const handleReset = useCallback(() => {
    binStatsRef.current = new Map();
    sampleCountRef.current = 0;
    collectionStartRef.current = null;
    setClassifiedChannels([]);
    setIsCollecting(true);
    setElapsedSeconds(0);
  }, []);

  // Handle mouse move over canvas
  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!spectrumInfo || classifiedChannels.length === 0) {
      setHoveredChannel(null);
      return;
    }

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;

    const { centerHz, freqs } = spectrumInfo;
    const freqMin = centerHz + freqs[0];
    const freqMax = centerHz + freqs[freqs.length - 1];
    const freqSpan = freqMax - freqMin;

    // Find nearest channel within 10 pixels
    let nearestChannel: ClassifiedChannel | null = null;
    let nearestDist = 15; // Max distance in pixels

    classifiedChannels.forEach((ch) => {
      const chX = ((ch.freqHz - freqMin) / freqSpan) * width;
      const dist = Math.abs(chX - x);
      if (dist < nearestDist) {
        nearestDist = dist;
        nearestChannel = ch;
      }
    });

    setHoveredChannel(nearestChannel);
    setMousePos(nearestChannel ? { x: e.clientX, y: e.clientY } : null);
  }, [spectrumInfo, classifiedChannels, width]);

  const handleMouseLeave = useCallback(() => {
    setHoveredChannel(null);
    setMousePos(null);
  }, []);

  // Render the classifier bar
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !spectrumInfo) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const { centerHz, freqs } = spectrumInfo;
    const freqMin = centerHz + freqs[0];
    const freqMax = centerHz + freqs[freqs.length - 1];
    const freqSpan = freqMax - freqMin;

    // Clear canvas
    ctx.fillStyle = "#1a1a1a";
    ctx.fillRect(0, 0, width, height);

    const barHeight = height - 18; // Leave room for legend

    // Draw background gradient showing variance density
    if (binStatsRef.current.size > 0 && !isCollecting) {
      const imageData = ctx.createImageData(width, barHeight);
      const data = imageData.data;

      for (let x = 0; x < width; x++) {
        // Map x to bin index
        const binIdx = Math.floor((x / width) * binStatsRef.current.size);
        const stats = binStatsRef.current.get(binIdx);

        let r = 30, g = 30, b = 30;

        if (stats && stats.count >= MIN_SAMPLES_PER_BIN) {
          const avg = stats.sum / stats.count;
          const variance = (stats.sumSq / stats.count) - (avg * avg);
          const stdDev = Math.sqrt(Math.max(0, variance));

          // Color based on classification
          if (avg < NOISE_THRESHOLD_DB) {
            // Noise - dark
            r = 30; g = 30; b = 30;
          } else if (stdDev < CONTROL_VARIANCE_THRESHOLD) {
            // Control channel - green
            const intensity = Math.min(1, (avg + 60) / 60);
            r = 0; g = Math.floor(80 + 100 * intensity); b = 0;
          } else if (stdDev > VOICE_VARIANCE_THRESHOLD) {
            // Voice channel - red/orange
            const intensity = Math.min(1, (avg + 60) / 60);
            r = Math.floor(150 * intensity); g = Math.floor(60 * intensity); b = 0;
          } else {
            // Variable - blue
            const intensity = Math.min(1, (avg + 60) / 60);
            r = 0; g = Math.floor(80 * intensity); b = Math.floor(150 * intensity);
          }
        }

        // Fill column
        for (let y = 0; y < barHeight; y++) {
          const idx = (y * width + x) * 4;
          data[idx] = r;
          data[idx + 1] = g;
          data[idx + 2] = b;
          data[idx + 3] = 255;
        }
      }

      ctx.putImageData(imageData, 0, 0);
    }

    // Draw channel markers for detected channels
    classifiedChannels.forEach((ch) => {
      const x = ((ch.freqHz - freqMin) / freqSpan) * width;

      if (x < 0 || x > width) return;

      // Draw marker
      let markerColor: string;
      switch (ch.type) {
        case "control":
          markerColor = "#00ff00";
          break;
        case "voice":
          markerColor = "#ff6600";
          break;
        case "variable":
          markerColor = "#0088ff";
          break;
        default:
          markerColor = "#666666";
      }

      // Highlight if hovered
      const isHovered = hoveredChannel?.freqHz === ch.freqHz;
      if (isHovered) {
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 4;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, barHeight);
        ctx.stroke();
      }

      ctx.strokeStyle = markerColor;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, barHeight);
      ctx.stroke();

      // Draw triangle at bottom
      ctx.fillStyle = markerColor;
      ctx.beginPath();
      ctx.moveTo(x, barHeight);
      ctx.lineTo(x - 5, barHeight + 6);
      ctx.lineTo(x + 5, barHeight + 6);
      ctx.closePath();
      ctx.fill();
    });

    // Draw legend at bottom
    ctx.font = "9px monospace";
    const legendY = height - 3;

    ctx.fillStyle = "#00ff00";
    ctx.fillRect(5, legendY - 7, 8, 8);
    ctx.fillStyle = "#ffffff";
    ctx.fillText("Control", 16, legendY);

    ctx.fillStyle = "#ff6600";
    ctx.fillRect(70, legendY - 7, 8, 8);
    ctx.fillStyle = "#ffffff";
    ctx.fillText("Voice", 81, legendY);

    ctx.fillStyle = "#0088ff";
    ctx.fillRect(120, legendY - 7, 8, 8);
    ctx.fillStyle = "#ffffff";
    ctx.fillText("Variable", 131, legendY);

    // Show collection status
    const remainingSeconds = Math.max(0, MIN_COLLECTION_SECONDS - elapsedSeconds);
    const statusText = isCollecting
      ? `Collecting... ${remainingSeconds}s remaining`
      : `${classifiedChannels.length} signals (${elapsedSeconds}s)`;

    ctx.fillStyle = "#888888";
    ctx.textAlign = "right";
    ctx.fillText(statusText, width - 5, legendY);
    ctx.textAlign = "left";

  }, [width, height, spectrumInfo, classifiedChannels, isCollecting, elapsedSeconds, hoveredChannel]);

  // Count by type
  const controlCount = classifiedChannels.filter(c => c.type === "control").length;
  const voiceCount = classifiedChannels.filter(c => c.type === "voice").length;

  // Format frequency for display
  const formatFreq = (hz: number) => `${(hz / 1e6).toFixed(4)} MHz`;

  // Get type label
  const getTypeLabel = (type: string) => {
    switch (type) {
      case "control": return "Control";
      case "voice": return "Voice";
      case "variable": return "Variable";
      default: return type;
    }
  };

  // Get type color
  const getTypeColor = (type: string) => {
    switch (type) {
      case "control": return "#00ff00";
      case "voice": return "#ff6600";
      case "variable": return "#0088ff";
      default: return "#888888";
    }
  };

  return (
    <div className="card shadow-sm mt-2">
      <div className="card-header bg-body-tertiary py-1 px-2">
        <div className="d-flex justify-content-between align-items-center">
          <small className="fw-semibold mb-0 d-flex align-items-center gap-1">
            <Radio size={12} />
            Channel Classifier
            {!isCollecting && (
              <span className="badge bg-success text-white ms-2" style={{ fontSize: "8px" }}>
                {controlCount} CC / {voiceCount} VC
              </span>
            )}
          </small>
          <div className="d-flex align-items-center gap-1">
            {!isCollecting && classifiedChannels.length > 0 && (
              <button
                className="btn btn-sm btn-outline-secondary p-0 d-flex align-items-center justify-content-center"
                style={{ width: "20px", height: "20px" }}
                onClick={() => setIsListExpanded(!isListExpanded)}
                title={isListExpanded ? "Hide channel list" : "Show channel list"}
              >
                {isListExpanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
              </button>
            )}
            <button
              className="btn btn-sm btn-outline-secondary p-0 d-flex align-items-center justify-content-center"
              style={{ width: "20px", height: "20px" }}
              onClick={handleReset}
              title="Reset classification"
            >
              <RotateCcw size={12} />
            </button>
          </div>
        </div>
      </div>
      <div className="card-body p-1" ref={containerRef}>
        <div style={{ position: "relative" }}>
          <canvas
            ref={canvasRef}
            width={width}
            height={height}
            style={{
              border: "1px solid #dee2e6",
              borderRadius: "4px",
              display: "block",
              width: "100%",
              cursor: classifiedChannels.length > 0 ? "crosshair" : "default",
            }}
            onMouseMove={handleMouseMove}
            onMouseLeave={handleMouseLeave}
          />

          {/* Tooltip */}
          {hoveredChannel && mousePos && (
            <div
              style={{
                position: "fixed",
                left: mousePos.x + 10,
                top: mousePos.y - 60,
                backgroundColor: "rgba(0, 0, 0, 0.9)",
                color: "#ffffff",
                padding: "8px 12px",
                borderRadius: "4px",
                fontSize: "11px",
                fontFamily: "monospace",
                zIndex: 1000,
                pointerEvents: "none",
                border: `2px solid ${getTypeColor(hoveredChannel.type)}`,
                minWidth: "180px",
              }}
            >
              <div style={{ color: getTypeColor(hoveredChannel.type), fontWeight: "bold", marginBottom: "4px" }}>
                {getTypeLabel(hoveredChannel.type)} Channel
              </div>
              <div><strong>Frequency:</strong> {formatFreq(hoveredChannel.freqHz)}</div>
              <div><strong>Power:</strong> {hoveredChannel.power.toFixed(1)} dB</div>
              <div><strong>Variance:</strong> {hoveredChannel.variance.toFixed(2)} dB</div>
            </div>
          )}
        </div>

        {/* Expandable channel list */}
        {isListExpanded && classifiedChannels.length > 0 && (
          <div
            className="mt-1"
            style={{
              maxHeight: "200px",
              overflowY: "auto",
              fontSize: "10px",
              fontFamily: "monospace",
            }}
          >
            <table className="table table-sm table-dark mb-0" style={{ fontSize: "10px" }}>
              <thead>
                <tr>
                  <th style={{ width: "60px" }}>Type</th>
                  <th>Frequency</th>
                  <th style={{ width: "60px" }}>Power</th>
                  <th style={{ width: "60px" }}>Std Dev</th>
                </tr>
              </thead>
              <tbody>
                {classifiedChannels.map((ch, idx) => (
                  <tr key={idx}>
                    <td>
                      <span
                        className="badge"
                        style={{
                          backgroundColor: getTypeColor(ch.type),
                          color: ch.type === "control" ? "#000" : "#fff",
                          fontSize: "9px",
                        }}
                      >
                        {getTypeLabel(ch.type)}
                      </span>
                    </td>
                    <td>{formatFreq(ch.freqHz)}</td>
                    <td>{ch.power.toFixed(1)} dB</td>
                    <td>{ch.variance.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
