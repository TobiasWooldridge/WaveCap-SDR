import { useEffect, useRef, useState } from "react";
import type { Capture, Channel } from "../../types";

export interface SpectrumAnalyzerProps {
  capture: Capture;
  channels?: Channel[];
  height?: number;
}

interface SpectrumData {
  power: number[];
  freqs: number[];
  centerHz: number;
  sampleRate: number;
}

export default function SpectrumAnalyzer({
  capture,
  channels = [],
  height = 200,
}: SpectrumAnalyzerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [spectrumData, setSpectrumData] = useState<SpectrumData | null>(null);
  const [width, setWidth] = useState(800);

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

  // Connect to WebSocket and receive FFT data
  useEffect(() => {
    if (capture.state !== "running") {
      // Disconnect if capture is not running
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      setIsConnected(false);
      setSpectrumData(null); // Clear spectrum data when stopped
      return;
    }

    // Create WebSocket connection
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/api/v1/stream/captures/${capture.id}/spectrum`;

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log("Spectrum WebSocket connected");
      setIsConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const data: SpectrumData = JSON.parse(event.data);
        setSpectrumData(data);
      } catch (error) {
        console.error("Error parsing spectrum data:", error);
      }
    };

    ws.onerror = (error) => {
      console.error("Spectrum WebSocket error:", error);
    };

    ws.onclose = () => {
      console.log("Spectrum WebSocket disconnected");
      setIsConnected(false);
    };

    // Cleanup on unmount
    return () => {
      if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
        ws.close();
      }
      wsRef.current = null;
    };
  }, [capture.id, capture.state]);

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

    // If no spectrum data (capture stopped), show graceful message
    if (!spectrumData) {
      // Draw muted grid
      ctx.strokeStyle = "#e9ecef";
      ctx.lineWidth = 1;

      // Horizontal grid lines
      for (let i = 0; i <= 4; i++) {
        const y = (i / 4) * height;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }

      // Vertical grid lines
      for (let i = 0; i <= 8; i++) {
        const x = (i / 8) * width;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }

      // Draw center line in very muted color
      const centerX = width / 2;
      ctx.strokeStyle = "#dee2e6";
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(centerX, 0);
      ctx.lineTo(centerX, height);
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw "Capture Stopped" message
      ctx.fillStyle = "#6c757d";
      ctx.font = "14px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("Capture Stopped", width / 2, height / 2 - 10);

      ctx.font = "12px sans-serif";
      ctx.fillStyle = "#adb5bd";
      ctx.fillText("Click Start to begin capturing spectrum", width / 2, height / 2 + 10);

      return;
    }

    const { power, freqs, centerHz } = spectrumData;
    if (power.length === 0) {
      return;
    }

    // Find min/max for scaling
    const minPower = Math.min(...power);
    const maxPower = Math.max(...power);
    const powerRange = maxPower - minPower;

    // Draw spectrum with primary blue color
    ctx.strokeStyle = "#0d6efd";
    ctx.lineWidth = 2;
    ctx.beginPath();

    for (let i = 0; i < power.length; i++) {
      const x = (i / power.length) * width;
      // Normalize power to canvas height (invert y-axis)
      const normalized = (power[i] - minPower) / (powerRange || 1);
      const y = height - normalized * height;

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }

    ctx.stroke();

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

    // Draw frequency labels
    ctx.fillStyle = "#6c757d";
    ctx.font = "10px monospace";
    const freqMin = centerHz + freqs[0];
    const freqMax = centerHz + freqs[freqs.length - 1];
    const freqMid = centerHz;

    ctx.fillText(`${(freqMin / 1e6).toFixed(3)} MHz`, 5, height - 5);
    ctx.fillText(`${(freqMid / 1e6).toFixed(3)} MHz`, width / 2 - 30, height - 5);
    ctx.fillText(`${(freqMax / 1e6).toFixed(3)} MHz`, width - 70, height - 5);

    // Draw power labels
    ctx.fillText(`${maxPower.toFixed(1)} dB`, 5, 12);
    ctx.fillText(`${minPower.toFixed(1)} dB`, 5, height - 15);

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

          // Draw channel label at top
          ctx.fillStyle = channel.state === "running" ? "#198754" : "#6c757d";
          ctx.font = "bold 10px monospace";
          const label = `CH${idx + 1}`;
          const labelWidth = ctx.measureText(label).width;
          ctx.fillText(label, x - labelWidth / 2, 10);
        }
      });
    }
  }, [spectrumData, width, height, channels]);

  return (
    <div className="card shadow-sm">
      <div className="card-header bg-body-tertiary">
        <div className="d-flex justify-content-between align-items-center">
          <h3 className="h6 mb-0">Spectrum Analyzer</h3>
          <span
            className={`badge ${isConnected ? "bg-success" : "bg-secondary"} text-white`}
            style={{ fontSize: "9px" }}
          >
            {isConnected ? "LIVE" : "OFFLINE"}
          </span>
        </div>
      </div>
      <div className="card-body" ref={containerRef} style={{ padding: "1rem" }}>
        <canvas
          ref={canvasRef}
          width={width}
          height={height}
          style={{
            border: "1px solid #dee2e6",
            borderRadius: "4px",
            display: "block",
            width: "100%",
          }}
        />
      </div>
    </div>
  );
}
