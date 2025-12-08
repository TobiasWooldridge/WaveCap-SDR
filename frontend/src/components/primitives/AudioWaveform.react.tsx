import { useRef, useEffect, useCallback } from "react";

interface AudioWaveformProps {
  channelId: string;
  isPlaying: boolean;
  height?: number;
}

/**
 * Real-time audio waveform visualizer for a channel.
 * Streams PCM audio and displays the waveform.
 */
export default function AudioWaveform({
  channelId,
  isPlaying,
  height = 40,
}: AudioWaveformProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number | null>(null);
  const audioBufferRef = useRef<Float32Array>(new Float32Array(512));
  const writeIndexRef = useRef(0);
  const readerRef = useRef<ReadableStreamDefaultReader<Uint8Array> | null>(null);
  const shouldStreamRef = useRef(false);
  const widthRef = useRef(200);

  // Draw the waveform
  const drawWaveform = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const buffer = audioBufferRef.current;
    const dpr = window.devicePixelRatio || 1;
    const width = widthRef.current;

    // Clear canvas
    ctx.fillStyle = "#1a1a2e";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw center line
    ctx.strokeStyle = "#333";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, canvas.height / 2);
    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();

    // Draw waveform
    ctx.strokeStyle = "#00ff88";
    ctx.lineWidth = 1.5 * dpr;
    ctx.beginPath();

    const samplesPerPixel = Math.max(1, Math.floor(buffer.length / (width * dpr)));
    const centerY = canvas.height / 2;
    const amplitude = (canvas.height / 2) * 0.9;

    for (let x = 0; x < canvas.width; x++) {
      const sampleIndex = Math.floor((x / canvas.width) * buffer.length);

      // Find min/max in this pixel's sample range for better visualization
      let min = 0, max = 0;
      for (let i = 0; i < samplesPerPixel && sampleIndex + i < buffer.length; i++) {
        const sample = buffer[(sampleIndex + i) % buffer.length];
        if (sample < min) min = sample;
        if (sample > max) max = sample;
      }

      const y1 = centerY - min * amplitude;
      const y2 = centerY - max * amplitude;

      if (x === 0) {
        ctx.moveTo(x, y1);
      }
      ctx.lineTo(x, y1);
      ctx.lineTo(x, y2);
    }

    ctx.stroke();

    // Draw level indicator bars on sides
    const rms = Math.sqrt(buffer.reduce((sum, s) => sum + s * s, 0) / buffer.length);
    const levelHeight = Math.min(rms * 5, 1) * canvas.height;

    ctx.fillStyle = rms > 0.3 ? "#ff4444" : rms > 0.1 ? "#ffaa00" : "#00ff88";
    ctx.fillRect(0, canvas.height - levelHeight, 3 * dpr, levelHeight);
    ctx.fillRect(canvas.width - 3 * dpr, canvas.height - levelHeight, 3 * dpr, levelHeight);

    animationRef.current = requestAnimationFrame(drawWaveform);
  }, []);

  // Start streaming audio data
  const startStreaming = useCallback(async () => {
    // Cancel any existing stream first
    if (readerRef.current) {
      try {
        await readerRef.current.cancel();
      } catch {}
      readerRef.current = null;
    }

    shouldStreamRef.current = true;

    // Clear buffer for fresh start
    audioBufferRef.current.fill(0);
    writeIndexRef.current = 0;

    try {
      const streamUrl = `${window.location.origin}/api/v1/stream/channels/${channelId}.pcm`;
      const response = await fetch(streamUrl);

      if (!response.ok || !response.body) {
        console.error("Failed to fetch audio stream for waveform");
        return;
      }

      const reader = response.body.getReader();
      readerRef.current = reader;

      const processStream = async () => {
        while (shouldStreamRef.current && readerRef.current === reader) {
          try {
            const { done, value } = await reader.read();
            if (done) break;

            // Convert 16-bit PCM to float samples
            const dataView = new DataView(value.buffer, value.byteOffset, value.byteLength);
            const sampleCount = Math.floor(value.length / 2);

            for (let i = 0; i < sampleCount; i++) {
              const sample = dataView.getInt16(i * 2, true) / 32768.0;
              audioBufferRef.current[writeIndexRef.current] = sample;
              writeIndexRef.current = (writeIndexRef.current + 1) % audioBufferRef.current.length;
            }
          } catch (e) {
            if (shouldStreamRef.current && readerRef.current === reader) {
              console.error("Stream read error:", e);
            }
            break;
          }
        }
      };

      processStream();
    } catch (error) {
      console.error("Failed to start waveform stream:", error);
    }
  }, [channelId]);

  // Stop streaming
  const stopStreaming = useCallback(() => {
    shouldStreamRef.current = false;

    if (readerRef.current) {
      readerRef.current.cancel().catch(() => {});
      readerRef.current = null;
    }

    // Clear the buffer
    audioBufferRef.current.fill(0);
    writeIndexRef.current = 0;
  }, []);

  // Handle play state changes
  useEffect(() => {
    if (isPlaying) {
      startStreaming();
      animationRef.current = requestAnimationFrame(drawWaveform);
    } else {
      stopStreaming();
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
        animationRef.current = null;
      }
      // Draw empty waveform
      const canvas = canvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext("2d");
        if (ctx) {
          ctx.fillStyle = "#1a1a2e";
          ctx.fillRect(0, 0, canvas.width, canvas.height);
          ctx.strokeStyle = "#333";
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(0, canvas.height / 2);
          ctx.lineTo(canvas.width, canvas.height / 2);
          ctx.stroke();
        }
      }
    }

    return () => {
      stopStreaming();
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isPlaying, startStreaming, stopStreaming, drawWaveform]);

  // Set up canvas with proper DPR and handle resize
  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas) return;

    const updateSize = () => {
      const width = container.offsetWidth;
      widthRef.current = width;

      const dpr = window.devicePixelRatio || 1;
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;

      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.scale(dpr, dpr);
      }
    };

    updateSize();

    const resizeObserver = new ResizeObserver(updateSize);
    resizeObserver.observe(container);

    return () => resizeObserver.disconnect();
  }, [height]);

  return (
    <div ref={containerRef} style={{ width: "100%" }}>
      <canvas
        ref={canvasRef}
        style={{
          width: "100%",
          height: `${height}px`,
          borderRadius: "4px",
          display: "block",
        }}
      />
    </div>
  );
}
