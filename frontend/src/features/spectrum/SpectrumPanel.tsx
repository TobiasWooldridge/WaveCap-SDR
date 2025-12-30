import { useCallback } from "react";
import { Settings } from "lucide-react";
import type { Capture, Channel } from "../../types";
import { useUpdateCapture } from "../../hooks/useCaptures";
import { useCreateChannel, useStartChannel } from "../../hooks/useChannels";
import { useToast } from "../../hooks/useToast";
import { formatFrequencyWithUnit } from "../../utils/frequency";
import SpectrumAnalyzer from "../../components/primitives/SpectrumAnalyzer.react";
import WaterfallDisplay from "../../components/primitives/WaterfallDisplay.react";
import ChannelClassifierBar from "../../components/primitives/ChannelClassifierBar.react";

const FPS_OPTIONS = [
  { value: 5, label: "5 fps" },
  { value: 10, label: "10 fps" },
  { value: 15, label: "15 fps" },
  { value: 20, label: "20 fps" },
  { value: 30, label: "30 fps" },
  { value: 60, label: "60 fps" },
];

const MAX_FPS_OPTIONS = [
  { value: 15, label: "15 fps" },
  { value: 30, label: "30 fps" },
  { value: 60, label: "60 fps" },
  { value: 120, label: "120 fps" },
];
const FFT_SIZE_OPTIONS = [
  { value: 512, label: "512 bins" },
  { value: 1024, label: "1024 bins" },
  { value: 2048, label: "2048 bins" },
  { value: 4096, label: "4096 bins" },
];

interface SpectrumPanelProps {
  capture: Capture;
  channels?: Channel[];
}

export function SpectrumPanel({ capture, channels }: SpectrumPanelProps) {
  const updateCapture = useUpdateCapture();
  const createChannel = useCreateChannel();
  const startChannel = useStartChannel(capture.id);
  const toast = useToast();

  const handleFpsChange = (fps: number) => {
    updateCapture.mutate({ captureId: capture.id, request: { fftFps: fps } });
  };

  const handleMaxFpsChange = (maxFps: number) => {
    updateCapture.mutate({ captureId: capture.id, request: { fftMaxFps: maxFps } });
  };

  const handleFftSizeChange = (size: number) => {
    updateCapture.mutate({ captureId: capture.id, request: { fftSize: size } });
  };

  // Handle click on spectrum to tune or create channel
  const handleFrequencyClick = useCallback(
    async (frequencyHz: number) => {
      // Check if clicked on an existing channel
      const existingChannel = channels?.find((ch) => {
        const channelFreq = capture.centerHz + ch.offsetHz;
        // Within 5kHz of the channel
        return Math.abs(channelFreq - frequencyHz) < 5000;
      });

      if (existingChannel) {
        // If clicking near an existing channel, just tune to it
        toast.info(`Channel exists at ${formatFrequencyWithUnit(frequencyHz, 3)}`);
        return;
      }

      // Create a new channel at clicked frequency
      try {
        const result = await createChannel.mutateAsync({
          captureId: capture.id,
          request: {
            mode: "wbfm",
            offsetHz: frequencyHz - capture.centerHz,
            audioRate: 48000,
            squelchDb: -60,
          },
        });

        await startChannel.mutateAsync(result.id);
        toast.success(`Channel created at ${formatFrequencyWithUnit(frequencyHz, 3)}`);
      } catch (error) {
        toast.error("Failed to create channel");
      }
    },
    [capture.id, capture.centerHz, channels, createChannel, startChannel, toast]
  );

  return (
    <div className="d-flex flex-column gap-2">
      {/* FFT Settings Bar */}
      <div
        className="bg-body-tertiary border rounded d-flex align-items-center gap-3 px-2 py-1"
        style={{ fontSize: "11px" }}
      >
        <span className="fw-semibold text-muted d-flex align-items-center gap-1">
          <Settings size={12} />
          Display
        </span>

        <div className="d-flex align-items-center gap-1">
          <label className="text-muted mb-0" title="Target update rate for spectrum/waterfall when not actively viewing.">Target:</label>
          <select
            className="form-select form-select-sm border-0 bg-transparent"
            style={{ width: "auto", fontSize: "11px", padding: "1px 20px 1px 4px" }}
            value={capture.fftFps}
            onChange={(e) => handleFpsChange(parseInt(e.target.value, 10))}
            title="Target FPS. Actual rate adapts based on viewer activity."
          >
            {FPS_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>

        <div className="d-flex align-items-center gap-1">
          <label className="text-muted mb-0" title="Maximum FPS cap. Prevents excessive CPU/bandwidth usage.">Max:</label>
          <select
            className="form-select form-select-sm border-0 bg-transparent"
            style={{ width: "auto", fontSize: "11px", padding: "1px 20px 1px 4px" }}
            value={capture.fftMaxFps}
            onChange={(e) => handleMaxFpsChange(parseInt(e.target.value, 10))}
            title="Hard cap on FPS. Will never exceed this rate."
          >
            {MAX_FPS_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>

        <div className="d-flex align-items-center gap-1">
          <label className="text-muted mb-0" title="FFT bin count. Higher = sharper frequency detail but more CPU.">FFT Size:</label>
          <select
            className="form-select form-select-sm border-0 bg-transparent"
            style={{ width: "auto", fontSize: "11px", padding: "1px 20px 1px 4px" }}
            value={capture.fftSize}
            onChange={(e) => handleFftSizeChange(parseInt(e.target.value, 10))}
            title="FFT bin count. Higher values show sharper frequency detail."
          >
            {FFT_SIZE_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Spectrum Analyzer - its own card */}
      <SpectrumAnalyzer
        capture={capture}
        channels={channels}
        onFrequencyClick={handleFrequencyClick}
      />

      {/* Waterfall Display - its own card */}
      <WaterfallDisplay capture={capture} channels={channels} />

      {/* Channel Classifier Bar - its own card */}
      <ChannelClassifierBar capture={capture} />
    </div>
  );
}
