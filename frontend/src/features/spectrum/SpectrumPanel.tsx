import { useState, useCallback } from "react";
import type { Capture, Channel } from "../../types";
// useUpdateCapture may be used in the future for tuning via spectrum click
import { useCreateChannel, useStartChannel } from "../../hooks/useChannels";
import { useToast } from "../../hooks/useToast";
import SpectrumAnalyzer from "../../components/primitives/SpectrumAnalyzer.react";
import WaterfallDisplay from "../../components/primitives/WaterfallDisplay.react";
import Flex from "../../components/primitives/Flex.react";

interface SpectrumPanelProps {
  capture: Capture;
  channels?: Channel[];
}

export function SpectrumPanel({ capture, channels }: SpectrumPanelProps) {
  const [spectrumHeight, setSpectrumHeight] = useState(() => {
    const saved = localStorage.getItem("spectrum-height");
    return saved ? parseInt(saved) : 200;
  });

  const createChannel = useCreateChannel();
  const startChannel = useStartChannel(capture.id);
  const toast = useToast();

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
        toast.info(`Channel exists at ${(frequencyHz / 1e6).toFixed(3)} MHz`);
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
        toast.success(`Channel created at ${(frequencyHz / 1e6).toFixed(3)} MHz`);
      } catch (error) {
        toast.error("Failed to create channel");
      }
    },
    [capture.id, capture.centerHz, channels, createChannel, startChannel, toast]
  );

  // Save spectrum height to localStorage
  const handleHeightChange = useCallback((height: number) => {
    setSpectrumHeight(height);
    localStorage.setItem("spectrum-height", height.toString());
  }, []);

  const [waterfallHeight, setWaterfallHeight] = useState(() => {
    const saved = localStorage.getItem("waterfall-height");
    return saved ? parseInt(saved) : 200;
  });

  // Save waterfall height to localStorage
  const handleWaterfallHeightChange = useCallback((height: number) => {
    setWaterfallHeight(height);
    localStorage.setItem("waterfall-height", height.toString());
  }, []);

  return (
    <Flex direction="column" gap={0}>
      {/* Spectrum Analyzer */}
      <div>
        <SpectrumAnalyzer
          capture={capture}
          channels={channels}
          height={spectrumHeight}
          onFrequencyClick={handleFrequencyClick}
        />
      </div>

      {/* Resize Handle */}
      <div
        className="bg-secondary"
        style={{
          height: "4px",
          cursor: "ns-resize",
          flexShrink: 0,
        }}
        onMouseDown={(e) => {
          e.preventDefault();
          const startY = e.clientY;
          const startSpectrumHeight = spectrumHeight;
          const startWaterfallHeight = waterfallHeight;

          const handleMouseMove = (moveEvent: MouseEvent) => {
            const deltaY = moveEvent.clientY - startY;
            // Increase spectrum, decrease waterfall (or vice versa)
            const newSpectrumHeight = Math.max(80, Math.min(500, startSpectrumHeight + deltaY));
            const newWaterfallHeight = Math.max(80, Math.min(500, startWaterfallHeight - deltaY));
            handleHeightChange(newSpectrumHeight);
            handleWaterfallHeightChange(newWaterfallHeight);
          };

          const handleMouseUp = () => {
            document.removeEventListener("mousemove", handleMouseMove);
            document.removeEventListener("mouseup", handleMouseUp);
          };

          document.addEventListener("mousemove", handleMouseMove);
          document.addEventListener("mouseup", handleMouseUp);
        }}
      />

      {/* Waterfall Display */}
      <div style={{ height: `${waterfallHeight}px` }}>
        <WaterfallDisplay capture={capture} channels={channels} />
      </div>
    </Flex>
  );
}
