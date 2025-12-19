import { Radio, Cpu, BookOpen, Scan, Bookmark, Sliders } from "lucide-react";
import type { Capture, Device } from "../../types";
import { useChannels } from "../../hooks/useChannels";
import { useMemoryBanks } from "../../hooks/useMemoryBanks";
import { useUpdateCapture } from "../../hooks/useCaptures";
import { useCreateChannel } from "../../hooks/useChannels";
import { formatFrequencyMHz } from "../../utils/frequency";
import { getDeviceDisplayName } from "../../utils/device";
import { AccordionGroup, AccordionItem } from "../../components/primitives/Accordion.react";
import { TuningAccordions } from "./TuningControls";
import { DeviceControlsContent } from "./DeviceControls";
import { AdvancedSettingsContent } from "./AdvancedSettings";
import { RecipeSelectorContent } from "./RecipeSelector";
import ScannerControl from "../../components/ScannerControl.react";
import { BookmarkManager } from "../../components/BookmarkManager.react";
import Flex from "../../components/primitives/Flex.react";
import Spinner from "../../components/primitives/Spinner.react";

interface RadioPanelProps {
  capture: Capture;
  device: Device | undefined;
}

export function RadioPanel({ capture, device }: RadioPanelProps) {
  const { data: channels } = useChannels(capture.id);
  const { getMemoryBank } = useMemoryBanks();
  const updateCapture = useUpdateCapture();
  const createChannel = useCreateChannel();

  // Handle loading a memory bank
  const handleLoadMemoryBank = (bankId: string) => {
    const bank = getMemoryBank(bankId);
    if (!bank) return;

    // Update capture configuration
    updateCapture.mutate({
      captureId: capture.id,
      request: {
        centerHz: bank.captureConfig.centerHz,
        sampleRate: bank.captureConfig.sampleRate,
        gain: bank.captureConfig.gain ?? undefined,
        bandwidth: bank.captureConfig.bandwidth ?? undefined,
        ppm: bank.captureConfig.ppm ?? undefined,
        antenna: bank.captureConfig.antenna ?? undefined,
      },
    });

    // Recreate channels from memory bank
    bank.channels.forEach((channelConfig) => {
      createChannel.mutate({
        captureId: capture.id,
        request: {
          mode: channelConfig.mode,
          offsetHz: channelConfig.offsetHz,
          audioRate: channelConfig.audioRate,
          squelchDb: channelConfig.squelchDb,
          name: channelConfig.name,
        },
      });
    });
  };

  // Handle tuning to a frequency from bookmarks
  const handleTuneToFrequency = (freq: number) => {
    updateCapture.mutate({
      captureId: capture.id,
      request: { centerHz: freq },
    });
  };

  const isRunning = capture.state === "running";
  const hasError = capture.state === "failed" || capture.state === "error";

  // Status badge
  const statusBadge = (
    <span
      className={`badge ${
        hasError
          ? "bg-danger"
          : isRunning
          ? "bg-success"
          : capture.state === "starting" || capture.state === "stopping"
          ? "bg-warning text-dark"
          : "bg-secondary"
      }`}
      style={{ fontSize: "0.65rem" }}
    >
      {capture.state.toUpperCase()}
    </span>
  );

  return (
    <Flex direction="column" gap={1} style={{ padding: "0.5rem" }}>
      {/* Main header - always visible */}
      <div className="d-flex align-items-center gap-2 p-2 bg-dark text-white rounded">
        <Radio size={18} className="flex-shrink-0" />
        <div className="d-flex flex-column flex-grow-1 overflow-hidden">
          <span className="fw-bold" style={{ fontSize: "1.1rem" }}>
            {formatFrequencyMHz(capture.centerHz)} MHz
          </span>
          <span className="text-white-50 text-truncate" style={{ fontSize: "0.7rem" }}>
            {device ? getDeviceDisplayName(device) : "Unknown Device"}
          </span>
        </div>
        {updateCapture.isPending && <Spinner size="sm" />}
        {statusBadge}
      </div>

      {/* Device - at top, expanded by default */}
      <AccordionGroup allowMultiple>
        <AccordionItem
          id="device"
          defaultOpen
          header={
            <Flex align="center" gap={1}>
              <Cpu size={14} />
              <span className="small fw-semibold">Device</span>
              {statusBadge}
            </Flex>
          }
        >
          <DeviceControlsContent capture={capture} device={device} />
        </AccordionItem>
      </AccordionGroup>

      {/* Individual tuning accordions - each setting has its own */}
      <TuningAccordions capture={capture} device={device} />

      {/* Other accordions */}
      <AccordionGroup allowMultiple>
        {/* Recipes */}
        <AccordionItem
          id="recipes"
          header={
            <Flex align="center" gap={1}>
              <BookOpen size={14} />
              <span className="small fw-semibold">Recipes</span>
            </Flex>
          }
        >
          <RecipeSelectorContent capture={capture} />
        </AccordionItem>

        {/* Advanced */}
        <AccordionItem
          id="advanced"
          header={
            <Flex align="center" gap={1}>
              <Sliders size={14} />
              <span className="small fw-semibold">Advanced</span>
            </Flex>
          }
        >
          <AdvancedSettingsContent capture={capture} device={device} />
        </AccordionItem>

        {/* Scanner */}
        <AccordionItem
          id="scanner"
          header={
            <Flex align="center" gap={1}>
              <Scan size={14} />
              <span className="small fw-semibold">Scanner</span>
            </Flex>
          }
        >
          <ScannerControl captureId={capture.id} />
        </AccordionItem>

        {/* Bookmarks */}
        <AccordionItem
          id="bookmarks"
          header={
            <Flex align="center" gap={1}>
              <Bookmark size={14} />
              <span className="small fw-semibold">Bookmarks</span>
            </Flex>
          }
        >
          <BookmarkManager
            currentFrequency={capture.centerHz}
            onTuneToFrequency={handleTuneToFrequency}
            currentCapture={capture}
            currentChannels={channels}
            onLoadMemoryBank={handleLoadMemoryBank}
          />
        </AccordionItem>
      </AccordionGroup>
    </Flex>
  );
}
