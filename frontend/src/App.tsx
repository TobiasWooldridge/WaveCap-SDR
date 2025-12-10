import { useState, useCallback } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Wand2, Radio, Antenna } from "lucide-react";
import { ToastProvider } from "./hooks/useToast";
import { ErrorProvider } from "./context/ErrorContext";
import { useSelectedCapture } from "./hooks/useSelectedCapture";
import { useChannels } from "./hooks/useChannels";
import { useDeleteCapture } from "./hooks/useCaptures";
import { useStateWebSocket } from "./hooks/useStateWebSocket";
import { useAudio } from "./hooks/useAudio";
import { RadioTabBar, RadioPanel } from "./features/radio";
import { ChannelList } from "./features/channel";
import { SpectrumPanel } from "./features/spectrum";
import { TrunkingPanel } from "./features/trunking";
import { CreateCaptureWizard } from "./components/CreateCaptureWizard.react";
import { DeviceSettingsModal } from "./components/DeviceSettingsModal.react";
import ErrorBoundary from "./components/ErrorBoundary.react";
import Spinner from "./components/primitives/Spinner.react";

type AppMode = "radio" | "trunking";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

function AppContent() {
  // Subscribe to real-time state updates via WebSocket
  // This updates React Query cache when captures/channels change
  useStateWebSocket();

  const [appMode, setAppMode] = useState<AppMode>("radio");

  const {
    selectedCaptureId,
    selectedCapture,
    selectedDevice,
    selectCapture: baseSelectCapture,
    captures,
    devices,
    isLoading,
  } = useSelectedCapture();

  const { data: channels } = useChannels(selectedCaptureId ?? "");
  const deleteCapture = useDeleteCapture();
  const { stopAll } = useAudio();

  // Stop all audio when changing tabs/radios or modes
  const selectCapture = useCallback((captureId: string) => {
    if (captureId !== selectedCaptureId) {
      stopAll();
    }
    baseSelectCapture(captureId);
  }, [baseSelectCapture, selectedCaptureId, stopAll]);

  const handleModeChange = useCallback((mode: AppMode) => {
    if (mode !== appMode) {
      stopAll();
    }
    setAppMode(mode);
  }, [appMode, stopAll]);

  const [showWizard, setShowWizard] = useState(false);
  const [showDeviceSettings, setShowDeviceSettings] = useState(false);

  const handleDeleteCapture = (captureId: string) => {
    if (confirm("Delete this capture and all its channels?")) {
      deleteCapture.mutate(captureId);
    }
  };

  const handleCreateSuccess = (captureId: string) => {
    selectCapture(captureId);
    setShowWizard(false);
  };

  if (isLoading) {
    return (
      <div className="d-flex justify-content-center align-items-center" style={{ height: "100vh" }}>
        <Spinner size="md" />
      </div>
    );
  }

  return (
    <div className="d-flex flex-column" style={{ minHeight: "100vh" }}>
      {/* Top Navigation - sticky at top */}
      <div className="sticky-top bg-body border-bottom" style={{ zIndex: 1020 }}>
        {/* Mode selector */}
        <div className="d-flex align-items-center bg-dark border-bottom border-secondary">
          <div className="d-flex">
            <button
              className={`btn btn-sm rounded-0 border-0 px-3 py-2 d-flex align-items-center gap-1 ${
                appMode === "radio" ? "bg-body text-body" : "bg-dark text-light"
              }`}
              onClick={() => handleModeChange("radio")}
            >
              <Radio size={14} />
              Radio
            </button>
            <button
              className={`btn btn-sm rounded-0 border-0 px-3 py-2 d-flex align-items-center gap-1 ${
                appMode === "trunking" ? "bg-body text-body" : "bg-dark text-light"
              }`}
              onClick={() => handleModeChange("trunking")}
            >
              <Antenna size={14} />
              Trunking
            </button>
          </div>
        </div>

        {/* Radio Tab Bar - only show in radio mode */}
        {appMode === "radio" && (
          <RadioTabBar
            captures={captures}
            devices={devices}
            selectedCaptureId={selectedCaptureId}
            onSelectCapture={selectCapture}
            onCreateCapture={() => setShowWizard(true)}
            onDeleteCapture={handleDeleteCapture}
            onOpenSettings={() => setShowDeviceSettings(true)}
          />
        )}
      </div>

      {/* Main Content */}
      {appMode === "trunking" ? (
        /* Trunking Mode */
        <div className="flex-grow-1">
          <TrunkingPanel />
        </div>
      ) : selectedCapture ? (
        /* Radio Mode with capture selected */
        <div className="d-flex flex-column flex-lg-row">
          {/* Spectrum Panel - self-sizing based on internal state */}
          <div className="d-flex flex-column border-end" style={{ flex: "1 1 33%", minWidth: 0 }}>
            <SpectrumPanel capture={selectedCapture} channels={channels} />
          </div>

          {/* Radio Panel */}
          <div className="border-end" style={{ flex: "1 1 33%", minWidth: 0 }}>
            <RadioPanel capture={selectedCapture} device={selectedDevice ?? undefined} />
          </div>

          {/* Channel List */}
          <div style={{ flex: "1 1 33%", minWidth: 0 }}>
            <div className="p-2">
              <ChannelList capture={selectedCapture} />
            </div>
          </div>
        </div>
      ) : (
        /* Radio Mode with no capture selected */
        <div className="flex-grow-1 d-flex justify-content-center align-items-center">
          <div className="text-center text-muted">
            <Wand2 size={48} className="mb-3 opacity-50" />
            <h5>No Radio Selected</h5>
            <p className="small">Click "Add Radio" to create a new capture</p>
            <button className="btn btn-primary" onClick={() => setShowWizard(true)}>
              Add Radio
            </button>
          </div>
        </div>
      )}

      {/* Modals */}
      {showWizard && (
        <CreateCaptureWizard
          onClose={() => setShowWizard(false)}
          onSuccess={handleCreateSuccess}
        />
      )}

      {showDeviceSettings && (
        <DeviceSettingsModal onClose={() => setShowDeviceSettings(false)} />
      )}
    </div>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ToastProvider>
        <ErrorProvider>
          <ErrorBoundary>
            <AppContent />
          </ErrorBoundary>
        </ErrorProvider>
      </ToastProvider>
    </QueryClientProvider>
  );
}
