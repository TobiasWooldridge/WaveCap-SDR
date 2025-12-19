import { useState, useCallback } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Wand2 } from "lucide-react";
import { ToastProvider } from "./hooks/useToast";
import { ErrorProvider } from "./context/ErrorContext";
import { useSelectedRadio } from "./hooks/useSelectedRadio";
import { useChannels } from "./hooks/useChannels";
import { useDeleteCapture } from "./hooks/useCaptures";
import { useDeleteTrunkingSystem } from "./hooks/useTrunking";
import { useStateWebSocket } from "./hooks/useStateWebSocket";
import { useAudio } from "./hooks/useAudio";
import { RadioTabBar, RadioPanel } from "./features/radio";
import { ChannelList } from "./features/channel";
import { SpectrumPanel } from "./features/spectrum";
import { TrunkingPanel } from "./features/trunking";
import { CreateCaptureWizard } from "./components/CreateCaptureWizard.react";
import { CreateTrunkingWizard } from "./components/CreateTrunkingWizard.react";
import { DeviceSettingsModal } from "./components/DeviceSettingsModal.react";
import ErrorBoundary from "./components/ErrorBoundary.react";
import Spinner from "./components/primitives/Spinner.react";
import type { RadioTabType } from "./types";

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

  const {
    selectedType,
    selectedId,
    selectedCapture,
    selectedDevice,
    selectTab,
    tabs,
    isLoading,
  } = useSelectedRadio();

  const { data: channels } = useChannels(selectedCapture?.id ?? "");
  const deleteCapture = useDeleteCapture();
  const deleteTrunkingSystem = useDeleteTrunkingSystem();
  const { stopAll } = useAudio();

  // Stop all audio when changing tabs
  const handleSelectTab = useCallback((type: RadioTabType, id: string) => {
    if (type !== selectedType || id !== selectedId) {
      stopAll();
    }
    selectTab(type, id);
  }, [selectTab, selectedType, selectedId, stopAll]);

  const [showWizard, setShowWizard] = useState(false);
  const [showTrunkingWizard, setShowTrunkingWizard] = useState(false);
  const [showDeviceSettings, setShowDeviceSettings] = useState(false);

  const handleDeleteCapture = (captureId: string) => {
    if (confirm("Delete this capture and all its channels?")) {
      deleteCapture.mutate(captureId);
    }
  };

  const handleDeleteTrunkingSystem = (systemId: string) => {
    if (confirm("Delete this trunking system?")) {
      deleteTrunkingSystem.mutate(systemId);
    }
  };

  const handleCreateSuccess = (captureId: string) => {
    selectTab("capture", captureId);
    setShowWizard(false);
  };

  const handleTrunkingCreateSuccess = (systemId: string) => {
    selectTab("trunking", systemId);
    setShowTrunkingWizard(false);
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
        {/* Unified Tab Bar - shows both captures and trunking systems */}
        <RadioTabBar
          tabs={tabs}
          selectedType={selectedType}
          selectedId={selectedId}
          onSelectTab={handleSelectTab}
          onCreateCapture={() => setShowWizard(true)}
          onCreateTrunkingSystem={() => setShowTrunkingWizard(true)}
          onDeleteCapture={handleDeleteCapture}
          onDeleteTrunkingSystem={handleDeleteTrunkingSystem}
          onOpenSettings={() => setShowDeviceSettings(true)}
        />
      </div>

      {/* Main Content - depends on selection type */}
      {selectedType === "trunking" && selectedId ? (
        /* Trunking System Selected */
        <div className="flex-grow-1">
          <TrunkingPanel systemId={selectedId} />
        </div>
      ) : selectedType === "capture" && selectedCapture ? (
        /* Capture Selected */
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
        /* Nothing Selected */
        <div className="flex-grow-1 d-flex justify-content-center align-items-center">
          <div className="text-center text-muted">
            <Wand2 size={48} className="mb-3 opacity-50" />
            <h5>No Radio Selected</h5>
            <p className="small">Click the + button to add a radio or trunking system</p>
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

      {showTrunkingWizard && (
        <CreateTrunkingWizard
          onClose={() => setShowTrunkingWizard(false)}
          onSuccess={handleTrunkingCreateSuccess}
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
