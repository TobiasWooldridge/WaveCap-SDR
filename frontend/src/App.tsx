import { useState } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Wand2 } from "lucide-react";
import { ToastProvider } from "./hooks/useToast";
import { ErrorProvider } from "./context/ErrorContext";
import { useSelectedCapture } from "./hooks/useSelectedCapture";
import { useChannels } from "./hooks/useChannels";
import { useDeleteCapture } from "./hooks/useCaptures";
import { RadioTabBar, RadioPanel } from "./features/radio";
import { ChannelList } from "./features/channel";
import { SpectrumPanel } from "./features/spectrum";
import { CreateCaptureWizard } from "./components/CreateCaptureWizard.react";
import { DeviceSettingsModal } from "./components/DeviceSettingsModal.react";
import ErrorBoundary from "./components/ErrorBoundary.react";
import Spinner from "./components/primitives/Spinner.react";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

function AppContent() {
  const {
    selectedCaptureId,
    selectedCapture,
    selectedDevice,
    selectCapture,
    captures,
    devices,
    isLoading,
  } = useSelectedCapture();

  const { data: channels } = useChannels(selectedCaptureId ?? "");
  const deleteCapture = useDeleteCapture();

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
      {/* Tab Bar - sticky at top */}
      <div className="sticky-top bg-body border-bottom" style={{ zIndex: 1020 }}>
        <RadioTabBar
          captures={captures}
          devices={devices}
          selectedCaptureId={selectedCaptureId}
          onSelectCapture={selectCapture}
          onCreateCapture={() => setShowWizard(true)}
          onDeleteCapture={handleDeleteCapture}
          onOpenSettings={() => setShowDeviceSettings(true)}
        />
      </div>

      {/* Main Content - scrolls as a whole */}
      {selectedCapture ? (
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
