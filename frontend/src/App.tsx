import { useState, useEffect } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Radio, Plus, Wand2, Trash2 } from "lucide-react";
import { useDevices } from "./hooks/useDevices";
import { useCaptures, useCreateCapture, useDeleteCapture } from "./hooks/useCaptures";
import { useChannels } from "./hooks/useChannels";
import { RadioTuner } from "./components/RadioTuner.react";
import { ChannelManager } from "./components/ChannelManager.react";
import { CreateCaptureWizard } from "./components/CreateCaptureWizard.react";
import SpectrumAnalyzer from "./components/primitives/SpectrumAnalyzer.react";
import { formatFrequencyMHz } from "./utils/frequency";
import Flex from "./components/primitives/Flex.react";
import Spinner from "./components/primitives/Spinner.react";
import Button from "./components/primitives/Button.react";
import { FrequencyLabel } from "./components/FrequencyLabel.react";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

// Format capture ID for display (e.g., "c1" -> "Capture 1")
function formatCaptureId(id: string): string {
  const match = id.match(/^c(\d+)$/);
  return match ? `Capture ${match[1]}` : id;
}

interface CaptureListItemProps {
  capture: any;
  captureDevice: any;
  isSelected: boolean;
  onClick: () => void;
  onDelete: () => void;
}

function CaptureListItem({ capture, captureDevice, isSelected, onClick, onDelete }: CaptureListItemProps) {
  const { data: channels } = useChannels(capture.id);

  return (
    <div
      className={`list-group-item list-group-item-action ${isSelected ? "active" : ""}`}
      style={{ cursor: "pointer", position: "relative" }}
      onClick={onClick}
    >
      <Flex direction="column" gap={1}>
        <Flex justify="between" align="start">
          <div className="fw-semibold">{formatCaptureId(capture.id)}</div>
          <button
            className="btn btn-sm p-0 ms-2"
            style={{
              width: "20px",
              height: "20px",
              opacity: isSelected ? 0.8 : 0.5,
              color: isSelected ? "white" : "inherit"
            }}
            onClick={(e) => {
              e.stopPropagation();
              onDelete();
            }}
            title="Delete capture"
          >
            <Trash2 size={14} />
          </button>
        </Flex>
        {captureDevice && (
          <div className="small opacity-75">
            {captureDevice.driver.toUpperCase()}
          </div>
        )}
        <div className="small">
          {formatFrequencyMHz(capture.centerHz)} MHz
          <div><FrequencyLabel frequencyHz={capture.centerHz} /></div>
        </div>
        {channels && channels.length > 0 && (
          <div className="small opacity-75">
            {channels.length} channel{channels.length !== 1 ? "s" : ""}
            {channels.length > 0 && (
              <div className="mt-1">
                {channels.map((ch) => (
                  <div key={ch.id}>
                    {formatFrequencyMHz(capture.centerHz + ch.offsetHz)} MHz
                    <div><FrequencyLabel frequencyHz={capture.centerHz + ch.offsetHz} autoName={ch.autoName} /></div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
        <span className={`badge bg-${capture.state === "running" ? "success" : capture.state === "failed" ? "danger" : "secondary"} w-auto`}>
          {capture.state}
        </span>
      </Flex>
    </div>
  );
}

function AppContent() {
  const { data: devices, isLoading: devicesLoading } = useDevices();
  const { data: captures, isLoading: capturesLoading } = useCaptures();
  const createCapture = useCreateCapture();
  const deleteCapture = useDeleteCapture();

  // Initialize from URL query parameter
  const [selectedCaptureId, setSelectedCaptureId] = useState<string | null>(() => {
    const params = new URLSearchParams(window.location.search);
    return params.get('capture');
  });
  const [showNewCapture, setShowNewCapture] = useState(false);
  const [showWizard, setShowWizard] = useState(false);
  const [newCaptureDeviceId, setNewCaptureDeviceId] = useState<string>("");
  const [newCaptureFreq, setNewCaptureFreq] = useState<number>(100_000_000);

  // Update URL when capture selection changes
  useEffect(() => {
    if (selectedCaptureId) {
      const params = new URLSearchParams(window.location.search);
      params.set('capture', selectedCaptureId);
      const newUrl = `${window.location.pathname}?${params.toString()}`;
      window.history.pushState({}, '', newUrl);
    } else if (captures?.[0]) {
      // Auto-select first capture if none selected
      setSelectedCaptureId(captures[0].id);
    }
  }, [selectedCaptureId, captures]);

  // Find the selected capture, or use first available
  const selectedCapture = captures?.find((c) => c.id === selectedCaptureId) ?? captures?.[0];

  // Find device for selected capture
  const selectedDevice = devices?.find((d) => d.id === selectedCapture?.deviceId);

  // Get channels for selected capture
  const { data: selectedCaptureChannels } = useChannels(selectedCapture?.id);

  const handleCreateCapture = () => {
    if (!newCaptureDeviceId) return;

    const device = devices?.find((d) => d.id === newCaptureDeviceId);
    const sampleRate = device?.sampleRates[0] ?? 2_000_000;

    createCapture.mutate({
      deviceId: newCaptureDeviceId,
      centerHz: newCaptureFreq,
      sampleRate,
    }, {
      onSuccess: (newCapture) => {
        setShowNewCapture(false);
        setSelectedCaptureId(newCapture.id);
      },
    });
  };

  if (devicesLoading || capturesLoading) {
    return (
      <Flex justify="center" align="center" className="min-vh-100">
        <Flex direction="column" align="center" gap={3}>
          <Spinner />
          <div className="text-muted">Loading...</div>
        </Flex>
      </Flex>
    );
  }

  // Initialize new capture form with first device
  if (devices && devices.length > 0 && !newCaptureDeviceId) {
    setNewCaptureDeviceId(devices[0].id);
  }

  return (
    <div className="min-vh-100 bg-light">
      {/* Header */}
      <nav className="navbar navbar-dark bg-primary shadow-sm mb-4">
        <div className="container-fluid">
          <Flex align="center" gap={2}>
            <Radio size={28} />
            <span className="navbar-brand mb-0 h1">WaveCap SDR</span>
          </Flex>
          <span className="badge bg-light text-dark">
            {devices?.length ?? 0} device{devices?.length !== 1 ? "s" : ""} • {captures?.length ?? 0} capture{captures?.length !== 1 ? "s" : ""}
          </span>
        </div>
      </nav>

      <div className="container-fluid">
        <div className="row g-4">
          {/* Capture Selector Sidebar */}
          <div className="col-lg-4 col-xl-3">
            <div className="card shadow-sm">
              <div className="card-header bg-body-tertiary">
                <Flex justify="between" align="center">
                  <h3 className="h6 mb-0">Captures</h3>
                  <Flex gap={1}>
                    <Button
                      use="success"
                      size="sm"
                      onClick={() => setShowWizard(true)}
                      title="Use Recipe Wizard"
                    >
                      <Wand2 size={16} />
                    </Button>
                    <Button
                      use="primary"
                      size="sm"
                      onClick={() => setShowNewCapture(!showNewCapture)}
                      title="Manual Setup"
                    >
                      <Plus size={16} />
                    </Button>
                  </Flex>
                </Flex>
              </div>

              {/* New Capture Form */}
              {showNewCapture && (
                <div className="card-body border-bottom">
                  <Flex direction="column" gap={3}>
                    <h6 className="mb-0">New Capture</h6>

                    <Flex direction="column" gap={1}>
                      <label className="form-label small mb-1">Device</label>
                      <select
                        className="form-select form-select-sm"
                        value={newCaptureDeviceId}
                        onChange={(e) => setNewCaptureDeviceId(e.target.value)}
                      >
                        {devices?.map((device) => (
                          <option key={device.id} value={device.id}>
                            {device.driver.toUpperCase()} - {device.label.substring(0, 40)}
                          </option>
                        ))}
                      </select>
                    </Flex>

                    <Flex direction="column" gap={1}>
                      <label className="form-label small mb-1">Frequency (MHz)</label>
                      <input
                        type="number"
                        className="form-control form-control-sm"
                        value={(newCaptureFreq / 1_000_000).toFixed(3)}
                        onChange={(e) => setNewCaptureFreq(parseFloat(e.target.value) * 1_000_000)}
                        step="0.1"
                      />
                    </Flex>

                    <Flex gap={2}>
                      <Button
                        use="success"
                        size="sm"
                        onClick={handleCreateCapture}
                        disabled={createCapture.isPending}
                      >
                        Create
                      </Button>
                      <Button
                        use="secondary"
                        size="sm"
                        onClick={() => setShowNewCapture(false)}
                      >
                        Cancel
                      </Button>
                    </Flex>
                  </Flex>
                </div>
              )}

              <div className="list-group list-group-flush">
                {captures && captures.length > 0 ? (
                  captures.map((capture) => {
                    const captureDevice = devices?.find((d) => d.id === capture.deviceId);
                    return (
                      <CaptureListItem
                        key={capture.id}
                        capture={capture}
                        captureDevice={captureDevice}
                        isSelected={selectedCapture?.id === capture.id}
                        onClick={() => setSelectedCaptureId(capture.id)}
                        onDelete={() => {
                          deleteCapture.mutate(capture.id);
                          // If we're deleting the currently selected capture, clear selection
                          if (selectedCaptureId === capture.id) {
                            setSelectedCaptureId(null);
                          }
                        }}
                      />
                    );
                  })
                ) : (
                  <div className="list-group-item text-muted small">
                    No captures. Click + to create one.
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Main Content - Radio Tuner + Spectrum + Channels */}
          <div className="col-lg-8 col-xl-9">
            {selectedCapture ? (
              <Flex direction="column" gap={4}>
                <SpectrumAnalyzer capture={selectedCapture} channels={selectedCaptureChannels} height={200} />
                <RadioTuner capture={selectedCapture} device={selectedDevice} />
                <ChannelManager capture={selectedCapture} />
              </Flex>
            ) : (
              <div className="card shadow-sm">
                <div className="card-body text-center py-5">
                  <p className="text-muted">No capture selected. Create a new capture to get started.</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Wizard Modal */}
      {showWizard && (
        <CreateCaptureWizard
          onClose={() => setShowWizard(false)}
          onSuccess={(captureId) => {
            setSelectedCaptureId(captureId);
            setShowWizard(false);
          }}
        />
      )}
    </div>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AppContent />
    </QueryClientProvider>
  );
}
