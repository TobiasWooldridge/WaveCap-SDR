import { useState, useEffect, useRef } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Radio, Plus, Wand2, X, Edit2, Settings } from "lucide-react";
import { ToastProvider, useToast } from "./hooks/useToast";
import { useDevices } from "./hooks/useDevices";
import { useCaptures, useCreateCapture, useDeleteCapture, useUpdateCapture } from "./hooks/useCaptures";
import { useChannels } from "./hooks/useChannels";
import { RadioTuner } from "./components/RadioTuner.react";
import { ChannelManager } from "./components/ChannelManager.react";
import { CreateCaptureWizard } from "./components/CreateCaptureWizard.react";
import SpectrumAnalyzer from "./components/primitives/SpectrumAnalyzer.react";
import WaterfallDisplay from "./components/primitives/WaterfallDisplay.react";
import { formatFrequencyMHz } from "./utils/frequency";
import Flex from "./components/primitives/Flex.react";
import Spinner from "./components/primitives/Spinner.react";
import Button from "./components/primitives/Button.react";
import { DeviceSettingsModal } from "./components/DeviceSettingsModal.react";
import ErrorBoundary from "./components/ErrorBoundary.react";

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

interface CaptureTabProps {
  capture: any;
  captureDevice: any;
  isSelected: boolean;
  onClick: () => void;
  onDelete: () => void;
  onUpdateName: (name: string | null) => void;
  channelCount: number;
}

function CaptureTab({ capture, captureDevice: _captureDevice, isSelected, onClick, onDelete, onUpdateName, channelCount }: CaptureTabProps) {
  const stateColor = capture.state === "running" ? "success" : capture.state === "failed" ? "danger" : "secondary";
  const [isEditing, setIsEditing] = useState(false);
  const [editValue, setEditValue] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const displayName = capture.name || capture.autoName || formatCaptureId(capture.id);
  const hasCustomName = !!capture.name;

  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [isEditing]);

  const handleStartEdit = (e: React.MouseEvent) => {
    e.stopPropagation();
    setEditValue(capture.name || capture.autoName || "");
    setIsEditing(true);
  };

  const handleSaveEdit = () => {
    const trimmedValue = editValue.trim();
    onUpdateName(trimmedValue || null);
    setIsEditing(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleSaveEdit();
    } else if (e.key === "Escape") {
      setIsEditing(false);
    }
  };

  return (
    <button
      className={`btn btn-sm d-flex align-items-center gap-2 ${isSelected ? 'btn-light' : 'btn-outline-light'}`}
      onClick={onClick}
      style={{
        position: "relative",
        borderRadius: "0.375rem 0.375rem 0 0",
        borderBottom: "none",
        whiteSpace: "nowrap",
      }}
    >
      <span className={`badge bg-${stateColor}`} style={{ width: "8px", height: "8px", padding: 0, borderRadius: "50%" }}></span>

      {isEditing ? (
        <input
          ref={inputRef}
          type="text"
          className="form-control form-control-sm"
          style={{ width: "120px", height: "20px", fontSize: "12px", padding: "2px 6px" }}
          value={editValue}
          onChange={(e) => setEditValue(e.target.value)}
          onBlur={handleSaveEdit}
          onKeyDown={handleKeyDown}
          onClick={(e) => e.stopPropagation()}
        />
      ) : (
        <span
          className={`fw-semibold ${isSelected ? 'text-dark' : 'text-white'}`}
          title={hasCustomName && capture.autoName ? `Auto: ${capture.autoName}` : undefined}
        >
          {displayName}
        </span>
      )}

      {!isEditing && (
        <button
          className={`btn btn-sm p-0 ${isSelected ? 'text-dark' : 'text-white'}`}
          style={{ width: "14px", height: "14px", lineHeight: 1 }}
          onClick={handleStartEdit}
          title="Edit name"
        >
          <Edit2 size={10} />
        </button>
      )}

      <span className={`small ${isSelected ? 'text-muted' : 'text-white opacity-75'}`}>
        {formatFrequencyMHz(capture.centerHz)} MHz • {channelCount} ch
      </span>
      <button
        className={`btn btn-sm p-0 ms-1 ${isSelected ? 'text-dark' : 'text-white'}`}
        style={{ width: "16px", height: "16px", lineHeight: 1 }}
        onClick={(e) => {
          e.stopPropagation();
          onDelete();
        }}
        title="Delete capture"
      >
        <X size={12} />
      </button>
    </button>
  );
}

// Wrapper component that can use hooks properly
function CaptureTabWithData({ capture, devices, isSelected, onClick, onDelete, onUpdateName }: {
  capture: any;
  devices: any[] | undefined;
  isSelected: boolean;
  onClick: () => void;
  onDelete: () => void;
  onUpdateName: (name: string | null) => void;
}) {
  const { data: channels } = useChannels(capture.id);
  const captureDevice = devices?.find((d) => d.id === capture.deviceId);
  const channelCount = channels?.length ?? 0;

  return (
    <CaptureTab
      capture={capture}
      captureDevice={captureDevice}
      isSelected={isSelected}
      onClick={onClick}
      onDelete={onDelete}
      onUpdateName={onUpdateName}
      channelCount={channelCount}
    />
  );
}

function AppContent() {
  const { data: devices, isLoading: devicesLoading } = useDevices();
  const { data: captures, isLoading: capturesLoading } = useCaptures();
  const createCapture = useCreateCapture();
  const deleteCapture = useDeleteCapture();
  const updateCapture = useUpdateCapture();
  const toast = useToast();

  // Initialize from URL query parameter
  const [selectedCaptureId, setSelectedCaptureId] = useState<string | null>(() => {
    const params = new URLSearchParams(window.location.search);
    return params.get('capture');
  });
  const [showNewCaptureModal, setShowNewCaptureModal] = useState(false);
  const [showWizard, setShowWizard] = useState(false);
  const [showDeviceSettings, setShowDeviceSettings] = useState(false);
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

  const handleFrequencyClick = (frequencyHz: number) => {
    if (!selectedCapture) return;

    updateCapture.mutate({
      captureId: selectedCapture.id,
      request: {
        centerHz: frequencyHz,
      },
    }, {
      onSuccess: () => {
        toast.success(`Tuned to ${(frequencyHz / 1e6).toFixed(3)} MHz`);
      },
      onError: (error: any) => {
        toast.error(error?.message || "Failed to tune frequency");
      },
    });
  };

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
        setShowNewCaptureModal(false);
        setSelectedCaptureId(newCapture.id);
        toast.success("Capture created successfully");
      },
      onError: (error: any) => {
        toast.error(error?.message || "Failed to create capture");
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
      {/* Header with Capture Tabs */}
      <nav className="navbar navbar-dark bg-primary shadow-sm mb-0" style={{ paddingBottom: 0 }}>
        <div className="container-fluid" style={{ flexDirection: "column", alignItems: "stretch", gap: "0.5rem" }}>
          {/* Top Row: Branding and Device Info */}
          <Flex align="center" justify="between" style={{ paddingBottom: "0.5rem" }}>
            {/* Branding */}
            <Flex align="center" gap={2} className="text-white">
              <Radio size={24} />
              <span className="navbar-brand mb-0 h5 text-white">WaveCap SDR</span>
            </Flex>

            {/* Device Settings and Count */}
            <Flex align="center" gap={2}>
              <Button
                use="light"
                size="sm"
                onClick={() => setShowDeviceSettings(true)}
                title="Device Settings"
              >
                <Settings size={16} />
              </Button>
              <span className="badge bg-light text-dark">
                {devices?.length ?? 0} device{devices?.length !== 1 ? "s" : ""}
              </span>
            </Flex>
          </Flex>

          {/* Bottom Row: Capture Tabs */}
          <Flex align="end" gap={2} style={{ marginBottom: "-1px" }}>
            {captures && captures.length > 0 && (
              <>
                {captures.map((capture) => (
                  <CaptureTabWithData
                    key={capture.id}
                    capture={capture}
                    devices={devices}
                    isSelected={selectedCapture?.id === capture.id}
                    onClick={() => setSelectedCaptureId(capture.id)}
                    onDelete={() => {
                      deleteCapture.mutate(capture.id, {
                        onSuccess: () => {
                          toast.success("Capture deleted successfully");
                        },
                        onError: (error: any) => {
                          toast.error(error?.message || "Failed to delete capture");
                        },
                      });
                      if (selectedCaptureId === capture.id) {
                        setSelectedCaptureId(null);
                      }
                    }}
                    onUpdateName={(name) => {
                      updateCapture.mutate({
                        captureId: capture.id,
                        request: { name },
                      }, {
                        onSuccess: () => {
                          toast.success("Capture name updated");
                        },
                        onError: (error: any) => {
                          toast.error(error?.message || "Failed to update capture name");
                        },
                      });
                    }}
                  />
                ))}
              </>
            )}

            {/* Add Capture Buttons */}
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
                use="light"
                size="sm"
                onClick={() => setShowNewCaptureModal(true)}
                title="Manual Setup"
              >
                <Plus size={16} />
              </Button>
            </Flex>
          </Flex>
        </div>
      </nav>

      {/* Main Content - Responsive Multi-Column Layout */}
      <div className="container-fluid px-4 py-3" style={{ height: 'calc(100vh - 120px)' }}>
        {selectedCapture ? (
          <div className="row g-2" style={{ height: '100%' }}>
            {/* Column 1: Visualizations (Spectrum + Waterfall) */}
            <div className="col-12 col-lg-6 col-xxl-4" style={{ height: '100%' }}>
              <div style={{ height: '100%', overflowY: 'auto', paddingRight: '4px' }}>
                <Flex direction="column" gap={2}>
                  {/* Spectrum Analyzer */}
                  <ErrorBoundary>
                    <SpectrumAnalyzer
                      capture={selectedCapture}
                      channels={selectedCaptureChannels}
                      height={200}
                      onFrequencyClick={handleFrequencyClick}
                    />
                  </ErrorBoundary>

                  {/* Waterfall Display */}
                  <ErrorBoundary>
                    <WaterfallDisplay
                      capture={selectedCapture}
                      channels={selectedCaptureChannels}
                      height={300}
                      timeSpanSeconds={30}
                      colorScheme="heat"
                      intensity={1.2}
                    />
                  </ErrorBoundary>
                </Flex>
              </div>
            </div>

            {/* Column 2: Radio Tuner (Controls) */}
            <div className="col-12 col-lg-6 col-xxl-4" style={{ height: '100%' }}>
              <div style={{ height: '100%', overflowY: 'auto', paddingRight: '4px' }}>
                <ErrorBoundary>
                  <RadioTuner capture={selectedCapture} device={selectedDevice} />
                </ErrorBoundary>
              </div>
            </div>

            {/* Column 3: Channel Manager */}
            <div className="col-12 col-xxl-4" style={{ height: '100%' }}>
              <div style={{ height: '100%', overflowY: 'auto', paddingRight: '4px' }}>
                <ErrorBoundary>
                  <ChannelManager capture={selectedCapture} />
                </ErrorBoundary>
              </div>
            </div>
          </div>
        ) : (
          <div className="card shadow-sm">
            <div className="card-body text-center py-5">
              <p className="text-muted">No captures available. Click + to create one or use the Wizard.</p>
            </div>
          </div>
        )}
      </div>

      {/* New Capture Modal */}
      {showNewCaptureModal && (
        <div className="modal d-block" style={{ backgroundColor: "rgba(0,0,0,0.5)" }} onClick={() => setShowNewCaptureModal(false)}>
          <div className="modal-dialog modal-dialog-centered" onClick={(e) => e.stopPropagation()}>
            <div className="modal-content">
              <div className="modal-header">
                <h5 className="modal-title">New Capture</h5>
                <button type="button" className="btn-close" onClick={() => setShowNewCaptureModal(false)}></button>
              </div>
              <div className="modal-body">
                <Flex direction="column" gap={3}>
                  <Flex direction="column" gap={1}>
                    <label className="form-label">Device</label>
                    <select
                      className="form-select"
                      value={newCaptureDeviceId}
                      onChange={(e) => setNewCaptureDeviceId(e.target.value)}
                    >
                      {devices?.map((device) => (
                        <option key={device.id} value={device.id}>
                          {device.driver.toUpperCase()} - {device.label.substring(0, 60)}
                        </option>
                      ))}
                    </select>
                  </Flex>

                  <Flex direction="column" gap={1}>
                    <label className="form-label">Frequency (MHz)</label>
                    <input
                      type="number"
                      className="form-control"
                      value={(newCaptureFreq / 1_000_000).toFixed(3)}
                      onChange={(e) => setNewCaptureFreq(parseFloat(e.target.value) * 1_000_000)}
                      step="0.1"
                    />
                  </Flex>
                </Flex>
              </div>
              <div className="modal-footer">
                <Button
                  use="secondary"
                  onClick={() => setShowNewCaptureModal(false)}
                >
                  Cancel
                </Button>
                <Button
                  use="success"
                  onClick={handleCreateCapture}
                  disabled={createCapture.isPending}
                >
                  Create
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}

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

      {/* Device Settings Modal */}
      {showDeviceSettings && (
        <DeviceSettingsModal
          onClose={() => setShowDeviceSettings(false)}
        />
      )}
    </div>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ToastProvider>
        <ErrorBoundary>
          <AppContent />
        </ErrorBoundary>
      </ToastProvider>
    </QueryClientProvider>
  );
}
