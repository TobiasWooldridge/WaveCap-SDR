import { useState, useEffect, useRef, useMemo } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Radio, Plus, Wand2, X, Edit2, Settings, Play } from "lucide-react";
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
import { ErrorProvider } from "./context/ErrorContext";

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

// Extract a stable identifier from a SoapySDR device ID string.
// Device IDs can contain volatile fields like 'tuner' that change based on
// device availability. This function extracts driver + serial (or label) to
// create a stable ID for matching captures to devices.
// For non-SoapySDR format IDs (e.g., "device0"), returns the original ID.
function getStableDeviceId(deviceId: string): string {
  // Check if this looks like a SoapySDR format ID (has key=value pairs)
  if (!deviceId.includes("=")) {
    return deviceId; // Non-SoapySDR format, use as-is
  }

  let driver = "";
  let serial = "";
  let label = "";
  for (const part of deviceId.split(",")) {
    if (part.startsWith("driver=")) {
      driver = part.split("=")[1] || "";
    } else if (part.startsWith("serial=")) {
      serial = part.split("=")[1] || "";
    } else if (part.startsWith("label=")) {
      label = part.split("=")[1] || "";
    }
  }

  // If we couldn't extract useful fields, fall back to original ID
  if (!driver && !serial && !label) {
    return deviceId;
  }

  return serial ? `${driver}:${serial}` : `${driver}:${label}`;
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
  const isRunning = capture.state === "running";
  const isFailed = capture.state === "failed";
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
      className={`btn btn-sm d-flex align-items-center gap-2 ${isSelected ? 'btn-light' : ''}`}
      onClick={onClick}
      style={{
        position: "relative",
        borderRadius: "0.375rem 0.375rem 0 0",
        whiteSpace: "nowrap",
        ...(isSelected ? {
          borderBottom: "none",
        } : {
          borderTop: "1px solid rgba(255,255,255,0.5)",
          borderLeft: "1px solid rgba(255,255,255,0.5)",
          borderRight: "1px solid rgba(255,255,255,0.5)",
          borderBottom: "none",
          color: "white",
          backgroundColor: "transparent",
        }),
      }}
      onMouseEnter={(e) => {
        if (!isSelected) {
          e.currentTarget.style.backgroundColor = "rgba(255,255,255,0.15)";
        }
      }}
      onMouseLeave={(e) => {
        if (!isSelected) {
          e.currentTarget.style.backgroundColor = "transparent";
        }
      }}
    >
      {isRunning ? (
        <Play size={12} fill="currentColor" className="text-success" />
      ) : isFailed ? (
        <span className="badge bg-danger" style={{ width: "8px", height: "8px", padding: 0, borderRadius: "50%" }}></span>
      ) : (
        <span className="badge bg-secondary" style={{ width: "8px", height: "8px", padding: 0, borderRadius: "50%" }}></span>
      )}

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

      {!isEditing && isSelected && (
        <span
          role="button"
          tabIndex={0}
          className="btn btn-sm p-0 text-dark"
          style={{ width: "14px", height: "14px", lineHeight: 1 }}
          onClick={handleStartEdit}
          onKeyDown={(e) => e.key === "Enter" && handleStartEdit(e as unknown as React.MouseEvent)}
          title="Edit name"
        >
          <Edit2 size={10} />
        </span>
      )}

      <span className={`small ${isSelected ? 'text-muted' : 'text-white opacity-75'}`}>
        {formatFrequencyMHz(capture.centerHz)} MHz • {channelCount} ch
      </span>
      <span
        role="button"
        tabIndex={0}
        className={`btn btn-sm p-0 ms-1 ${isSelected ? 'text-dark' : 'text-white'}`}
        style={{ width: "16px", height: "16px", lineHeight: 1 }}
        onClick={(e) => {
          e.stopPropagation();
          if (window.confirm(`Delete capture "${displayName}"?\n\nThis will stop the capture and remove all channels.`)) {
            onDelete();
          }
        }}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            e.stopPropagation();
            if (window.confirm(`Delete capture "${displayName}"?\n\nThis will stop the capture and remove all channels.`)) {
              onDelete();
            }
          }
        }}
        title="Delete capture"
      >
        <X size={12} />
      </span>
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

  // Initialize new capture form with first device
  useEffect(() => {
    if (devices && devices.length > 0 && !newCaptureDeviceId) {
      setNewCaptureDeviceId(devices[0].id);
    }
  }, [devices, newCaptureDeviceId]);

  // Find the selected capture, or use first available
  const selectedCapture = captures?.find((c) => c.id === selectedCaptureId) ?? captures?.[0];

  // Find device for selected capture (using stable ID matching for volatile fields)
  const selectedDevice = devices?.find((d) => {
    if (!selectedCapture?.deviceId) return false;
    // Try exact match first
    if (d.id === selectedCapture.deviceId) return true;
    // Fall back to stable ID matching (handles volatile fields like 'tuner')
    return getStableDeviceId(d.id) === getStableDeviceId(selectedCapture.deviceId);
  });

  // Get channels for selected capture
  const { data: selectedCaptureChannels } = useChannels(selectedCapture?.id);

  // Group captures by device for the UI
  const capturesByDevice = useMemo(() => {
    if (!captures || !devices) return [];

    // Create a map of deviceId -> { device, captures }
    const groupMap = new Map<string, { device: typeof devices[0] | null; captures: typeof captures }>();

    // Create a map from stable device ID to device object for matching
    const stableIdToDevice = new Map<string, typeof devices[0]>();
    for (const device of devices) {
      stableIdToDevice.set(getStableDeviceId(device.id), device);
    }

    // Initialize groups for all devices
    for (const device of devices) {
      groupMap.set(device.id, { device, captures: [] });
    }

    // Add an "Unassigned" group for captures without a device
    groupMap.set("_unassigned", { device: null, captures: [] });

    // Group captures by their deviceId using stable ID matching
    for (const capture of captures) {
      if (!capture.deviceId) {
        groupMap.get("_unassigned")!.captures.push(capture);
        continue;
      }

      // First try exact match
      let group = groupMap.get(capture.deviceId);
      if (group) {
        group.captures.push(capture);
        continue;
      }

      // Fall back to stable ID matching (handles volatile fields like 'tuner')
      const captureStableId = getStableDeviceId(capture.deviceId);
      const matchedDevice = stableIdToDevice.get(captureStableId);
      if (matchedDevice) {
        groupMap.get(matchedDevice.id)!.captures.push(capture);
      } else {
        // Device not found, put in unassigned
        groupMap.get("_unassigned")!.captures.push(capture);
      }
    }

    // Convert to array and filter out empty groups (except keep devices with no captures for context)
    return Array.from(groupMap.entries())
      .filter(([key, group]) => group.captures.length > 0 || (key !== "_unassigned" && group.device))
      .map(([deviceId, group]) => ({
        deviceId,
        device: group.device,
        captures: group.captures,
      }));
  }, [captures, devices]);

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

          {/* Bottom Row: Device-Grouped Capture Tabs */}
          <Flex align="end" gap={3} style={{ marginBottom: "-1px", flexWrap: "wrap" }}>
            {capturesByDevice.map((group) => (
              <Flex key={group.deviceId} align="end" gap={1}>
                {/* Device Label */}
                <span
                  style={{
                    fontSize: "0.7rem",
                    color: "rgba(255,255,255,0.7)",
                    padding: "0.25rem 0.5rem",
                    backgroundColor: "rgba(0,0,0,0.2)",
                    borderRadius: "0.25rem 0.25rem 0 0",
                    whiteSpace: "nowrap",
                    marginBottom: "0",
                  }}
                  title={group.device?.id || "Unassigned"}
                >
                  {group.device?.nickname || group.device?.shorthand || group.device?.label || "Unassigned"}
                </span>
                {/* Captures in this group */}
                {group.captures.map((capture) => (
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
              </Flex>
            ))}

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
        <ErrorProvider>
          <ErrorBoundary>
            <AppContent />
          </ErrorBoundary>
        </ErrorProvider>
      </ToastProvider>
    </QueryClientProvider>
  );
}
