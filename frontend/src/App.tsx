import { useState } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Radio, Plus } from "lucide-react";
import { useDevices } from "./hooks/useDevices";
import { useCaptures, useCreateCapture } from "./hooks/useCaptures";
import { RadioTuner } from "./components/RadioTuner.react";
import { ChannelManager } from "./components/ChannelManager.react";
import Flex from "./components/primitives/Flex.react";
import Spinner from "./components/primitives/Spinner.react";
import Button from "./components/primitives/Button.react";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

function AppContent() {
  const { data: devices, isLoading: devicesLoading } = useDevices();
  const { data: captures, isLoading: capturesLoading } = useCaptures();
  const createCapture = useCreateCapture();

  const [selectedCaptureId, setSelectedCaptureId] = useState<string | null>(null);
  const [showNewCapture, setShowNewCapture] = useState(false);
  const [newCaptureDeviceId, setNewCaptureDeviceId] = useState<string>("");
  const [newCaptureFreq, setNewCaptureFreq] = useState<number>(100_000_000);

  // Auto-select first capture
  const selectedCapture = captures?.find((c) => c.id === selectedCaptureId) ?? captures?.[0];

  // Find device for selected capture
  const selectedDevice = devices?.find((d) => d.id === selectedCapture?.deviceId);

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
          <div className="col-lg-3">
            <div className="card shadow-sm">
              <div className="card-header bg-body-tertiary">
                <Flex justify="between" align="center">
                  <h3 className="h6 mb-0">Captures</h3>
                  <Button
                    use="primary"
                    size="sm"
                    onClick={() => setShowNewCapture(!showNewCapture)}
                  >
                    <Plus size={16} />
                  </Button>
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
                      <button
                        key={capture.id}
                        className={`list-group-item list-group-item-action ${
                          selectedCapture?.id === capture.id ? "active" : ""
                        }`}
                        onClick={() => setSelectedCaptureId(capture.id)}
                      >
                        <Flex direction="column" gap={1}>
                          <div className="fw-semibold">{capture.id}</div>
                          {captureDevice && (
                            <div className="small opacity-75">
                              {captureDevice.driver.toUpperCase()}
                            </div>
                          )}
                          <div className="small">
                            {(capture.centerHz / 1_000_000).toFixed(3)} MHz
                          </div>
                          <span className={`badge bg-${capture.state === "running" ? "success" : capture.state === "failed" ? "danger" : "secondary"} w-auto`}>
                            {capture.state}
                          </span>
                        </Flex>
                      </button>
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

          {/* Main Content */}
          <div className="col-lg-6">
            {selectedCapture ? (
              <RadioTuner capture={selectedCapture} device={selectedDevice} />
            ) : (
              <div className="card shadow-sm">
                <div className="card-body text-center py-5">
                  <p className="text-muted">No capture selected. Create a new capture to get started.</p>
                </div>
              </div>
            )}
          </div>

          {/* Channels Sidebar */}
          <div className="col-lg-3">
            {selectedCapture ? (
              <ChannelManager capture={selectedCapture} />
            ) : (
              <div className="card shadow-sm">
                <div className="card-body text-center text-muted py-4">
                  Select a capture to manage channels
                </div>
              </div>
            )}

            {/* Device Info */}
            {selectedDevice && (
              <div className="card shadow-sm mt-4">
                <div className="card-header bg-body-tertiary">
                  <h3 className="h6 mb-0">Device Info</h3>
                </div>
                <div className="card-body">
                  <dl className="mb-0">
                    <dt className="small text-muted">Driver</dt>
                    <dd className="mb-2">{selectedDevice.driver}</dd>

                    <dt className="small text-muted">Label</dt>
                    <dd className="mb-2 small">{selectedDevice.label}</dd>

                    <dt className="small text-muted">Frequency Range</dt>
                    <dd className="mb-0 small">
                      {(selectedDevice.freqMinHz / 1_000_000).toFixed(1)} -{" "}
                      {(selectedDevice.freqMaxHz / 1_000_000).toFixed(0)} MHz
                    </dd>
                  </dl>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
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
