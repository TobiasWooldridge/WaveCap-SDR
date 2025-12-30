import { useState, useMemo } from "react";
import {
  useScanners,
  useCreateScanner,
  useStartScanner,
  useStopScanner,
  usePauseScanner,
  useResumeScanner,
  useLockScanner,
  useUnlockScanner,
  useLockoutFrequency,
  useClearLockout,
  useClearAllLockouts,
  useDeleteScanner,
} from "../hooks/useScanners";
import type { Scanner, CreateScannerRequest } from "../types";
import Flex from "./primitives/Flex.react";
import Button from "./primitives/Button.react";
import { FrequencyDisplay } from "./primitives/FrequencyDisplay.react";

interface ScannerControlProps {
  captureId: string;
}

export default function ScannerControl({ captureId }: ScannerControlProps) {
  const { data: allScanners, isLoading } = useScanners();
  const createScanner = useCreateScanner();
  const deleteScanner = useDeleteScanner();

  const scanners = useMemo(() => {
    return allScanners?.filter((s) => s.captureId === captureId) || [];
  }, [allScanners, captureId]);

  const [showCreateWizard, setShowCreateWizard] = useState(false);
  const [selectedScannerId, setSelectedScannerId] = useState<string | null>(null);

  const scanner = scanners.find((s) => s.id === selectedScannerId);

  if (isLoading) {
    return <div style={{ padding: "12px", color: "#6c757d" }}>Loading scanners...</div>;
  }

  if (showCreateWizard) {
    return (
      <CreateScannerWizard
        captureId={captureId}
        onCancel={() => setShowCreateWizard(false)}
        onCreate={(req) => {
          createScanner.mutate(req, {
            onSuccess: (newScanner) => {
              setSelectedScannerId(newScanner.id);
              setShowCreateWizard(false);
            },
          });
        }}
      />
    );
  }

  if (scanners.length === 0) {
    return (
      <Flex direction="column" gap={3} style={{ padding: "16px" }}>
        <div style={{ color: "#6c757d", textAlign: "center" }}>
          No scanners configured.
        </div>
        <Button onClick={() => setShowCreateWizard(true)}>Create Scanner</Button>
      </Flex>
    );
  }

  if (!scanner) {
    return (
      <Flex direction="column" gap={3} style={{ padding: "16px" }}>
        <Flex direction="row" gap={2} align="center" justify="between">
          <h3 style={{ margin: 0, fontSize: "14px", fontWeight: 600 }}>Scanners</h3>
          <Button size="sm" onClick={() => setShowCreateWizard(true)}>New</Button>
        </Flex>
        <Flex direction="column" gap={2}>
          {scanners.map((s) => (
            <div
              key={s.id}
              onClick={() => setSelectedScannerId(s.id)}
              style={{
                padding: "12px",
                border: "1px solid #dee2e6",
                borderRadius: "6px",
                cursor: "pointer",
                backgroundColor: "#f8f9fa",
              }}
            >
              <Flex direction="row" justify="between" align="center">
                <div>
                  <div style={{ fontWeight: 600, fontSize: "13px" }}>Scanner {s.id}</div>
                  <div style={{ fontSize: "11px", color: "#6c757d" }}>
                    {s.scanList.length} freqs • {s.mode} • {s.state}
                  </div>
                </div>
                <div
                  style={{
                    width: "10px",
                    height: "10px",
                    borderRadius: "50%",
                    backgroundColor:
                      s.state === "scanning" ? "#28a745" : s.state === "paused" ? "#ffc107" : "#6c757d",
                  }}
                />
              </Flex>
            </div>
          ))}
        </Flex>
      </Flex>
    );
  }

  return (
    <ScannerDetail
      scanner={scanner}
      onBack={() => setSelectedScannerId(null)}
      onDelete={() => {
        deleteScanner.mutate(scanner.id, {
          onSuccess: () => setSelectedScannerId(null),
        });
      }}
    />
  );
}

interface ScannerDetailProps {
  scanner: Scanner;
  onBack: () => void;
  onDelete: () => void;
}

function ScannerDetail({ scanner, onBack, onDelete }: ScannerDetailProps) {
  const startScanner = useStartScanner();
  const stopScanner = useStopScanner();
  const pauseScanner = usePauseScanner();
  const resumeScanner = useResumeScanner();
  const lockScanner = useLockScanner();
  const unlockScanner = useUnlockScanner();
  const lockoutCurrent = useLockoutFrequency();
  const clearLockout = useClearLockout();
  const clearAllLockouts = useClearAllLockouts();

  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);

  const stateColor =
    scanner.state === "scanning"
      ? "#28a745"
      : scanner.state === "paused"
      ? "#ffc107"
      : scanner.state === "locked"
      ? "#dc3545"
      : "#6c757d";

  return (
    <Flex direction="column" gap={3} style={{ padding: "16px" }}>
      <Flex direction="row" justify="between" align="center">
        <Flex direction="row" gap={2} align="center">
          <Button use="secondary" size="sm" onClick={onBack}>← Back</Button>
          <h3 style={{ margin: 0, fontSize: "14px", fontWeight: 600 }}>Scanner {scanner.id}</h3>
        </Flex>
        <div
          style={{
            padding: "4px 10px",
            borderRadius: "12px",
            fontSize: "11px",
            fontWeight: 600,
            backgroundColor: stateColor,
            color: "white",
            textTransform: "uppercase",
          }}
        >
          {scanner.state}
        </div>
      </Flex>

      <div
        style={{
          padding: "16px",
          backgroundColor: "#212529",
          borderRadius: "8px",
          border: "2px solid #495057",
        }}
      >
        <div style={{ fontSize: "11px", color: "#adb5bd", marginBottom: "4px" }}>
          CURRENT FREQUENCY
        </div>
        <div style={{ fontSize: "24px", fontWeight: 700, color: "#ffffff", fontFamily: "monospace" }}>
          <FrequencyDisplay frequencyHz={scanner.currentFrequency} decimals={4}  unit="MHz"/>
        </div>
        <div style={{ fontSize: "11px", color: "#6c757d", marginTop: "4px" }}>
          {scanner.currentIndex + 1} / {scanner.scanList.length}
        </div>
      </div>

      <Flex direction="row" gap={2} wrap="wrap">
        {scanner.state === "stopped" && (
          <Button use="primary" onClick={() => startScanner.mutate(scanner.id)}>
            ▶ Start
          </Button>
        )}
        {scanner.state === "scanning" && (
          <>
            <Button use="warning" onClick={() => pauseScanner.mutate(scanner.id)}>⏸ Pause</Button>
            <Button use="danger" onClick={() => stopScanner.mutate(scanner.id)}>⏹ Stop</Button>
            <Button use="secondary" onClick={() => lockScanner.mutate(scanner.id)}>🔒 Lock</Button>
            <Button use="secondary" onClick={() => lockoutCurrent.mutate(scanner.id)}>⛔ Lockout</Button>
          </>
        )}
        {scanner.state === "paused" && (
          <>
            <Button use="success" onClick={() => resumeScanner.mutate(scanner.id)}>▶ Resume</Button>
            <Button use="danger" onClick={() => stopScanner.mutate(scanner.id)}>⏹ Stop</Button>
          </>
        )}
        {scanner.state === "locked" && (
          <>
            <Button use="success" onClick={() => unlockScanner.mutate(scanner.id)}>🔓 Unlock</Button>
            <Button use="danger" onClick={() => stopScanner.mutate(scanner.id)}>⏹ Stop</Button>
          </>
        )}
      </Flex>

      <div style={{ padding: "12px", backgroundColor: "#f8f9fa", borderRadius: "6px" }}>
        <div style={{ fontSize: "12px", fontWeight: 600, marginBottom: "8px" }}>Config</div>
        <div style={{ fontSize: "11px" }}>
          <div>Mode: <strong>{scanner.mode}</strong></div>
          <div>Dwell: <strong>{scanner.dwellTimeMs}ms</strong></div>
          <div>Squelch: <strong>{scanner.squelchThresholdDb} dB</strong></div>
        </div>
      </div>

      {scanner.lockoutList.length > 0 && (
        <div style={{ padding: "12px", backgroundColor: "#fff3cd", borderRadius: "6px" }}>
          <Flex direction="row" justify="between" align="center" style={{ marginBottom: "8px" }}>
            <div style={{ fontSize: "12px", fontWeight: 600 }}>
              Lockouts ({scanner.lockoutList.length})
            </div>
            <Button use="warning" size="sm" onClick={() => clearAllLockouts.mutate(scanner.id)}>
              Clear All
            </Button>
          </Flex>
          <div style={{ fontSize: "11px", fontFamily: "monospace" }}>
            {scanner.lockoutList.map((freq, idx) => (
              <Flex key={idx} direction="row" justify="between" align="center">
                <span>
                  <FrequencyDisplay frequencyHz={freq} decimals={4}  unit="MHz"/>
                </span>
                <button
                  onClick={() => clearLockout.mutate({ scannerId: scanner.id, frequency: freq })}
                  style={{
                    background: "none",
                    border: "none",
                    color: "#dc3545",
                    cursor: "pointer",
                    fontSize: "12px",
                  }}
                >
                  ✕
                </button>
              </Flex>
            ))}
          </div>
        </div>
      )}

      {scanner.hits.length > 0 && (
        <div style={{ padding: "12px", backgroundColor: "#d4edda", borderRadius: "6px" }}>
          <div style={{ fontSize: "12px", fontWeight: 600, marginBottom: "8px" }}>
            Activity ({scanner.hits.length})
          </div>
          <div style={{ maxHeight: "120px", overflowY: "auto", fontSize: "11px", fontFamily: "monospace" }}>
            {scanner.hits.slice().reverse().map((hit, idx) => (
              <div key={idx} style={{ padding: "3px 0" }}>
                {new Date(hit.timestamp).toLocaleTimeString()} -{" "}
                <FrequencyDisplay frequencyHz={hit.frequencyHz} decimals={4}  unit="MHz"/>
              </div>
            ))}
          </div>
        </div>
      )}

      <div style={{ marginTop: "12px", paddingTop: "12px", borderTop: "1px solid #dee2e6" }}>
        {!showDeleteConfirm ? (
          <Button use="danger" size="sm" onClick={() => setShowDeleteConfirm(true)}>
            Delete
          </Button>
        ) : (
          <Flex direction="row" gap={2} align="center">
            <span style={{ fontSize: "12px", color: "#dc3545" }}>Are you sure?</span>
            <Button use="danger" size="sm" onClick={onDelete}>Yes</Button>
            <Button use="secondary" size="sm" onClick={() => setShowDeleteConfirm(false)}>
              Cancel
            </Button>
          </Flex>
        )}
      </div>
    </Flex>
  );
}

interface CreateScannerWizardProps {
  captureId: string;
  onCancel: () => void;
  onCreate: (req: CreateScannerRequest) => void;
}

function CreateScannerWizard({ captureId, onCancel, onCreate }: CreateScannerWizardProps) {
  const [mode, setMode] = useState<"sequential" | "priority" | "activity">("sequential");
  const [scanListText, setScanListText] = useState("");
  const [dwellTimeMs, setDwellTimeMs] = useState(500);
  const [squelchDb, setSquelchDb] = useState(-50);

  const handleCreate = () => {
    const scanList = scanListText
      .split(/[,\s]+/)
      .map((s) => parseFloat(s.trim()) * 1_000_000)
      .filter((f) => !isNaN(f) && f > 0);

    if (scanList.length === 0) {
      alert("Please enter at least one frequency");
      return;
    }

    const req: CreateScannerRequest = {
      captureId,
      scanList,
      mode,
      dwellTimeMs,
      squelchThresholdDb: squelchDb,
    };

    onCreate(req);
  };

  return (
    <Flex direction="column" gap={3} style={{ padding: "16px" }}>
      <Flex direction="row" justify="between" align="center">
        <h3 style={{ margin: 0, fontSize: "14px", fontWeight: 600 }}>Create Scanner</h3>
        <Button use="secondary" size="sm" onClick={onCancel}>Cancel</Button>
      </Flex>

      <div>
        <label style={{ display: "block", marginBottom: "6px", fontSize: "12px", fontWeight: 600 }}>
          Frequencies (MHz, comma/space-separated)
        </label>
        <textarea
          value={scanListText}
          onChange={(e) => setScanListText(e.target.value)}
          placeholder="156.8 156.65 156.55 or 156.8,156.65,156.55"
          rows={4}
          style={{
            width: "100%",
            padding: "8px",
            border: "1px solid #ced4da",
            borderRadius: "4px",
            fontSize: "11px",
            fontFamily: "monospace",
          }}
        />
      </div>

      <div>
        <label style={{ display: "block", marginBottom: "6px", fontSize: "12px", fontWeight: 600 }}>
          Scan Mode
        </label>
        <select
          value={mode}
          onChange={(e) => setMode(e.target.value as Scanner["mode"])}
          style={{
            width: "100%",
            padding: "8px",
            border: "1px solid #ced4da",
            borderRadius: "4px",
            fontSize: "12px",
          }}
        >
          <option value="sequential">Sequential</option>
          <option value="priority">Priority</option>
          <option value="activity">Activity</option>
        </select>
      </div>

      <Flex direction="row" gap={3}>
        <div style={{ flex: 1 }}>
          <label style={{ display: "block", marginBottom: "6px", fontSize: "12px", fontWeight: 600 }}>
            Dwell (ms)
          </label>
          <input
            type="number"
            value={dwellTimeMs}
            onChange={(e) => setDwellTimeMs(parseInt(e.target.value) || 500)}
            min={100}
            max={5000}
            step={100}
            style={{
              width: "100%",
              padding: "8px",
              border: "1px solid #ced4da",
              borderRadius: "4px",
              fontSize: "12px",
            }}
          />
        </div>
        <div style={{ flex: 1 }}>
          <label style={{ display: "block", marginBottom: "6px", fontSize: "12px", fontWeight: 600 }}>
            Squelch (dB)
          </label>
          <input
            type="number"
            value={squelchDb}
            onChange={(e) => setSquelchDb(parseInt(e.target.value) || -50)}
            min={-80}
            max={0}
            step={5}
            style={{
              width: "100%",
              padding: "8px",
              border: "1px solid #ced4da",
              borderRadius: "4px",
              fontSize: "12px",
            }}
          />
        </div>
      </Flex>

      <Button use="primary" onClick={handleCreate}>Create</Button>
    </Flex>
  );
}
