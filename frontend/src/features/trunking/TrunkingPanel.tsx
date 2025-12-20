import { useState, useMemo } from "react";
import {
  Radio,
  Phone,
  Users,
  Plus,
  AlertTriangle,
  CheckCircle,
  XCircle,
} from "lucide-react";
import {
  useTrunkingSystem,
  useTalkgroups,
  useActiveCalls,
  useStartTrunkingSystem,
  useStopTrunkingSystem,
  useVocoderStatus,
} from "../../hooks/useTrunking";
import { useTrunkingWebSocket } from "../../hooks/useTrunkingWebSocket";
import type { ActiveCall } from "../../types/trunking";
import { SystemStatusPanel } from "./SystemStatusPanel";
import { ActiveCallsTable } from "./ActiveCallsTable";
import { TalkgroupDirectory } from "./TalkgroupDirectory";
import Flex from "../../components/primitives/Flex.react";
import Spinner from "../../components/primitives/Spinner.react";

type TabId = "active" | "talkgroups" | "history";

interface TrunkingPanelProps {
  /** System ID to display - selection is managed by parent */
  systemId: string;
  onCreateSystem?: () => void;
}

export function TrunkingPanel({ systemId, onCreateSystem }: TrunkingPanelProps) {
  const [activeTab, setActiveTab] = useState<TabId>("active");

  // Data fetching for the selected system
  const { data: system, isLoading: systemLoading } = useTrunkingSystem(systemId);
  const { data: vocoder } = useVocoderStatus();
  const { data: talkgroups } = useTalkgroups(systemId);
  const { data: activeCalls } = useActiveCalls(systemId);

  // WebSocket for real-time updates
  const {
    isConnected: wsConnected,
    activeCalls: wsActiveCalls,
  } = useTrunkingWebSocket({
    systemId,
    enabled: !!systemId,
  });

  // Use WebSocket calls if available, otherwise fall back to polling
  const displayCalls = useMemo(
    () => (wsConnected && wsActiveCalls.length > 0 ? wsActiveCalls : activeCalls ?? []),
    [activeCalls, wsActiveCalls, wsConnected]
  );

  // Mutations
  const startSystem = useStartTrunkingSystem();
  const stopSystem = useStopTrunkingSystem();

  // Active talkgroup IDs for highlighting
  const activeTalkgroupIds = useMemo(
    () => new Set(displayCalls.map((c: ActiveCall) => c.talkgroupId)),
    [displayCalls]
  );

  if (systemLoading) {
    return (
      <div className="d-flex justify-content-center align-items-center p-4">
        <Spinner size="md" />
      </div>
    );
  }

  // No system found (shouldn't happen if parent is managing selection correctly)
  if (!system) {
    return (
      <div className="text-center p-4">
        <Radio size={48} className="mb-3 text-muted opacity-50" />
        <h5>Trunking System Not Found</h5>
        <p className="text-muted small">
          The selected trunking system could not be loaded.
        </p>

        {/* Vocoder status */}
        {vocoder && !vocoder.anyAvailable && (
          <div className="alert alert-warning mt-2 small">
            <AlertTriangle size={14} className="me-1" />
            No vocoder available. Voice decoding requires DSD-FME.
          </div>
        )}

        {onCreateSystem && (
          <button className="btn btn-primary" onClick={onCreateSystem}>
            <Plus size={18} className="me-1" />
            Add System
          </button>
        )}
      </div>
    );
  }

  return (
    <Flex direction="column" gap={2} className="p-2">
      {/* System status panel */}
      <SystemStatusPanel
        system={system}
        onStart={() => startSystem.mutate(system.id)}
        onStop={() => stopSystem.mutate(system.id)}
        isStarting={startSystem.isPending}
        isStopping={stopSystem.isPending}
      />

      {/* Vocoder warning if not available */}
      {vocoder && !vocoder.anyAvailable && (
        <div className="alert alert-warning py-2 small d-flex align-items-center gap-2">
          <AlertTriangle size={14} />
          <span>No vocoder available. Voice decoding requires DSD-FME.</span>
          <div className="ms-auto d-flex gap-2">
            <span className="d-flex align-items-center gap-1">
              {vocoder.imbe.available ? (
                <CheckCircle size={14} className="text-success" />
              ) : (
                <XCircle size={14} className="text-danger" />
              )}
              IMBE
            </span>
            <span className="d-flex align-items-center gap-1">
              {vocoder.ambe2.available ? (
                <CheckCircle size={14} className="text-success" />
              ) : (
                <XCircle size={14} className="text-danger" />
              )}
              AMBE+2
            </span>
          </div>
        </div>
      )}

      {/* Content tabs */}
      <div className="card">
        <div className="card-header py-2">
          <ul className="nav nav-tabs card-header-tabs">
            <li className="nav-item">
              <button
                className={`nav-link ${activeTab === "active" ? "active" : ""}`}
                onClick={() => setActiveTab("active")}
              >
                <Phone size={14} className="me-1" />
                Active
                {displayCalls.length > 0 && (
                  <span className="badge bg-success ms-1">
                    {displayCalls.length}
                  </span>
                )}
              </button>
            </li>
            <li className="nav-item">
              <button
                className={`nav-link ${
                  activeTab === "talkgroups" ? "active" : ""
                }`}
                onClick={() => setActiveTab("talkgroups")}
              >
                <Users size={14} className="me-1" />
                Talkgroups
                {talkgroups && (
                  <span className="badge bg-secondary ms-1">
                    {talkgroups.length}
                  </span>
                )}
              </button>
            </li>
          </ul>
        </div>

        <div className="card-body">
          {activeTab === "active" && (
            <ActiveCallsTable calls={displayCalls} />
          )}

          {activeTab === "talkgroups" && talkgroups && (
            <TalkgroupDirectory
              talkgroups={talkgroups}
              activeTalkgroups={activeTalkgroupIds}
            />
          )}
        </div>
      </div>

      {/* WebSocket connection status */}
      <div className="small text-muted d-flex align-items-center gap-1">
        <span
          className={`badge ${wsConnected ? "bg-success" : "bg-danger"}`}
          style={{ width: 8, height: 8, padding: 0 }}
        />
        {wsConnected ? "Live" : "Disconnected"}
      </div>
    </Flex>
  );
}

/**
 * Empty state component shown when no trunking system is selected.
 * Used by App.tsx when tabs are empty or no system selected.
 */
export function TrunkingEmptyState({ onCreateSystem }: { onCreateSystem?: () => void }) {
  const { data: vocoder } = useVocoderStatus();

  return (
    <div className="text-center p-4">
      <Radio size={48} className="mb-3 text-muted opacity-50" />
      <h5>No Trunking System Selected</h5>
      <p className="text-muted small">
        Create a P25 trunking system to monitor digital radio traffic.
      </p>

      {/* Vocoder status */}
      {vocoder && (
        <div className="mb-3">
          <div className="d-flex justify-content-center gap-3">
            <span className="d-flex align-items-center gap-1">
              {vocoder.imbe.available ? (
                <CheckCircle size={14} className="text-success" />
              ) : (
                <XCircle size={14} className="text-danger" />
              )}
              IMBE
            </span>
            <span className="d-flex align-items-center gap-1">
              {vocoder.ambe2.available ? (
                <CheckCircle size={14} className="text-success" />
              ) : (
                <XCircle size={14} className="text-danger" />
              )}
              AMBE+2
            </span>
          </div>
          {!vocoder.anyAvailable && (
            <div className="alert alert-warning mt-2 small">
              <AlertTriangle size={14} className="me-1" />
              No vocoder available. Voice decoding requires DSD-FME.
            </div>
          )}
        </div>
      )}

      {onCreateSystem && (
        <button className="btn btn-primary" onClick={onCreateSystem}>
          <Plus size={18} className="me-1" />
          Add System
        </button>
      )}
    </div>
  );
}
