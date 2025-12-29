import { useState, useMemo, useCallback, useEffect } from "react";
import {
  Radio,
  Phone,
  Users,
  Plus,
  AlertTriangle,
  CheckCircle,
  XCircle,
  History,
  MessageSquare,
  FileText,
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
import { useTrunkingSystemAudio } from "../../hooks/useAudio";
import { useToast } from "../../hooks/useToast";
import { copyToClipboard } from "../../utils/clipboard";
import type { ActiveCall } from "../../types/trunking";
import { SystemStatusPanel } from "./SystemStatusPanel";
import { SystemConfigPanel } from "./SystemConfigPanel";
import { ActiveCallsTable } from "./ActiveCallsTable";
import { TalkgroupDirectory } from "./TalkgroupDirectory";
import { CallEventLog, CallEvent } from "./CallEventLog";
import { MessageLog } from "./MessageLog";
import { ActivitySummary } from "./ActivitySummary";
import {
  StreamLinks,
  TRUNKING_SYSTEM_STREAM_FORMATS,
} from "../../components/StreamLinks";
import Flex from "../../components/primitives/Flex.react";
import Spinner from "../../components/primitives/Spinner.react";

type TabId = "active" | "talkgroups" | "messages" | "history" | "summary";

const MAX_EVENTS = 100; // Keep last 100 events

interface TrunkingPanelProps {
  /** System ID to display - selection is managed by parent */
  systemId: string;
  onCreateSystem?: () => void;
}

export function TrunkingPanel({
  systemId,
  onCreateSystem,
}: TrunkingPanelProps) {
  const [activeTab, setActiveTab] = useState<TabId>("active");
  const [callEvents, setCallEvents] = useState<CallEvent[]>([]);
  const toast = useToast();

  // Track events for the event log
  const handleCallStart = useCallback((call: ActiveCall) => {
    const event: CallEvent = {
      id: call.id,
      timestamp: call.startTime,
      type: "start",
      talkgroupId: call.talkgroupId,
      talkgroupName: call.talkgroupName,
      sourceId: call.sourceId,
      frequencyHz: call.frequencyHz,
      encrypted: call.encrypted,
    };
    setCallEvents((prev) => [...prev.slice(-MAX_EVENTS + 1), event]);
  }, []);

  const handleCallEnd = useCallback((call: ActiveCall) => {
    const event: CallEvent = {
      id: call.id,
      timestamp: Date.now() / 1000, // Use current time for end event
      type: "end",
      talkgroupId: call.talkgroupId,
      talkgroupName: call.talkgroupName,
      sourceId: call.sourceId,
      frequencyHz: call.frequencyHz,
      durationSeconds: call.durationSeconds,
      encrypted: call.encrypted,
    };
    setCallEvents((prev) => [...prev.slice(-MAX_EVENTS + 1), event]);
  }, []);

  // Data fetching for the selected system
  const { data: system, isLoading: systemLoading } =
    useTrunkingSystem(systemId);
  const { data: vocoder } = useVocoderStatus();
  const { data: talkgroups } = useTalkgroups(systemId);
  const { data: activeCalls } = useActiveCalls(systemId);

  // WebSocket for real-time updates
  const {
    isConnected: wsConnected,
    activeCalls: wsActiveCalls,
    messages: wsMessages,
    callHistory: wsCallHistory,
  } = useTrunkingWebSocket({
    systemId,
    enabled: !!systemId,
    onCallStart: handleCallStart,
    onCallEnd: handleCallEnd,
  });

  // Initialize call events from server-buffered call history on first load
  useEffect(() => {
    if (wsCallHistory.length > 0 && callEvents.length === 0) {
      // Convert server call history to CallEvent format
      // Each ended call becomes two events: start and end
      const events: CallEvent[] = [];
      for (const call of wsCallHistory) {
        // Add start event
        events.push({
          id: call.id + "_start",
          timestamp: call.startTime,
          type: "start",
          talkgroupId: call.talkgroupId,
          talkgroupName: call.talkgroupName,
          sourceId: call.sourceId,
          frequencyHz: call.frequencyHz,
          encrypted: call.encrypted,
        });
        // Add end event
        events.push({
          id: call.id + "_end",
          timestamp:
            call.endTime || call.startTime + (call.durationSeconds || 0),
          type: "end",
          talkgroupId: call.talkgroupId,
          talkgroupName: call.talkgroupName,
          sourceId: call.sourceId,
          frequencyHz: call.frequencyHz,
          durationSeconds: call.durationSeconds,
          encrypted: call.encrypted,
        });
      }
      // Sort by timestamp and keep last MAX_EVENTS
      events.sort((a, b) => a.timestamp - b.timestamp);
      setCallEvents(events.slice(-MAX_EVENTS));
    }
  }, [wsCallHistory, callEvents.length]);

  // Audio playback for trunking system
  const {
    isPlaying: isPlayingAudio,
    play: playAudio,
    stop: stopAudio,
  } = useTrunkingSystemAudio(systemId);

  // Use WebSocket calls if available, otherwise fall back to polling
  const displayCalls = useMemo(
    () =>
      wsConnected && wsActiveCalls.length > 0
        ? wsActiveCalls
        : (activeCalls ?? []),
    [activeCalls, wsActiveCalls, wsConnected],
  );

  // Mutations
  const startSystem = useStartTrunkingSystem();
  const stopSystem = useStopTrunkingSystem();

  // Active talkgroup IDs for highlighting
  const activeTalkgroupIds = useMemo(
    () => new Set(displayCalls.map((c: ActiveCall) => c.talkgroupId)),
    [displayCalls],
  );

  // Handle stream URL copy
  const handleCopyUrl = useCallback(
    async (url: string) => {
      const success = await copyToClipboard(url);
      if (success) {
        toast.success("URL copied to clipboard");
      } else {
        toast.error("Failed to copy URL");
      }
    },
    [toast],
  );

  // Per-call audio playback state (using system audio for now)
  // Note: Per-call playback would require additional state tracking
  // For MVP, the master play button handles all audio
  const [playingCallId, setPlayingCallId] = useState<string | null>(null);

  const handlePlayCall = useCallback(
    async (callId: string, _streamId: string) => {
      if (playingCallId === callId) {
        // Stop this call - for now just toggle off the indicator
        setPlayingCallId(null);
      } else {
        // Start this call - for now just toggle on the indicator
        // The system audio already includes all calls
        setPlayingCallId(callId);
        if (!isPlayingAudio) {
          await playAudio();
        }
      }
    },
    [playingCallId, isPlayingAudio, playAudio],
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
        isPlayingAudio={isPlayingAudio}
        onPlayAudio={playAudio}
        onStopAudio={stopAudio}
      />

      {/* System configuration panel - network info, site info, control channels */}
      <SystemConfigPanel system={system} />

      {/* System-level stream links */}
      <StreamLinks
        formats={TRUNKING_SYSTEM_STREAM_FORMATS}
        baseUrl={`/api/v1/trunking/stream/${systemId}/voice`}
        buttonLabel="System Stream URLs"
        onCopyUrl={handleCopyUrl}
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
        <div className="card-header py-2 d-flex align-items-center">
          <ul className="nav nav-tabs card-header-tabs flex-grow-1">
            <li className="nav-item">
              <button
                className={`nav-link ${activeTab === "active" ? "active" : ""}`}
                onClick={() => setActiveTab("active")}
                title="Live voice calls currently on the system"
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
                title="Directory of configured talkgroups"
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
            <li className="nav-item">
              <button
                className={`nav-link ${activeTab === "messages" ? "active" : ""}`}
                onClick={() => setActiveTab("messages")}
                title="P25 control channel messages (TSBKs)"
              >
                <MessageSquare size={14} className="me-1" />
                Messages
                {wsMessages.length > 0 && (
                  <span className="badge bg-dark ms-1">
                    {wsMessages.length}
                  </span>
                )}
              </button>
            </li>
            <li className="nav-item">
              <button
                className={`nav-link ${activeTab === "history" ? "active" : ""}`}
                onClick={() => setActiveTab("history")}
                title="Log of call start/end events"
              >
                <History size={14} className="me-1" />
                History
                {callEvents.length > 0 && (
                  <span className="badge bg-info ms-1">
                    {callEvents.length}
                  </span>
                )}
              </button>
            </li>
            <li className="nav-item">
              <button
                className={`nav-link ${activeTab === "summary" ? "active" : ""}`}
                onClick={() => setActiveTab("summary")}
                title="Activity summary and statistics"
              >
                <FileText size={14} className="me-1" />
                Summary
              </button>
            </li>
          </ul>
          {/* Connection status indicator */}
          <span
            className={`badge ${wsConnected ? "bg-success" : "bg-danger"} ms-2`}
            title={
              wsConnected
                ? "WebSocket connected - receiving live updates"
                : "WebSocket disconnected"
            }
            style={{ fontSize: "0.65rem" }}
          >
            {wsConnected ? "●" : "○"}
          </span>
        </div>

        <div className="card-body">
          {activeTab === "active" && (
            <ActiveCallsTable
              calls={displayCalls}
              systemId={systemId}
              onPlayAudio={handlePlayCall}
              playingCallId={playingCallId}
              onCopyUrl={handleCopyUrl}
            />
          )}

          {activeTab === "talkgroups" && talkgroups && (
            <TalkgroupDirectory
              talkgroups={talkgroups}
              activeTalkgroups={activeTalkgroupIds}
            />
          )}

          {activeTab === "messages" && (
            <MessageLog messages={wsMessages} maxHeight={400} />
          )}

          {activeTab === "history" && (
            <CallEventLog events={callEvents} maxHeight={400} />
          )}

          {activeTab === "summary" && system && (
            <ActivitySummary
              system={system}
              activeCalls={displayCalls}
              messages={wsMessages}
            />
          )}
        </div>
      </div>
    </Flex>
  );
}

/**
 * Empty state component shown when no trunking system is selected.
 * Used by App.tsx when tabs are empty or no system selected.
 */
export function TrunkingEmptyState({
  onCreateSystem,
}: {
  onCreateSystem?: () => void;
}) {
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
