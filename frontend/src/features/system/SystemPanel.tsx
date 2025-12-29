import { useState, useRef, useEffect, useCallback, useMemo } from "react";
import {
  Cpu,
  MemoryStick,
  Thermometer,
  Radio,
  AlertTriangle,
  Pause,
  Play,
  Trash2,
  Search,
  ChevronDown,
  ChevronRight,
  Wifi,
  WifiOff,
} from "lucide-react";
import { useSystemStream } from "../../hooks/useSystemStream";
import type { CaptureMetrics, ErrorEvent } from "../../types";

type LogLevel = "DEBUG" | "INFO" | "WARNING" | "ERROR" | "CRITICAL";

const LOG_LEVEL_ORDER: LogLevel[] = [
  "DEBUG",
  "INFO",
  "WARNING",
  "ERROR",
  "CRITICAL",
];

function getLevelColor(level: LogLevel): string {
  switch (level) {
    case "DEBUG":
      return "text-body-secondary";
    case "INFO":
      return "text-info";
    case "WARNING":
      return "text-warning";
    case "ERROR":
      return "text-danger";
    case "CRITICAL":
      return "text-danger fw-bold";
    default:
      return "text-body";
  }
}

function getLevelBadgeClass(level: LogLevel): string {
  switch (level) {
    case "DEBUG":
      return "bg-secondary";
    case "INFO":
      return "bg-info";
    case "WARNING":
      return "bg-warning text-dark";
    case "ERROR":
      return "bg-danger";
    case "CRITICAL":
      return "bg-danger";
    default:
      return "bg-secondary";
  }
}

function formatTime(timestamp: number): string {
  const date = new Date(timestamp * 1000);
  return date.toLocaleTimeString("en-US", {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function formatBytes(mb: number): string {
  if (mb >= 1024) {
    return `${(mb / 1024).toFixed(1)} GB`;
  }
  return `${mb.toFixed(0)} MB`;
}

interface MetricCardProps {
  icon: React.ReactNode;
  label: string;
  value: string;
  subValue?: string;
  color?: string;
  progress?: number;
}

function MetricCard({
  icon,
  label,
  value,
  subValue,
  color = "text-primary",
  progress,
}: MetricCardProps) {
  return (
    <div className="card h-100">
      <div className="card-body p-2">
        <div className="d-flex align-items-center gap-2 mb-1">
          <span className={`d-flex ${color}`}>{icon}</span>
          <span className="text-muted small">{label}</span>
        </div>
        <div className="d-flex align-items-baseline gap-2">
          <span className="fs-4 fw-bold">{value}</span>
          {subValue && <span className="text-muted small">{subValue}</span>}
        </div>
        {progress !== undefined && (
          <div className="progress mt-2" style={{ height: "4px" }}>
            <div
              className={`progress-bar ${progress > 80 ? "bg-danger" : progress > 60 ? "bg-warning" : "bg-success"}`}
              style={{ width: `${progress}%` }}
            />
          </div>
        )}
      </div>
    </div>
  );
}

interface CaptureMetricsCardProps {
  metrics: CaptureMetrics;
  expanded: boolean;
  onToggle: () => void;
}

function CaptureMetricsCard({
  metrics,
  expanded,
  onToggle,
}: CaptureMetricsCardProps) {
  const hasErrors = metrics.iqOverflowRate > 0 || metrics.totalDrops > 0;
  const stateColor =
    metrics.state === "running"
      ? "text-success"
      : metrics.state === "stopped"
        ? "text-muted"
        : metrics.state === "failed"
          ? "text-danger"
          : "text-warning";

  return (
    <div className="card mb-2">
      <div
        className="card-header d-flex align-items-center gap-2 py-1 px-2"
        style={{ cursor: "pointer" }}
        onClick={onToggle}
      >
        <span className="d-flex">
          {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        </span>
        <span className={`d-flex ${stateColor}`}>
          <Radio size={14} />
        </span>
        <span className="fw-medium small">{metrics.captureId.slice(0, 8)}</span>
        <span
          className={`badge ${stateColor === "text-success" ? "bg-success" : stateColor === "text-danger" ? "bg-danger" : "bg-secondary"}`}
          style={{ fontSize: "0.65rem" }}
        >
          {metrics.state}
        </span>
        {hasErrors && (
          <span className="d-flex ms-auto">
            <AlertTriangle size={12} className="text-warning" />
          </span>
        )}
        <span className="text-muted small ms-auto">
          {metrics.channelCount} channels
        </span>
      </div>
      {expanded && (
        <div className="card-body p-2">
          <div className="row g-2" style={{ fontSize: "0.75rem" }}>
            <div className="col-6 col-md-3">
              <div className="text-muted">IQ Overflow</div>
              <div
                className={
                  metrics.iqOverflowRate > 0 ? "text-danger fw-bold" : ""
                }
              >
                {metrics.iqOverflowCount} ({metrics.iqOverflowRate.toFixed(1)}
                /s)
              </div>
            </div>
            <div className="col-6 col-md-3">
              <div className="text-muted">Audio Drops</div>
              <div className={metrics.totalDrops > 0 ? "text-warning" : ""}>
                {metrics.totalDrops}
              </div>
            </div>
            <div className="col-6 col-md-3">
              <div className="text-muted">Subscribers</div>
              <div>{metrics.totalSubscribers}</div>
            </div>
            <div className="col-6 col-md-3">
              <div className="text-muted">Channels</div>
              <div>{metrics.channelCount}</div>
            </div>
            <div className="col-6 col-md-4">
              <div className="text-muted">Loop Time</div>
              <div className={metrics.perfLoopMs > 50 ? "text-warning" : ""}>
                {metrics.perfLoopMs.toFixed(1)} ms
              </div>
            </div>
            <div className="col-6 col-md-4">
              <div className="text-muted">DSP Time</div>
              <div className={metrics.perfDspMs > 30 ? "text-warning" : ""}>
                {metrics.perfDspMs.toFixed(1)} ms
              </div>
            </div>
            <div className="col-6 col-md-4">
              <div className="text-muted">FFT Time</div>
              <div className={metrics.perfFftMs > 20 ? "text-warning" : ""}>
                {metrics.perfFftMs.toFixed(1)} ms
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

interface ErrorSummaryProps {
  errors: ErrorEvent[];
  onClear: () => void;
}

function ErrorSummary({ errors, onClear }: ErrorSummaryProps) {
  if (errors.length === 0) return null;

  const recentErrors = errors.slice(0, 5);
  const errorCounts: Record<string, number> = {};
  for (const error of errors) {
    errorCounts[error.type] = (errorCounts[error.type] || 0) + error.count;
  }

  return (
    <div className="card border-warning mb-3">
      <div className="card-header bg-warning bg-opacity-10 d-flex align-items-center gap-2 py-1 px-2">
        <span className="d-flex text-warning">
          <AlertTriangle size={14} />
        </span>
        <span className="fw-medium small">Recent Errors ({errors.length})</span>
        <button
          className="btn btn-sm btn-outline-secondary ms-auto"
          onClick={onClear}
          style={{ padding: "1px 6px", fontSize: "0.7rem" }}
        >
          Clear
        </button>
      </div>
      <div className="card-body p-2" style={{ fontSize: "0.75rem" }}>
        <div className="d-flex gap-3 mb-2">
          {Object.entries(errorCounts).map(([type, count]) => (
            <span key={type}>
              <span className="text-muted">{type}:</span> {count}
            </span>
          ))}
        </div>
        <div className="small text-muted">
          {recentErrors.map((error, idx) => (
            <div key={idx}>
              {formatTime(error.timestamp)} - {error.type} ({error.count}x) -{" "}
              {error.capture_id.slice(0, 8)}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export function SystemPanel() {
  const {
    isConnected,
    systemMetrics,
    captureMetrics,
    logs,
    errors,
    clearLogs,
    clearErrors,
  } = useSystemStream();

  const [minLevel, setMinLevel] = useState<LogLevel>("INFO");
  const [searchText, setSearchText] = useState("");
  const [isPaused, setIsPaused] = useState(false);
  const [expandedCaptures, setExpandedCaptures] = useState<Set<string>>(
    new Set(),
  );

  const scrollRef = useRef<HTMLDivElement>(null);
  const pausedLogsRef = useRef<typeof logs>([]);

  // Filter logs based on level and search
  const filteredLogs = useMemo(() => {
    const levelIdx = LOG_LEVEL_ORDER.indexOf(minLevel);
    return logs.filter((log) => {
      const logLevelIdx = LOG_LEVEL_ORDER.indexOf(log.level);
      if (logLevelIdx < levelIdx) return false;
      if (
        searchText &&
        !log.message.toLowerCase().includes(searchText.toLowerCase()) &&
        !log.loggerName.toLowerCase().includes(searchText.toLowerCase())
      ) {
        return false;
      }
      return true;
    });
  }, [logs, minLevel, searchText]);

  // When paused, show frozen logs; when not paused, show live logs
  const displayedLogs = useMemo(() => {
    if (isPaused) {
      return pausedLogsRef.current;
    }
    // Update the paused snapshot when not paused
    pausedLogsRef.current = filteredLogs;
    return filteredLogs;
  }, [isPaused, filteredLogs]);

  // Auto-scroll to top when new logs arrive (if not paused)
  useEffect(() => {
    if (!isPaused && scrollRef.current) {
      scrollRef.current.scrollTop = 0; // Logs are newest-first
    }
  }, [filteredLogs, isPaused]);

  // Handle scroll: auto-pause when scrolling down, auto-unpause at top
  const handleScroll = useCallback(() => {
    if (scrollRef.current) {
      const isAtTop = scrollRef.current.scrollTop < 20;

      if (isAtTop && isPaused) {
        // User scrolled back to top - resume
        setIsPaused(false);
      } else if (!isAtTop && !isPaused) {
        // User scrolled down - pause
        setIsPaused(true);
      }
    }
  }, [isPaused]);

  const toggleCapture = useCallback((captureId: string) => {
    setExpandedCaptures((prev) => {
      const next = new Set(prev);
      if (next.has(captureId)) {
        next.delete(captureId);
      } else {
        next.add(captureId);
      }
      return next;
    });
  }, []);

  return (
    <div className="d-flex flex-column h-100 overflow-hidden">
      {/* Connection status */}
      <div className="d-flex align-items-center gap-2 px-3 py-1 border-bottom bg-body-secondary">
        {isConnected ? (
          <>
            <span className="d-flex text-success">
              <Wifi size={14} />
            </span>
            <span className="small text-success">Connected</span>
          </>
        ) : (
          <>
            <span className="d-flex text-danger">
              <WifiOff size={14} />
            </span>
            <span className="small text-danger">Disconnected</span>
          </>
        )}
        {systemMetrics && (
          <span className="text-muted small ms-auto">
            Updated: {formatTime(systemMetrics.timestamp)}
          </span>
        )}
      </div>

      {/* Main content */}
      <div className="flex-grow-1 overflow-auto p-3">
        {/* System metrics cards */}
        <div className="row g-2 mb-3">
          <div className="col-6 col-md-3">
            <MetricCard
              icon={<Cpu size={16} />}
              label="CPU"
              value={
                systemMetrics
                  ? `${systemMetrics.cpuPercent.toFixed(0)}%`
                  : "---"
              }
              subValue={
                systemMetrics ? `${systemMetrics.cpuPerCore.length} cores` : ""
              }
              color="text-primary"
              progress={systemMetrics?.cpuPercent}
            />
          </div>
          <div className="col-6 col-md-3">
            <MetricCard
              icon={<MemoryStick size={16} />}
              label="Memory"
              value={
                systemMetrics ? formatBytes(systemMetrics.memoryUsedMb) : "---"
              }
              subValue={
                systemMetrics
                  ? `/ ${formatBytes(systemMetrics.memoryTotalMb)}`
                  : ""
              }
              color="text-info"
              progress={systemMetrics?.memoryPercent}
            />
          </div>
          <div className="col-6 col-md-3">
            <MetricCard
              icon={<Thermometer size={16} />}
              label="Temperature"
              value={
                systemMetrics &&
                Object.keys(systemMetrics.temperatures).length > 0
                  ? `${Object.values(systemMetrics.temperatures)[0].toFixed(0)}°C`
                  : "N/A"
              }
              subValue={
                systemMetrics &&
                Object.keys(systemMetrics.temperatures).length > 0
                  ? Object.keys(systemMetrics.temperatures)[0]
                  : "Not available"
              }
              color="text-warning"
            />
          </div>
          <div className="col-6 col-md-3">
            <MetricCard
              icon={<Radio size={16} />}
              label="Captures"
              value={captureMetrics.length.toString()}
              subValue={`${captureMetrics.filter((c) => c.state === "running").length} running`}
              color="text-success"
            />
          </div>
        </div>

        {/* CPU per-core breakdown */}
        {systemMetrics && systemMetrics.cpuPerCore.length > 0 && (
          <div className="card mb-3">
            <div className="card-header py-1 px-2 d-flex align-items-center gap-2">
              <span className="d-flex">
                <Cpu size={14} />
              </span>
              <span className="small fw-medium">CPU Cores</span>
            </div>
            <div className="card-body p-2">
              <div className="d-flex flex-wrap gap-1">
                {systemMetrics.cpuPerCore.map((usage, idx) => (
                  <div
                    key={idx}
                    className="d-flex align-items-center gap-1 bg-body-secondary rounded px-2 py-1"
                    style={{ fontSize: "0.7rem", minWidth: "60px" }}
                    title={`Core ${idx}: ${usage.toFixed(1)}%`}
                  >
                    <span className="text-muted">C{idx}</span>
                    <div
                      className="progress flex-grow-1"
                      style={{ height: "3px", width: "30px" }}
                    >
                      <div
                        className={`progress-bar ${usage > 80 ? "bg-danger" : usage > 60 ? "bg-warning" : "bg-success"}`}
                        style={{ width: `${usage}%` }}
                      />
                    </div>
                    <span>{usage.toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Error summary */}
        <ErrorSummary errors={errors} onClear={clearErrors} />

        {/* Per-capture metrics */}
        {captureMetrics.length > 0 && (
          <div className="mb-3">
            <h6 className="small fw-medium mb-2">Capture Metrics</h6>
            {captureMetrics.map((metrics) => (
              <CaptureMetricsCard
                key={metrics.captureId}
                metrics={metrics}
                expanded={expandedCaptures.has(metrics.captureId)}
                onToggle={() => toggleCapture(metrics.captureId)}
              />
            ))}
          </div>
        )}

        {/* Log viewer */}
        <div className="card flex-grow-1" style={{ minHeight: 300 }}>
          <div className="card-header py-1 px-2 d-flex align-items-center gap-2">
            <span className="small fw-medium">Logs</span>

            {/* Level filter */}
            <select
              className="form-select form-select-sm"
              style={{
                width: "auto",
                fontSize: "0.75rem",
                padding: "2px 24px 2px 8px",
              }}
              value={minLevel}
              onChange={(e) => setMinLevel(e.target.value as LogLevel)}
            >
              {LOG_LEVEL_ORDER.map((level) => (
                <option key={level} value={level}>
                  {level}+
                </option>
              ))}
            </select>

            {/* Search */}
            <div
              className="input-group input-group-sm"
              style={{ width: "150px" }}
            >
              <span className="input-group-text" style={{ padding: "2px 6px" }}>
                <Search size={12} />
              </span>
              <input
                type="text"
                className="form-control"
                placeholder="Filter..."
                style={{ fontSize: "0.75rem", padding: "2px 6px" }}
                value={searchText}
                onChange={(e) => setSearchText(e.target.value)}
              />
            </div>

            <span className="text-muted small ms-auto">
              {displayedLogs.length} / {logs.length}
              {isPaused && displayedLogs.length !== filteredLogs.length && (
                <span className="text-warning ms-1">
                  (+{filteredLogs.length - displayedLogs.length} new)
                </span>
              )}
            </span>

            {/* Controls */}
            <button
              className={`btn btn-sm ${isPaused ? "btn-warning" : "btn-outline-secondary"}`}
              onClick={() => setIsPaused(!isPaused)}
              title={isPaused ? "Resume auto-scroll" : "Pause auto-scroll"}
              style={{ padding: "2px 6px" }}
            >
              {isPaused ? <Play size={12} /> : <Pause size={12} />}
            </button>
            <button
              className="btn btn-sm btn-outline-secondary"
              onClick={clearLogs}
              title="Clear logs"
              style={{ padding: "2px 6px" }}
            >
              <Trash2 size={12} />
            </button>
          </div>

          <div
            ref={scrollRef}
            className="card-body p-0 overflow-auto font-monospace bg-body-tertiary"
            style={{ fontSize: "0.7rem", maxHeight: 400 }}
            onScroll={handleScroll}
          >
            {displayedLogs.length === 0 ? (
              <div className="text-center text-muted py-4">
                {logs.length === 0 ? "No logs yet" : "No logs match filter"}
              </div>
            ) : (
              <table className="table table-sm table-hover mb-0">
                <tbody>
                  {displayedLogs.map((log, idx) => (
                    <tr key={`${log.timestamp}-${idx}`}>
                      <td
                        className="text-muted"
                        style={{ width: "70px", whiteSpace: "nowrap" }}
                      >
                        {formatTime(log.timestamp)}
                      </td>
                      <td style={{ width: "60px" }}>
                        <span
                          className={`badge ${getLevelBadgeClass(log.level)}`}
                          style={{ fontSize: "0.6rem" }}
                        >
                          {log.level}
                        </span>
                      </td>
                      <td
                        className="text-muted"
                        style={{
                          width: "120px",
                          whiteSpace: "nowrap",
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                        }}
                        title={log.loggerName}
                      >
                        {log.loggerName.split(".").pop()}
                      </td>
                      <td
                        className={getLevelColor(log.level)}
                        style={{ wordBreak: "break-word" }}
                      >
                        {log.message}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default SystemPanel;
