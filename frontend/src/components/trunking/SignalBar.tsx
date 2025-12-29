/**
 * SignalBar - Visual SNR indicator with progress bar and dB value
 */

interface SignalBarProps {
  snrDb: number | null;
  maxDb?: number;
  width?: string;
}

export function SignalBar({ snrDb, maxDb = 30, width = "80px" }: SignalBarProps) {
  if (snrDb === null) return <span className="text-muted">---</span>;

  const percent = Math.min(100, Math.max(0, (snrDb / maxDb) * 100));

  let color = "bg-danger";
  if (percent > 60) color = "bg-success";
  else if (percent > 30) color = "bg-warning";

  return (
    <div className="d-flex align-items-center gap-1" style={{ width }}>
      <div
        className="progress"
        style={{ height: "8px", width: "50px", backgroundColor: "#333" }}
      >
        <div
          className={`progress-bar ${color}`}
          style={{ width: `${percent}%` }}
        />
      </div>
      <small className="text-muted" style={{ fontSize: "0.7rem", minWidth: "35px" }}>
        {snrDb.toFixed(0)} dB
      </small>
    </div>
  );
}
