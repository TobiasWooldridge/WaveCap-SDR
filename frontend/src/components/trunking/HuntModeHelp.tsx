/**
 * HuntModeHelp - Explanatory text for control channel hunting modes
 */
import { Lock } from "lucide-react";

export function HuntModeHelp() {
  return (
    <div
      className="text-muted small mb-2 px-1"
      style={{ fontSize: "0.7rem", lineHeight: 1.3 }}
    >
      <p className="mb-1">
        <strong>Control channels</strong> carry P25 trunking signaling. The system hunts
        for the best channel based on signal quality.
      </p>
      <p className="mb-1">
        <strong>Modes:</strong>{" "}
        <em>Auto</em> switches channels if signal degrades.{" "}
        <em>Manual</em> locks to one channel.{" "}
        <em>Scan Once</em> finds the best channel then stays.
      </p>
      <p className="mb-0">
        Use the <Lock size={10} className="mx-1" /> button to lock/unlock a specific
        channel.
      </p>
    </div>
  );
}
