import { Radio, Antenna, Binary, Activity } from "lucide-react";

export type ViewMode = "radio" | "trunking" | "digital" | "system";

interface ModeTabBarProps {
  activeMode: ViewMode;
  onModeChange: (mode: ViewMode) => void;
  hasTrunking: boolean; // Whether trunking is configured for this device
  hasDigital?: boolean; // Whether digital modes are available (future)
}

interface ModeTab {
  id: ViewMode;
  label: string;
  icon: React.ReactNode;
  enabled: boolean;
  tooltip?: string;
}

export function ModeTabBar({
  activeMode,
  onModeChange,
  hasTrunking,
  hasDigital = false,
}: ModeTabBarProps) {
  const tabs: ModeTab[] = [
    {
      id: "radio",
      label: "Radio",
      icon: <Radio size={14} />,
      enabled: true,
      tooltip: "Spectrum analyzer, tuning controls, and channel list",
    },
    {
      id: "trunking",
      label: "Trunking",
      icon: <Antenna size={14} />,
      enabled: hasTrunking,
      tooltip: hasTrunking
        ? "P25 trunking system control and monitoring"
        : "No trunking system configured for this device",
    },
    {
      id: "digital",
      label: "Digital",
      icon: <Binary size={14} />,
      enabled: hasDigital,
      tooltip: hasDigital
        ? "Digital mode decoders (POCSAG, DMR, etc.)"
        : "Digital decoders not yet available",
    },
    {
      id: "system",
      label: "System",
      icon: <Activity size={14} />,
      enabled: true,
      tooltip: "System metrics, logs, and diagnostics",
    },
  ];

  return (
    <div className="d-flex border-bottom bg-body-secondary">
      <nav className="nav nav-pills nav-fill px-2 py-1" style={{ gap: "4px" }}>
        {tabs.map((tab) => (
          <button
            key={tab.id}
            className={`nav-link d-flex align-items-center gap-1 py-1 px-3 ${
              activeMode === tab.id ? "active" : ""
            } ${!tab.enabled ? "disabled text-muted" : ""}`}
            onClick={() => tab.enabled && onModeChange(tab.id)}
            disabled={!tab.enabled}
            title={tab.tooltip}
            style={{
              fontSize: "0.8rem",
              borderRadius: "4px",
              cursor: tab.enabled ? "pointer" : "not-allowed",
              opacity: tab.enabled ? 1 : 0.5,
            }}
          >
            {tab.icon}
            {tab.label}
          </button>
        ))}
      </nav>
    </div>
  );
}
