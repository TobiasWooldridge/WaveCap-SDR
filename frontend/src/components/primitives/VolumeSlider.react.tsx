import { Volume2, VolumeX } from "lucide-react";

interface VolumeSliderProps {
  value: number;  // 0-1
  onChange: (value: number) => void;
  width?: number;
}

export default function VolumeSlider({ value, onChange, width = 80 }: VolumeSliderProps) {
  const isMuted = value === 0;

  const toggleMute = () => {
    onChange(isMuted ? 0.5 : 0);
  };

  return (
    <div className="d-flex align-items-center gap-1">
      <button
        className="btn btn-sm p-0 border-0 bg-transparent"
        onClick={toggleMute}
        title={isMuted ? "Unmute" : "Mute"}
        style={{ lineHeight: 1 }}
      >
        {isMuted ? (
          <VolumeX size={16} className="text-muted" />
        ) : (
          <Volume2 size={16} className="text-primary" />
        )}
      </button>
      <input
        type="range"
        className="form-range"
        min={0}
        max={1}
        step={0.01}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        style={{ width: `${width}px` }}
      />
    </div>
  );
}
