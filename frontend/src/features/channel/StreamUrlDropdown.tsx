import { useState } from "react";
import { Copy, CheckCircle, ChevronUp } from "lucide-react";
import Button from "../../components/primitives/Button.react";

interface StreamUrlDropdownProps {
  channelId: string;
  onCopyUrl: (url: string) => void;
}

const STREAM_FORMATS = [
  { format: "PCM", ext: ".pcm", label: "Raw PCM" },
  { format: "MP3", ext: ".mp3", label: "MP3 (128k)" },
  { format: "Opus", ext: ".opus", label: "Opus" },
  { format: "AAC", ext: ".aac", label: "AAC" },
];

export function StreamUrlDropdown({ channelId, onCopyUrl }: StreamUrlDropdownProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [copiedFormat, setCopiedFormat] = useState<string | null>(null);

  const handleCopy = (format: string, ext: string) => {
    const url = `${window.location.origin}/api/v1/stream/channels/${channelId}${ext}`;
    onCopyUrl(url);
    setCopiedFormat(format);
    setTimeout(() => setCopiedFormat(null), 2000);
    setIsOpen(false);
  };

  return (
    <div className="dropdown" style={{ position: "relative" }}>
      <Button
        use="secondary"
        size="sm"
        onClick={() => setIsOpen(!isOpen)}
        className="w-100 d-flex justify-content-between align-items-center"
      >
        <span className="small">Copy Stream URL</span>
        <ChevronUp size={12} />
      </Button>

      {isOpen && (
        <div
          className="dropdown-menu show w-100"
          style={{ position: "absolute", bottom: "100%", zIndex: 1000 }}
        >
          {STREAM_FORMATS.map(({ format, ext, label }) => {
            const isCopied = copiedFormat === format;
            return (
              <button
                key={format}
                className="dropdown-item d-flex justify-content-between align-items-center"
                onClick={() => handleCopy(format, ext)}
              >
                <span className="small">{label}</span>
                {isCopied ? (
                  <CheckCircle size={12} className="text-success" />
                ) : (
                  <Copy size={12} />
                )}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
