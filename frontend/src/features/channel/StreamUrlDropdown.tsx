import { useState } from "react";
import { Copy, CheckCircle, Link } from "lucide-react";
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
  const [showModal, setShowModal] = useState(false);
  const [copiedFormat, setCopiedFormat] = useState<string | null>(null);

  const handleCopy = (format: string, ext: string) => {
    const url = `${window.location.origin}/api/v1/stream/channels/${channelId}${ext}`;
    onCopyUrl(url);
    setCopiedFormat(format);
    setTimeout(() => setCopiedFormat(null), 2000);
  };

  return (
    <>
      <Button
        use="secondary"
        size="sm"
        onClick={() => setShowModal(true)}
        className="w-100 d-flex justify-content-between align-items-center"
      >
        <span className="small">Copy Stream URL</span>
        <Link size={12} />
      </Button>

      {showModal && (
        <div
          className="modal d-block"
          style={{ backgroundColor: "rgba(0,0,0,0.5)" }}
          onClick={() => setShowModal(false)}
        >
          <div
            className="modal-dialog modal-dialog-centered modal-sm"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="modal-content">
              <div className="modal-header py-2">
                <h6 className="modal-title">Copy Stream URL</h6>
                <button
                  type="button"
                  className="btn-close btn-close-sm"
                  onClick={() => setShowModal(false)}
                  aria-label="Close"
                />
              </div>
              <div className="modal-body d-flex flex-column gap-2 p-3">
                {STREAM_FORMATS.map(({ format, ext, label }) => {
                  const isCopied = copiedFormat === format;
                  return (
                    <button
                      key={format}
                      className="btn btn-outline-secondary d-flex justify-content-between align-items-center"
                      onClick={() => handleCopy(format, ext)}
                    >
                      <span>{label}</span>
                      {isCopied ? (
                        <CheckCircle size={14} className="text-success" />
                      ) : (
                        <Copy size={14} />
                      )}
                    </button>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
