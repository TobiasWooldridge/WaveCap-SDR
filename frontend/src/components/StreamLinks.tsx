import { useState } from "react";
import { Copy, CheckCircle, Link } from "lucide-react";
import Button from "./primitives/Button.react";

/**
 * Configuration for a stream format option.
 */
export interface StreamFormat {
  key: string;
  label: string;
  buildUrl: (baseUrl: string) => string;
}

interface StreamLinksProps {
  /** Array of available stream formats */
  formats: StreamFormat[];
  /** Base URL for building stream URLs (without protocol for WebSocket URLs) */
  baseUrl: string;
  /** Label for the trigger button */
  buttonLabel?: string;
  /** Callback when a URL is copied */
  onCopyUrl: (url: string) => void;
}

/**
 * Helper to build a WebSocket URL from a path.
 */
export function buildWsUrl(path: string): string {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}${path}`;
}

/**
 * Helper to build an HTTP URL from a path.
 */
export function buildHttpUrl(path: string): string {
  return `${window.location.origin}${path}`;
}

// ============================================================================
// Format Presets
// ============================================================================

/** Stream formats for regular channel audio */
export const CHANNEL_STREAM_FORMATS: StreamFormat[] = [
  { key: "pcm", label: "Raw PCM", buildUrl: (base) => `${base}.pcm` },
  { key: "mp3", label: "MP3 (128k)", buildUrl: (base) => `${base}.mp3` },
  { key: "opus", label: "Opus", buildUrl: (base) => `${base}.opus` },
  { key: "aac", label: "AAC", buildUrl: (base) => `${base}.aac` },
];

/** Stream formats for trunking system-level voice (all calls multiplexed) */
export const TRUNKING_SYSTEM_STREAM_FORMATS: StreamFormat[] = [
  {
    key: "ws-all",
    label: "WebSocket (all calls)",
    buildUrl: (base) => buildWsUrl(base),
  },
  {
    key: "ws-json",
    label: "WebSocket + JSON metadata",
    buildUrl: (base) => `${buildWsUrl(base)}?format=json`,
  },
];

/** Stream formats for individual trunking voice streams */
export const TRUNKING_VOICE_STREAM_FORMATS: StreamFormat[] = [
  {
    key: "pcm",
    label: "HTTP PCM",
    buildUrl: (base) => buildHttpUrl(`${base}.pcm`),
  },
  {
    key: "ws-json",
    label: "WebSocket + JSON metadata",
    buildUrl: (base) => buildWsUrl(base),
  },
];

/**
 * Shared component for displaying and copying stream URLs.
 * Used by both channel cards and trunking UI.
 */
export function StreamLinks({
  formats,
  baseUrl,
  buttonLabel = "Copy Stream URL",
  onCopyUrl,
}: StreamLinksProps) {
  const [showModal, setShowModal] = useState(false);
  const [copiedKey, setCopiedKey] = useState<string | null>(null);

  const handleCopy = (format: StreamFormat) => {
    const url = format.buildUrl(baseUrl);
    onCopyUrl(url);
    setCopiedKey(format.key);
    setTimeout(() => setCopiedKey(null), 2000);
  };

  return (
    <>
      <Button
        use="secondary"
        size="sm"
        onClick={() => setShowModal(true)}
        className="w-100 d-flex justify-content-between align-items-center"
      >
        <span className="small">{buttonLabel}</span>
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
                <h6 className="modal-title">{buttonLabel}</h6>
                <button
                  type="button"
                  className="btn-close btn-close-sm"
                  onClick={() => setShowModal(false)}
                  aria-label="Close"
                />
              </div>
              <div className="modal-body d-flex flex-column gap-2 p-3">
                {formats.map((format) => {
                  const isCopied = copiedKey === format.key;
                  return (
                    <button
                      key={format.key}
                      className="btn btn-outline-secondary d-flex justify-content-between align-items-center"
                      onClick={() => handleCopy(format)}
                    >
                      <span>{format.label}</span>
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
