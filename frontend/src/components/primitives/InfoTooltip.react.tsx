import { Info } from "lucide-react";
import { useState, useRef, useEffect } from "react";

interface InfoTooltipProps {
  content: string;
}

export default function InfoTooltip({ content }: InfoTooltipProps) {
  const [isVisible, setIsVisible] = useState(false);
  const [position, setPosition] = useState<"top" | "bottom">("bottom");
  const tooltipRef = useRef<HTMLDivElement>(null);
  const iconRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (isVisible && iconRef.current) {
      const rect = iconRef.current.getBoundingClientRect();
      const spaceBelow = window.innerHeight - rect.bottom;
      const spaceAbove = rect.top;

      // Show above if not enough space below
      setPosition(spaceBelow < 150 && spaceAbove > spaceBelow ? "top" : "bottom");
    }
  }, [isVisible]);

  return (
    <span className="position-relative d-inline-block ms-1">
      <span
        ref={iconRef}
        className="d-inline-flex align-items-center"
        onMouseEnter={() => setIsVisible(true)}
        onMouseLeave={() => setIsVisible(false)}
        style={{ cursor: "help" }}
      >
        <Info size={14} className="text-muted opacity-75" />
      </span>
      {isVisible && (
        <div
          ref={tooltipRef}
          className="position-absolute start-50 translate-middle-x p-2 rounded shadow-sm"
          style={{
            [position === "top" ? "bottom" : "top"]: "100%",
            marginTop: position === "bottom" ? "4px" : undefined,
            marginBottom: position === "top" ? "4px" : undefined,
            backgroundColor: "#f8f9fa",
            border: "1px solid #dee2e6",
            fontSize: "12px",
            width: "250px",
            zIndex: 1000,
            pointerEvents: "none",
          }}
        >
          <div className="text-dark">{content}</div>
        </div>
      )}
    </span>
  );
}
