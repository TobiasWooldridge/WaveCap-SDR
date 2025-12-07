import Flex from "../../components/primitives/Flex.react";

interface RdsData {
  psName?: string | null;
  radioText?: string | null;
  ptyName?: string | null;
  piCode?: string | null;
  tp?: boolean;
  ta?: boolean;
  ms?: boolean;
}

interface RdsDisplayProps {
  rdsData: RdsData;
}

export function RdsDisplay({ rdsData }: RdsDisplayProps) {
  if (!rdsData.psName && !rdsData.radioText) {
    return null;
  }

  return (
    <div className="border rounded p-2 bg-dark text-light" style={{ fontSize: "10px" }}>
      <Flex direction="column" gap={1}>
        {/* Station Name (PS) */}
        {rdsData.psName && (
          <Flex align="center" gap={1}>
            <span className="badge bg-info text-dark" style={{ fontSize: "8px", width: "28px" }}>
              RDS
            </span>
            <span
              className="fw-bold font-monospace"
              style={{ fontSize: "14px", letterSpacing: "1px" }}
            >
              {rdsData.psName}
            </span>
            {rdsData.ptyName && rdsData.ptyName !== "None" && (
              <span className="badge bg-secondary ms-auto" style={{ fontSize: "8px" }}>
                {rdsData.ptyName}
              </span>
            )}
          </Flex>
        )}

        {/* Radio Text (RT) */}
        {rdsData.radioText && (
          <div
            className="text-truncate font-monospace text-muted"
            title={rdsData.radioText}
          >
            {rdsData.radioText}
          </div>
        )}

        {/* Flags */}
        {(rdsData.ta || rdsData.tp || rdsData.piCode) && (
          <Flex gap={1} align="center">
            {rdsData.piCode && (
              <span className="text-muted" style={{ fontSize: "8px" }}>
                PI:{rdsData.piCode}
              </span>
            )}
            {rdsData.tp && (
              <span className="badge bg-warning text-dark" style={{ fontSize: "7px" }}>
                TP
              </span>
            )}
            {rdsData.ta && (
              <span className="badge bg-danger" style={{ fontSize: "7px" }}>
                TA
              </span>
            )}
            {!rdsData.ms && (
              <span className="badge bg-primary" style={{ fontSize: "7px" }}>
                Speech
              </span>
            )}
          </Flex>
        )}
      </Flex>
    </div>
  );
}
