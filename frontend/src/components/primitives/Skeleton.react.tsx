interface SkeletonProps {
  width?: string | number;
  height?: string | number;
  borderRadius?: string | number;
  className?: string;
}

export const Skeleton = ({
  width = "100%",
  height = "20px",
  borderRadius = "4px",
  className = "",
}: SkeletonProps) => {
  return (
    <div
      className={`skeleton ${className}`}
      style={{
        width,
        height,
        borderRadius,
        backgroundColor: "#e9ecef",
        animation: "pulse 1.5s ease-in-out infinite",
      }}
    />
  );
};

export const SkeletonCard = () => {
  return (
    <div className="card shadow-sm">
      <div className="card-header bg-body-tertiary p-2">
        <Skeleton width="150px" height="16px" />
      </div>
      <div className="card-body p-3">
        <div className="d-flex flex-column gap-2">
          <Skeleton width="100%" height="16px" />
          <Skeleton width="80%" height="16px" />
          <Skeleton width="60%" height="16px" />
          <div className="mt-2">
            <Skeleton width="100%" height="40px" borderRadius="8px" />
          </div>
        </div>
      </div>
    </div>
  );
};

export const SkeletonChannelCard = () => {
  return (
    <div className="card shadow-sm h-100">
      <div className="card-header bg-body-tertiary p-2">
        <div className="d-flex justify-content-between align-items-center">
          <Skeleton width="120px" height="16px" />
          <div className="d-flex gap-1">
            <Skeleton width="70px" height="32px" borderRadius="4px" />
            <Skeleton width="32px" height="32px" borderRadius="4px" />
            <Skeleton width="32px" height="32px" borderRadius="4px" />
          </div>
        </div>
      </div>
      <div className="card-body p-2">
        <div className="d-flex flex-column gap-2">
          <Skeleton width="60%" height="14px" />
          <Skeleton width="100%" height="20px" />
          <div className="border rounded p-2 bg-light">
            <Skeleton width="100%" height="18px" />
            <div className="mt-1 d-flex gap-2">
              <Skeleton width="80px" height="12px" />
              <Skeleton width="80px" height="12px" />
            </div>
          </div>
          <Skeleton width="100%" height="36px" borderRadius="4px" />
        </div>
      </div>
      <div className="card-footer p-1 bg-body-tertiary">
        <div className="d-flex justify-content-between align-items-center">
          <Skeleton width="50px" height="12px" />
          <Skeleton width="80px" height="12px" />
        </div>
      </div>
    </div>
  );
};

export default Skeleton;
