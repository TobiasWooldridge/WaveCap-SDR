import { useState, useMemo } from "react";
import { Users, Search, Filter, Volume2, VolumeX, Circle } from "lucide-react";
import type { Talkgroup } from "../../types/trunking";

interface TalkgroupDirectoryProps {
  talkgroups: Talkgroup[];
  onToggleMonitor?: (tgid: number, monitor: boolean) => void;
  activeTalkgroups?: Set<number>; // Currently active talkgroups
  lastSeenMap?: Map<number, number>; // TGID -> timestamp (seconds)
}

type SortField = "tgid" | "name" | "category" | "priority" | "lastSeen";

function formatLastSeen(timestamp: number | undefined): string {
  if (!timestamp) return "";
  const now = Date.now() / 1000;
  const diff = now - timestamp;

  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}
type SortOrder = "asc" | "desc";

export function TalkgroupDirectory({
  talkgroups,
  onToggleMonitor,
  activeTalkgroups = new Set(),
  lastSeenMap = new Map(),
}: TalkgroupDirectoryProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [categoryFilter, setCategoryFilter] = useState<string | null>(null);
  const [sortField, setSortField] = useState<SortField>("priority");
  const [sortOrder, setSortOrder] = useState<SortOrder>("asc");
  const [showOnlyMonitored, setShowOnlyMonitored] = useState(false);

  // Get unique categories
  const categories = useMemo(() => {
    const cats = new Set<string>();
    talkgroups.forEach((tg) => {
      if (tg.category) cats.add(tg.category);
    });
    return Array.from(cats).sort();
  }, [talkgroups]);

  // Filter and sort talkgroups
  const filteredTalkgroups = useMemo(() => {
    let filtered = [...talkgroups];

    // Search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(
        (tg) =>
          tg.name.toLowerCase().includes(query) ||
          tg.alphaTag.toLowerCase().includes(query) ||
          tg.tgid.toString().includes(query),
      );
    }

    // Category filter
    if (categoryFilter) {
      filtered = filtered.filter((tg) => tg.category === categoryFilter);
    }

    // Monitored filter
    if (showOnlyMonitored) {
      filtered = filtered.filter((tg) => tg.monitor);
    }

    // Sort
    filtered.sort((a, b) => {
      let cmp = 0;
      switch (sortField) {
        case "tgid":
          cmp = a.tgid - b.tgid;
          break;
        case "name":
          cmp = a.name.localeCompare(b.name);
          break;
        case "category":
          cmp = (a.category || "").localeCompare(b.category || "");
          break;
        case "priority":
          cmp = a.priority - b.priority;
          break;
        case "lastSeen":
          // Sort by last seen (most recent first by default)
          cmp = (lastSeenMap.get(b.tgid) || 0) - (lastSeenMap.get(a.tgid) || 0);
          break;
      }
      return sortOrder === "asc" ? cmp : -cmp;
    });

    return filtered;
  }, [
    talkgroups,
    searchQuery,
    categoryFilter,
    sortField,
    sortOrder,
    showOnlyMonitored,
    lastSeenMap,
  ]);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortOrder(sortOrder === "asc" ? "desc" : "asc");
    } else {
      setSortField(field);
      setSortOrder("asc");
    }
  };

  const SortHeader = ({
    field,
    label,
    className = "",
    style = {},
  }: {
    field: SortField;
    label: string;
    className?: string;
    style?: React.CSSProperties;
  }) => (
    <th
      className={`${className} cursor-pointer user-select-none`}
      onClick={() => handleSort(field)}
      style={{ cursor: "pointer", ...style }}
    >
      {label}
      {sortField === field && (
        <span className="ms-1">{sortOrder === "asc" ? "▲" : "▼"}</span>
      )}
    </th>
  );

  if (talkgroups.length === 0) {
    return (
      <div className="text-center text-muted py-4">
        <Users size={32} className="mb-2 opacity-50" />
        <p className="small mb-0">No talkgroups configured</p>
      </div>
    );
  }

  return (
    <div>
      {/* Filters */}
      <div className="d-flex flex-wrap gap-2 mb-2">
        {/* Search */}
        <div className="input-group input-group-sm" style={{ maxWidth: 200 }}>
          <span className="input-group-text">
            <Search size={14} />
          </span>
          <input
            type="text"
            className="form-control"
            placeholder="Search..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>

        {/* Category filter */}
        <select
          className="form-select form-select-sm"
          style={{ maxWidth: 150 }}
          value={categoryFilter || ""}
          onChange={(e) => setCategoryFilter(e.target.value || null)}
        >
          <option value="">All Categories</option>
          {categories.map((cat) => (
            <option key={cat} value={cat}>
              {cat}
            </option>
          ))}
        </select>

        {/* Monitored only toggle */}
        <button
          className={`btn btn-sm ${
            showOnlyMonitored ? "btn-primary" : "btn-outline-secondary"
          }`}
          onClick={() => setShowOnlyMonitored(!showOnlyMonitored)}
          title="Show only monitored talkgroups"
        >
          <Filter size={14} className="me-1" />
          Monitored
        </button>

        <span className="text-muted small align-self-center ms-auto">
          {filteredTalkgroups.length} / {talkgroups.length} talkgroups
        </span>
      </div>

      {/* Table */}
      <div
        className="table-responsive bg-body-tertiary rounded font-monospace"
        style={{ maxHeight: 400, overflowY: "auto", fontSize: "0.7rem" }}
      >
        <table className="table table-sm table-hover mb-0">
          <thead className="sticky-top bg-body-secondary">
            <tr>
              <th style={{ width: 20 }}></th>
              <SortHeader
                field="tgid"
                label="TG"
                className="text-end"
                style={{ width: 50 }}
              />
              <SortHeader field="name" label="Name" />
              <SortHeader field="category" label="Cat" style={{ width: 80 }} />
              <SortHeader
                field="priority"
                label="Pri"
                className="text-center"
                style={{ width: 35 }}
              />
              <SortHeader
                field="lastSeen"
                label="Last Seen"
                style={{ width: 70 }}
              />
              {onToggleMonitor && <th style={{ width: 30 }}></th>}
            </tr>
          </thead>
          <tbody>
            {filteredTalkgroups.map((tg) => {
              const isActive = activeTalkgroups.has(tg.tgid);
              const lastSeen = lastSeenMap.get(tg.tgid);
              return (
                <tr
                  key={tg.tgid}
                  className={isActive ? "table-success" : undefined}
                >
                  <td className="text-center py-1">
                    {isActive && (
                      <Circle
                        size={6}
                        className="text-success"
                        fill="currentColor"
                      />
                    )}
                  </td>
                  <td className="text-end py-1">{tg.tgid}</td>
                  <td className="py-1" title={tg.name}>
                    {tg.name}
                  </td>
                  <td className="py-1">
                    {tg.category && (
                      <span
                        className="badge bg-secondary"
                        style={{ fontSize: "0.6rem", padding: "1px 3px" }}
                      >
                        {tg.category}
                      </span>
                    )}
                  </td>
                  <td className="text-center py-1">
                    <span
                      className={`badge ${
                        tg.priority <= 3
                          ? "bg-danger"
                          : tg.priority <= 6
                            ? "bg-warning text-dark"
                            : "bg-secondary"
                      }`}
                      style={{ fontSize: "0.55rem", padding: "1px 3px" }}
                    >
                      {tg.priority}
                    </span>
                  </td>
                  <td className="py-1 text-body-secondary">
                    {isActive ? (
                      <span className="text-success fw-semibold">active</span>
                    ) : (
                      formatLastSeen(lastSeen)
                    )}
                  </td>
                  {onToggleMonitor && (
                    <td className="text-center py-1">
                      <button
                        className={`btn btn-sm ${
                          tg.monitor ? "btn-primary" : "btn-outline-secondary"
                        }`}
                        onClick={() => onToggleMonitor(tg.tgid, !tg.monitor)}
                        title={
                          tg.monitor ? "Stop monitoring" : "Start monitoring"
                        }
                        style={{ padding: "0 3px", lineHeight: 1 }}
                      >
                        {tg.monitor ? (
                          <Volume2 size={10} />
                        ) : (
                          <VolumeX size={10} />
                        )}
                      </button>
                    </td>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
