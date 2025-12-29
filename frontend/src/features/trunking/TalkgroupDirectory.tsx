import { useState, useMemo } from "react";
import { Users, Search, Filter, Volume2, VolumeX, Circle } from "lucide-react";
import type { Talkgroup } from "../../types/trunking";

interface TalkgroupDirectoryProps {
  talkgroups: Talkgroup[];
  onToggleMonitor?: (tgid: number, monitor: boolean) => void;
  activeTalkgroups?: Set<number>; // Currently active talkgroups
}

type SortField = "tgid" | "name" | "category" | "priority";
type SortOrder = "asc" | "desc";

export function TalkgroupDirectory({
  talkgroups,
  onToggleMonitor,
  activeTalkgroups = new Set(),
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
        className="table-responsive"
        style={{ maxHeight: 400, overflowY: "auto" }}
      >
        <table className="table table-sm table-hover mb-0">
          <thead className="sticky-top bg-body">
            <tr>
              <th style={{ width: 30 }}></th>
              <SortHeader field="tgid" label="TGID" className="text-end" />
              <SortHeader
                field="name"
                label="Name"
                style={{ minWidth: "180px" }}
              />
              <SortHeader field="category" label="Category" />
              <SortHeader
                field="priority"
                label="Priority"
                className="text-center"
              />
              {onToggleMonitor && <th style={{ width: 40 }}></th>}
            </tr>
          </thead>
          <tbody>
            {filteredTalkgroups.map((tg) => {
              const isActive = activeTalkgroups.has(tg.tgid);
              return (
                <tr
                  key={tg.tgid}
                  className={isActive ? "table-success" : undefined}
                >
                  <td className="text-center">
                    {isActive && (
                      <Circle
                        size={8}
                        className="text-success"
                        fill="currentColor"
                      />
                    )}
                  </td>
                  <td className="text-end font-monospace">{tg.tgid}</td>
                  <td style={{ wordBreak: "break-word" }}>
                    <div className="fw-semibold" title={tg.name}>
                      {tg.name}
                    </div>
                    {/* Only show alphaTag if it's meaningfully different from name
                        (not just a truncated/mangled version) */}
                    {tg.alphaTag &&
                      !tg.name
                        .toUpperCase()
                        .replace(/\s+/g, "_")
                        .startsWith(tg.alphaTag) && (
                        <small className="text-muted" title={tg.alphaTag}>
                          {tg.alphaTag}
                        </small>
                      )}
                  </td>
                  <td>
                    {tg.category && (
                      <span className="badge bg-secondary">{tg.category}</span>
                    )}
                  </td>
                  <td className="text-center">
                    <span
                      className={`badge ${
                        tg.priority <= 3
                          ? "bg-danger"
                          : tg.priority <= 6
                            ? "bg-warning text-dark"
                            : "bg-secondary"
                      }`}
                    >
                      P{tg.priority}
                    </span>
                  </td>
                  {onToggleMonitor && (
                    <td className="text-center">
                      <button
                        className={`btn btn-sm ${
                          tg.monitor ? "btn-primary" : "btn-outline-secondary"
                        }`}
                        onClick={() => onToggleMonitor(tg.tgid, !tg.monitor)}
                        title={
                          tg.monitor ? "Stop monitoring" : "Start monitoring"
                        }
                      >
                        {tg.monitor ? (
                          <Volume2 size={14} />
                        ) : (
                          <VolumeX size={14} />
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
