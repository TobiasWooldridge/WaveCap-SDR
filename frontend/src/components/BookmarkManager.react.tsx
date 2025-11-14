import { useState } from "react";
import { Star, Trash2, Edit2, X, Clock, Save } from "lucide-react";
import { useBookmarks, Bookmark } from "../hooks/useBookmarks";
import { useFrequencyHistory } from "../hooks/useFrequencyHistory";
import { useMemoryBanks } from "../hooks/useMemoryBanks";
import type { Capture, Channel } from "../types";
import Button from "./primitives/Button.react";
import Flex from "./primitives/Flex.react";

interface BookmarkManagerProps {
  currentFrequency: number;
  onTuneToFrequency: (frequency: number) => void;
  currentCapture?: Capture;
  currentChannels?: Channel[];
  onLoadMemoryBank?: (bankId: string) => void;
}

export const BookmarkManager = ({ currentFrequency, onTuneToFrequency, currentCapture, currentChannels, onLoadMemoryBank }: BookmarkManagerProps) => {
  const { bookmarks, addBookmark, updateBookmark, deleteBookmark, getBookmarkByFrequency } = useBookmarks();
  const { getRecentHistory, addToHistory } = useFrequencyHistory();
  const { memoryBanks, saveToMemoryBank, deleteMemoryBank } = useMemoryBanks();
  const [showDropdown, setShowDropdown] = useState(false);
  const [activeTab, setActiveTab] = useState<"bookmarks" | "recent" | "memory">("bookmarks");
  const [showAddModal, setShowAddModal] = useState(false);
  const [showSaveMemoryModal, setShowSaveMemoryModal] = useState(false);
  const [memoryBankName, setMemoryBankName] = useState("");
  const [editingBookmark, setEditingBookmark] = useState<Bookmark | null>(null);
  const [bookmarkName, setBookmarkName] = useState("");
  const [bookmarkNotes, setBookmarkNotes] = useState("");

  const recentHistory = getRecentHistory(20);

  const currentBookmark = getBookmarkByFrequency(currentFrequency);
  const isBookmarked = currentBookmark !== undefined;

  const handleAddBookmark = () => {
    if (!bookmarkName.trim()) return;

    addBookmark({
      name: bookmarkName.trim(),
      frequency: currentFrequency,
      notes: bookmarkNotes.trim() || undefined,
    });

    setBookmarkName("");
    setBookmarkNotes("");
    setShowAddModal(false);
  };

  const handleEditBookmark = (bookmark: Bookmark) => {
    setEditingBookmark(bookmark);
    setBookmarkName(bookmark.name);
    setBookmarkNotes(bookmark.notes || "");
    setShowAddModal(true);
  };

  const handleUpdateBookmark = () => {
    if (!editingBookmark || !bookmarkName.trim()) return;

    updateBookmark(editingBookmark.id, {
      name: bookmarkName.trim(),
      notes: bookmarkNotes.trim() || undefined,
    });

    setEditingBookmark(null);
    setBookmarkName("");
    setBookmarkNotes("");
    setShowAddModal(false);
  };

  const handleDeleteBookmark = (id: string) => {
    deleteBookmark(id);
  };

  const handleTuneToBookmark = (bookmark: Bookmark) => {
    onTuneToFrequency(bookmark.frequency);
    addToHistory({ frequencyHz: bookmark.frequency });
    setShowDropdown(false);
  };

  const handleTuneToRecent = (frequencyHz: number) => {
    onTuneToFrequency(frequencyHz);
    setShowDropdown(false);
  };

  const handleSaveMemoryBank = () => {
    if (!memoryBankName.trim() || !currentCapture || !currentChannels) return;

    saveToMemoryBank(memoryBankName.trim(), currentCapture, currentChannels);
    setMemoryBankName("");
    setShowSaveMemoryModal(false);
  };

  const formatFrequency = (hz: number) => {
    if (hz >= 1e9) {
      return `${(hz / 1e9).toFixed(3)} GHz`;
    } else if (hz >= 1e6) {
      return `${(hz / 1e6).toFixed(3)} MHz`;
    } else if (hz >= 1e3) {
      return `${(hz / 1e3).toFixed(3)} kHz`;
    }
    return `${hz} Hz`;
  };

  return (
    <div style={{ position: "relative" }}>
      <Flex gap={1}>
        <Button
          use={isBookmarked ? "warning" : "secondary"}
          size="sm"
          onClick={() => {
            if (isBookmarked) {
              handleDeleteBookmark(currentBookmark!.id);
            } else {
              setShowAddModal(true);
            }
          }}
          title={isBookmarked ? "Remove bookmark" : "Add bookmark"}
        >
          <Star size={16} fill={isBookmarked ? "currentColor" : "none"} />
        </Button>

        <Button
          use="secondary"
          size="sm"
          onClick={() => setShowDropdown(!showDropdown)}
          title="View bookmarks, history, and memory banks"
        >
          <Clock size={16} />
        </Button>

        {currentCapture && currentChannels && currentChannels.length > 0 && (
          <Button
            use="primary"
            size="sm"
            onClick={() => setShowSaveMemoryModal(true)}
            title="Save current configuration to memory bank"
          >
            <Save size={16} />
          </Button>
        )}
      </Flex>

      {/* Main Dropdown with Tabs */}
      {showDropdown && (
        <div
          style={{
            position: "absolute",
            top: "100%",
            right: 0,
            marginTop: "4px",
            backgroundColor: "white",
            border: "1px solid #dee2e6",
            borderRadius: "4px",
            boxShadow: "0 2px 8px rgba(0,0,0,0.15)",
            minWidth: "350px",
            maxHeight: "450px",
            display: "flex",
            flexDirection: "column",
            zIndex: 1000,
          }}
        >
          {/* Tab Navigation */}
          <div style={{
            display: "flex",
            borderBottom: "2px solid #dee2e6",
            backgroundColor: "#f8f9fa"
          }}>
            <button
              className={`btn btn-sm ${activeTab === "bookmarks" ? "btn-primary" : "btn-light"}`}
              style={{
                flex: 1,
                borderRadius: 0,
                borderTopLeftRadius: "4px",
                fontSize: "12px"
              }}
              onClick={() => setActiveTab("bookmarks")}
            >
              Bookmarks ({bookmarks.length})
            </button>
            <button
              className={`btn btn-sm ${activeTab === "recent" ? "btn-primary" : "btn-light"}`}
              style={{
                flex: 1,
                borderRadius: 0,
                fontSize: "12px"
              }}
              onClick={() => setActiveTab("recent")}
            >
              Recent ({recentHistory.length})
            </button>
            <button
              className={`btn btn-sm ${activeTab === "memory" ? "btn-primary" : "btn-light"}`}
              style={{
                flex: 1,
                borderRadius: 0,
                borderTopRightRadius: "4px",
                fontSize: "12px"
              }}
              onClick={() => setActiveTab("memory")}
            >
              Memory ({memoryBanks.length})
            </button>
          </div>

          {/* Tab Content */}
          <div style={{ overflowY: "auto", maxHeight: "380px" }}>
            {/* Bookmarks Tab */}
            {activeTab === "bookmarks" && (
              <div>
                {bookmarks.length === 0 ? (
                  <div style={{ padding: "20px", textAlign: "center", color: "#6c757d" }}>
                    <Star size={32} style={{ opacity: 0.3, marginBottom: "8px" }} />
                    <div style={{ fontSize: "12px" }}>No bookmarks yet</div>
                    <div style={{ fontSize: "11px", marginTop: "4px" }}>Click the star icon to add a bookmark</div>
                  </div>
                ) : (
                  bookmarks.map((bookmark) => (
                    <div
                      key={bookmark.id}
                      style={{
                        padding: "8px 12px",
                        borderBottom: "1px solid #f0f0f0",
                        cursor: "pointer",
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.backgroundColor = "#f8f9fa";
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.backgroundColor = "white";
                      }}
                    >
                      <Flex justify="between" align="center">
                        <div
                          style={{ flex: 1 }}
                          onClick={() => handleTuneToBookmark(bookmark)}
                        >
                          <div style={{ fontWeight: 500 }}>{bookmark.name}</div>
                          <div style={{ fontSize: "12px", color: "#6c757d" }}>
                            {formatFrequency(bookmark.frequency)}
                          </div>
                          {bookmark.notes && (
                            <div style={{ fontSize: "12px", color: "#6c757d", marginTop: "2px" }}>
                              {bookmark.notes}
                            </div>
                          )}
                        </div>
                        <Flex gap={1}>
                          <button
                            className="btn btn-sm btn-icon-sm"
                            style={{ padding: "2px 6px" }}
                            onClick={(e) => {
                              e.stopPropagation();
                              handleEditBookmark(bookmark);
                              setShowDropdown(false);
                            }}
                            title="Edit bookmark"
                          >
                            <Edit2 size={14} />
                          </button>
                          <button
                            className="btn btn-sm btn-danger btn-icon-sm"
                            style={{ padding: "2px 6px" }}
                            onClick={(e) => {
                              e.stopPropagation();
                              handleDeleteBookmark(bookmark.id);
                            }}
                            title="Delete bookmark"
                          >
                            <Trash2 size={14} />
                          </button>
                        </Flex>
                      </Flex>
                    </div>
                  ))
                )}
              </div>
            )}

            {/* Recent Tab */}
            {activeTab === "recent" && (
              <div>
                {recentHistory.length === 0 ? (
                  <div style={{ padding: "20px", textAlign: "center", color: "#6c757d" }}>
                    <Clock size={32} style={{ opacity: 0.3, marginBottom: "8px" }} />
                    <div style={{ fontSize: "12px" }}>No recent history</div>
                  </div>
                ) : (
                  recentHistory.map((entry) => (
                    <div
                      key={`${entry.frequencyHz}-${entry.timestamp}`}
                      style={{
                        padding: "8px 12px",
                        borderBottom: "1px solid #f0f0f0",
                        cursor: "pointer",
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.backgroundColor = "#f8f9fa";
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.backgroundColor = "white";
                      }}
                      onClick={() => handleTuneToRecent(entry.frequencyHz)}
                    >
                      <Flex justify="between" align="center">
                        <div style={{ flex: 1 }}>
                          <div style={{ fontWeight: 500, fontSize: "14px" }}>
                            {formatFrequency(entry.frequencyHz)}
                          </div>
                          <div style={{ fontSize: "11px", color: "#6c757d" }}>
                            {new Date(entry.timestamp).toLocaleString()}
                            {entry.captureName && ` • ${entry.captureName}`}
                            {entry.mode && ` • ${entry.mode.toUpperCase()}`}
                          </div>
                        </div>
                      </Flex>
                    </div>
                  ))
                )}
              </div>
            )}

            {/* Memory Tab */}
            {activeTab === "memory" && (
              <div>
                {memoryBanks.length === 0 ? (
                  <div style={{ padding: "20px", textAlign: "center", color: "#6c757d" }}>
                    <Save size={32} style={{ opacity: 0.3, marginBottom: "8px" }} />
                    <div style={{ fontSize: "12px" }}>No saved memory banks</div>
                    <div style={{ fontSize: "11px", marginTop: "4px" }}>Click the save icon to save current configuration</div>
                  </div>
                ) : (
                  memoryBanks.map((bank) => (
                    <div
                      key={bank.id}
                      style={{
                        padding: "8px 12px",
                        borderBottom: "1px solid #f0f0f0",
                      }}
                    >
                      <Flex justify="between" align="center">
                        <div style={{ flex: 1 }}>
                          <div style={{ fontWeight: 500 }}>{bank.name}</div>
                          <div style={{ fontSize: "11px", color: "#6c757d" }}>
                            {formatFrequency(bank.captureConfig.centerHz)} •
                            {(bank.captureConfig.sampleRate / 1e6).toFixed(1)} MS/s •
                            {bank.channels.length} channel{bank.channels.length !== 1 ? 's' : ''}
                          </div>
                          <div style={{ fontSize: "10px", color: "#6c757d", marginTop: "2px" }}>
                            {new Date(bank.timestamp).toLocaleString()}
                          </div>
                        </div>
                        <Flex gap={1}>
                          {onLoadMemoryBank && (
                            <button
                              className="btn btn-sm btn-primary"
                              style={{ padding: "2px 8px", fontSize: "11px" }}
                              onClick={() => {
                                onLoadMemoryBank(bank.id);
                                setShowDropdown(false);
                              }}
                              title="Load this memory bank"
                            >
                              Load
                            </button>
                          )}
                          <button
                            className="btn btn-sm btn-danger btn-icon-sm"
                            style={{ padding: "2px 6px" }}
                            onClick={(e) => {
                              e.stopPropagation();
                              if (confirm(`Delete memory bank "${bank.name}"?`)) {
                                deleteMemoryBank(bank.id);
                              }
                            }}
                            title="Delete memory bank"
                          >
                            <Trash2 size={14} />
                          </button>
                        </Flex>
                      </Flex>
                    </div>
                  ))
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Add/Edit Bookmark Modal */}
      {showAddModal && (
        <div
          style={{
            position: "fixed",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: "rgba(0,0,0,0.5)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 2000,
          }}
          onClick={() => {
            setShowAddModal(false);
            setEditingBookmark(null);
            setBookmarkName("");
            setBookmarkNotes("");
          }}
        >
          <div
            style={{
              backgroundColor: "white",
              borderRadius: "8px",
              padding: "20px",
              maxWidth: "400px",
              width: "90%",
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <Flex justify="between" align="center" className="mb-3">
              <h5 className="mb-0">
                {editingBookmark ? "Edit Bookmark" : "Add Bookmark"}
              </h5>
              <button
                className="btn btn-sm btn-icon-sm"
                onClick={() => {
                  setShowAddModal(false);
                  setEditingBookmark(null);
                  setBookmarkName("");
                  setBookmarkNotes("");
                }}
              >
                <X size={20} />
              </button>
            </Flex>

            <div className="mb-3">
              <label className="form-label small">Name</label>
              <input
                type="text"
                className="form-control form-control-sm"
                value={bookmarkName}
                onChange={(e) => setBookmarkName(e.target.value)}
                placeholder="e.g., KEXP 90.3 FM"
                autoFocus
              />
            </div>

            <div className="mb-3">
              <label className="form-label small">Frequency</label>
              <input
                type="text"
                className="form-control form-control-sm"
                value={formatFrequency(editingBookmark?.frequency || currentFrequency)}
                disabled
              />
            </div>

            <div className="mb-3">
              <label className="form-label small">Notes (optional)</label>
              <textarea
                className="form-control form-control-sm"
                value={bookmarkNotes}
                onChange={(e) => setBookmarkNotes(e.target.value)}
                placeholder="e.g., Local indie radio station"
                rows={2}
              />
            </div>

            <Flex gap={2} justify="end">
              <Button
                use="secondary"
                size="sm"
                onClick={() => {
                  setShowAddModal(false);
                  setEditingBookmark(null);
                  setBookmarkName("");
                  setBookmarkNotes("");
                }}
              >
                Cancel
              </Button>
              <Button
                use="primary"
                size="sm"
                onClick={editingBookmark ? handleUpdateBookmark : handleAddBookmark}
                disabled={!bookmarkName.trim()}
              >
                {editingBookmark ? "Update" : "Add"}
              </Button>
            </Flex>
          </div>
        </div>
      )}

      {/* Save Memory Bank Modal */}
      {showSaveMemoryModal && (
        <div
          style={{
            position: "fixed",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: "rgba(0,0,0,0.5)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 2000,
          }}
          onClick={() => {
            setShowSaveMemoryModal(false);
            setMemoryBankName("");
          }}
        >
          <div
            style={{
              backgroundColor: "white",
              borderRadius: "8px",
              padding: "20px",
              maxWidth: "400px",
              width: "90%",
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <Flex justify="between" align="center" className="mb-3">
              <h5 className="mb-0">Save to Memory Bank</h5>
              <button
                className="btn btn-sm btn-icon-sm"
                onClick={() => {
                  setShowSaveMemoryModal(false);
                  setMemoryBankName("");
                }}
              >
                <X size={20} />
              </button>
            </Flex>

            <div className="mb-3">
              <label className="form-label small">Name</label>
              <input
                type="text"
                className="form-control form-control-sm"
                value={memoryBankName}
                onChange={(e) => setMemoryBankName(e.target.value)}
                placeholder="e.g., Local Public Safety, Ham Bands"
                autoFocus
              />
            </div>

            {currentCapture && (
              <div className="mb-3">
                <div className="small text-muted">This will save:</div>
                <ul className="small mb-0" style={{ paddingLeft: "20px" }}>
                  <li>Center: {formatFrequency(currentCapture.centerHz)}</li>
                  <li>Sample Rate: {(currentCapture.sampleRate / 1e6).toFixed(1)} MS/s</li>
                  <li>Channels: {currentChannels?.length || 0}</li>
                  {currentCapture.gain !== null && <li>Gain: {currentCapture.gain} dB</li>}
                </ul>
              </div>
            )}

            <Flex gap={2} justify="end">
              <Button
                use="secondary"
                size="sm"
                onClick={() => {
                  setShowSaveMemoryModal(false);
                  setMemoryBankName("");
                }}
              >
                Cancel
              </Button>
              <Button
                use="primary"
                size="sm"
                onClick={handleSaveMemoryBank}
                disabled={!memoryBankName.trim()}
              >
                Save
              </Button>
            </Flex>
          </div>
        </div>
      )}
    </div>
  );
};
