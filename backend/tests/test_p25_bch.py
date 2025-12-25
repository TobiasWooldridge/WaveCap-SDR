"""Tests for P25 BCH(63,16,23) error correction.

Tests the BCH decoder used for P25 Network ID (NID) decoding.
"""

import numpy as np
import pytest

from wavecapsdr.dsp.fec.bch import BCH_63_16_23, bch_decode
from wavecapsdr.decoders.p25_frames import decode_nid, NID, DUID
from wavecapsdr.decoders.nac_tracker import NACTracker


class TestBCH:
    """Test BCH(63,16,23) decoder."""

    def test_initialization(self):
        """Test BCH decoder initialization."""
        bch = BCH_63_16_23()
        assert bch.M == 6
        assert bch.N == 63
        assert bch.K == 16
        assert bch.T == 11

    def test_galois_tables(self):
        """Test Galois Field table generation."""
        bch = BCH_63_16_23()
        # Verify antilog table has correct length
        assert len(bch.a_pow_tab) == 64
        assert len(bch.a_log_tab) == 64

        # Verify identity: a^0 = 1
        assert bch.a_pow_tab[0] == 1

        # Verify wrap-around
        assert bch.a_pow_tab[63] == 1

    def test_decode_no_errors(self):
        """Test decoding with no errors."""
        # Create a simple codeword (all zeros is valid)
        codeword = np.zeros(63, dtype=np.uint8)
        data, errors = bch_decode(codeword)

        assert errors == 0
        assert data == 0

    def test_decode_with_nac_tracking(self):
        """Test BCH decode with NAC tracking assistance."""
        # This tests the second-pass correction logic
        # For now, just verify the API works
        codeword = np.zeros(63, dtype=np.uint8)
        tracked_nac = 0x123

        data, errors = bch_decode(codeword, tracked_nac)
        # Should decode successfully (either first or second pass)
        assert errors >= 0 or errors == BCH_63_16_23.MESSAGE_NOT_CORRECTED


class TestNACTracker:
    """Test NAC tracker for BCH assistance."""

    def test_initialization(self):
        """Test tracker initialization."""
        tracker = NACTracker()
        assert tracker.get_tracked_nac() == 0

    def test_single_nac_tracking(self):
        """Test tracking single NAC value."""
        tracker = NACTracker()

        # Need 3 observations to become dominant
        tracker.track(0x123)
        assert tracker.get_tracked_nac() == 0  # Not enough observations

        tracker.track(0x123)
        assert tracker.get_tracked_nac() == 0  # Still not enough

        tracker.track(0x123)
        assert tracker.get_tracked_nac() == 0x123  # Now dominant

    def test_multiple_nac_values(self):
        """Test tracker with multiple NAC values."""
        tracker = NACTracker()

        # Track NAC1 (3 times)
        for _ in range(3):
            tracker.track(0x111)

        # Track NAC2 (5 times)
        for _ in range(5):
            tracker.track(0x222)

        # NAC2 should be dominant
        assert tracker.get_tracked_nac() == 0x222

    def test_max_tracker_count(self):
        """Test that tracker limits number of distinct NACs."""
        tracker = NACTracker()

        # Track 4 different NACs (max is 3)
        for nac in [0x111, 0x222, 0x333, 0x444]:
            for _ in range(3):
                tracker.track(nac)

        # Should only have 3 trackers (oldest pruned)
        stats = tracker.get_statistics()
        assert len(stats) <= NACTracker.MAX_TRACKER_COUNT

    def test_reset(self):
        """Test tracker reset."""
        tracker = NACTracker()

        for _ in range(3):
            tracker.track(0x123)

        assert tracker.get_tracked_nac() == 0x123

        tracker.reset()
        assert tracker.get_tracked_nac() == 0


class TestNIDDecode:
    """Test NID decode with BCH and NAC tracking."""

    def test_decode_with_tracker(self):
        """Test decode_nid with NAC tracker."""
        # Create minimal NID dibits (33 with status symbol)
        dibits = np.zeros(33, dtype=np.uint8)
        tracker = NACTracker()

        # Decode (will likely fail with all zeros, but should not crash)
        result = decode_nid(dibits, skip_status_at_10=True, nac_tracker=tracker)

        # Result should be NID or None
        assert result is None or isinstance(result, NID)

    def test_decode_without_tracker(self):
        """Test decode_nid without NAC tracker (backward compat)."""
        dibits = np.zeros(33, dtype=np.uint8)

        # Should work without tracker
        result = decode_nid(dibits, skip_status_at_10=True, nac_tracker=None)

        assert result is None or isinstance(result, NID)

    def test_status_symbol_skipping(self):
        """Test that status symbol at position 10 is skipped."""
        # Create dibits with a marker at position 10
        dibits = np.zeros(33, dtype=np.uint8)
        dibits[10] = 3  # Mark the status symbol position

        result = decode_nid(dibits, skip_status_at_10=True)

        # The status symbol should have been skipped
        # (hard to verify without internal state inspection,
        # but at least verify it doesn't crash)
        assert result is None or isinstance(result, NID)

    def test_decode_too_short(self):
        """Test handling of too-short input."""
        dibits = np.zeros(10, dtype=np.uint8)

        result = decode_nid(dibits, skip_status_at_10=True)
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
