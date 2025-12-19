"""Identifier resolution and alias matching tests.

Tests alias matching, range matching, priority resolution, and identifier
qualification patterns. Adapted from SDRTrunk's P25AliasTest patterns.

Reference: https://github.com/DSheirer/sdrtrunk
"""

import pytest
from typing import Dict, List, Optional, Tuple

from wavecapsdr.trunking.identifiers import (
    Identifier,
    IdentifierCollection,
    MutableIdentifierCollection,
    IdentifierRole,
    IdentifierForm,
    TalkerAliasManager,
)


# ============================================================================
# Alias Configuration Classes (for testing)
# ============================================================================

class AliasRange:
    """Alias that matches a range of identifiers."""

    def __init__(self, name: str, min_id: int, max_id: int,
                 system_id: Optional[int] = None):
        self.name = name
        self.min_id = min_id
        self.max_id = max_id
        self.system_id = system_id  # Optional qualification

    def matches(self, identifier: int, system_id: Optional[int] = None) -> bool:
        """Check if identifier matches this range."""
        if self.system_id is not None and system_id != self.system_id:
            return False
        return self.min_id <= identifier <= self.max_id


class AliasList:
    """List of aliases with matching logic."""

    def __init__(self):
        self.exact_aliases: Dict[int, str] = {}
        self.range_aliases: List[AliasRange] = []
        self.qualified_aliases: Dict[Tuple[int, int], str] = {}  # (system_id, id) -> name

    def add_exact(self, identifier: int, name: str):
        """Add exact match alias."""
        self.exact_aliases[identifier] = name

    def add_range(self, name: str, min_id: int, max_id: int,
                  system_id: Optional[int] = None):
        """Add range alias."""
        self.range_aliases.append(AliasRange(name, min_id, max_id, system_id))

    def add_qualified(self, system_id: int, identifier: int, name: str):
        """Add fully qualified alias (system + id)."""
        self.qualified_aliases[(system_id, identifier)] = name

    def get_alias(self, identifier: int, system_id: Optional[int] = None) -> Optional[str]:
        """Get alias for identifier with optional system qualification.

        Priority:
        1. Fully qualified match (system + id)
        2. Exact match
        3. Range match (first matching range)
        """
        # Check qualified first (most specific)
        if system_id is not None:
            key = (system_id, identifier)
            if key in self.qualified_aliases:
                return self.qualified_aliases[key]

        # Check exact match
        if identifier in self.exact_aliases:
            return self.exact_aliases[identifier]

        # Check range matches
        for range_alias in self.range_aliases:
            if range_alias.matches(identifier, system_id):
                return range_alias.name

        return None


# ============================================================================
# Exact Alias Matching Tests
# ============================================================================

class TestExactAliasMatching:
    """Test exact identifier alias matching."""

    def test_single_talkgroup_alias(self):
        """Match single talkgroup to alias."""
        aliases = AliasList()
        aliases.add_exact(1001, "Fire Dispatch")

        assert aliases.get_alias(1001) == "Fire Dispatch"
        assert aliases.get_alias(1002) is None

    def test_multiple_talkgroup_aliases(self):
        """Match multiple talkgroups to aliases."""
        aliases = AliasList()
        aliases.add_exact(1001, "Fire Dispatch")
        aliases.add_exact(1002, "EMS")
        aliases.add_exact(1003, "Police TAC 1")

        assert aliases.get_alias(1001) == "Fire Dispatch"
        assert aliases.get_alias(1002) == "EMS"
        assert aliases.get_alias(1003) == "Police TAC 1"

    def test_radio_id_alias(self):
        """Match radio ID to alias."""
        aliases = AliasList()
        aliases.add_exact(12345678, "Engine 1")
        aliases.add_exact(12345679, "Engine 2")

        assert aliases.get_alias(12345678) == "Engine 1"
        assert aliases.get_alias(12345679) == "Engine 2"

    def test_no_match_returns_none(self):
        """Unknown identifier returns None."""
        aliases = AliasList()
        aliases.add_exact(1001, "Fire Dispatch")

        assert aliases.get_alias(9999) is None


# ============================================================================
# Range Alias Matching Tests
# ============================================================================

class TestRangeAliasMatching:
    """Test range-based alias matching (SDRTrunk pattern)."""

    def test_talkgroup_range_match(self):
        """Match talkgroup within range."""
        aliases = AliasList()
        aliases.add_range("Fire Department", 1000, 1099)

        assert aliases.get_alias(1000) == "Fire Department"
        assert aliases.get_alias(1050) == "Fire Department"
        assert aliases.get_alias(1099) == "Fire Department"

    def test_talkgroup_outside_range(self):
        """Talkgroup outside range doesn't match."""
        aliases = AliasList()
        aliases.add_range("Fire Department", 1000, 1099)

        assert aliases.get_alias(999) is None
        assert aliases.get_alias(1100) is None

    def test_multiple_ranges(self):
        """Multiple ranges with different aliases."""
        aliases = AliasList()
        aliases.add_range("Fire Department", 1000, 1099)
        aliases.add_range("Police Department", 2000, 2099)
        aliases.add_range("EMS", 3000, 3099)

        assert aliases.get_alias(1050) == "Fire Department"
        assert aliases.get_alias(2050) == "Police Department"
        assert aliases.get_alias(3050) == "EMS"

    def test_first_range_wins(self):
        """First matching range takes precedence."""
        aliases = AliasList()
        aliases.add_range("Range A", 1000, 2000)
        aliases.add_range("Range B", 1500, 2500)

        # 1500 matches both, first wins
        assert aliases.get_alias(1500) == "Range A"
        # 2200 only matches second
        assert aliases.get_alias(2200) == "Range B"


# ============================================================================
# Qualified Identifier Tests (SDRTrunk pattern)
# ============================================================================

class TestQualifiedIdentifiers:
    """Test fully qualified identifier matching."""

    def test_system_qualified_talkgroup(self):
        """Match talkgroup qualified by system ID."""
        aliases = AliasList()
        aliases.add_qualified(0x123, 1001, "System 123 Fire")
        aliases.add_qualified(0x456, 1001, "System 456 Fire")

        # Same talkgroup, different systems
        assert aliases.get_alias(1001, system_id=0x123) == "System 123 Fire"
        assert aliases.get_alias(1001, system_id=0x456) == "System 456 Fire"

    def test_qualified_takes_precedence(self):
        """Qualified alias takes precedence over exact."""
        aliases = AliasList()
        aliases.add_exact(1001, "Generic Fire")
        aliases.add_qualified(0x123, 1001, "System 123 Fire")

        # With system ID, qualified wins
        assert aliases.get_alias(1001, system_id=0x123) == "System 123 Fire"
        # Without system ID, exact match
        assert aliases.get_alias(1001) == "Generic Fire"

    def test_fallback_to_exact(self):
        """Fall back to exact when no qualified match."""
        aliases = AliasList()
        aliases.add_exact(1001, "Generic Fire")
        aliases.add_qualified(0x123, 1001, "System 123 Fire")

        # Unknown system falls back to exact
        assert aliases.get_alias(1001, system_id=0x999) == "Generic Fire"


# ============================================================================
# Alias Priority/Shadowing Tests
# ============================================================================

class TestAliasPriority:
    """Test alias priority and shadowing."""

    def test_exact_over_range(self):
        """Exact alias takes precedence over range."""
        aliases = AliasList()
        aliases.add_range("Fire Department", 1000, 1099)
        aliases.add_exact(1050, "Fire Chief")

        # Exact match for 1050
        assert aliases.get_alias(1050) == "Fire Chief"
        # Range match for others
        assert aliases.get_alias(1001) == "Fire Department"

    def test_priority_chain(self):
        """Test full priority chain: qualified > exact > range."""
        aliases = AliasList()
        aliases.add_range("Fire Department", 1000, 1099)
        aliases.add_exact(1050, "Fire Unit 50")
        aliases.add_qualified(0x123, 1050, "System 123 Unit 50")

        # All three could match 1050
        assert aliases.get_alias(1050, system_id=0x123) == "System 123 Unit 50"
        assert aliases.get_alias(1050, system_id=0x456) == "Fire Unit 50"
        assert aliases.get_alias(1050) == "Fire Unit 50"
        assert aliases.get_alias(1001) == "Fire Department"


# ============================================================================
# TalkerAliasManager Integration Tests
# ============================================================================

class TestTalkerAliasManagerResolution:
    """Test TalkerAliasManager identifier resolution."""

    def test_enrich_with_radio_alias(self):
        """Enrich collection with radio alias."""
        manager = TalkerAliasManager()
        manager.update_alias(12345678, "Engine 1")

        ic = IdentifierCollection([
            Identifier(12345678, IdentifierRole.FROM, IdentifierForm.RADIO),
            Identifier(1001, IdentifierRole.TO, IdentifierForm.TALKGROUP),
        ])

        enriched = manager.enrich(ic)

        assert enriched.get_alias(IdentifierRole.FROM) == "Engine 1"

    def test_enrich_with_talkgroup_alias(self):
        """Enrich collection with talkgroup alias."""
        manager = TalkerAliasManager()
        manager.update_talkgroup_alias(1001, "Fire Dispatch")

        ic = IdentifierCollection([
            Identifier(12345678, IdentifierRole.FROM, IdentifierForm.RADIO),
            Identifier(1001, IdentifierRole.TO, IdentifierForm.TALKGROUP),
        ])

        enriched = manager.enrich(ic)

        assert enriched.get_alias(IdentifierRole.TO) == "Fire Dispatch"

    def test_enrich_both_aliases(self):
        """Enrich with both radio and talkgroup aliases."""
        manager = TalkerAliasManager()
        manager.update_alias(12345678, "Engine 1")
        manager.update_talkgroup_alias(1001, "Fire Dispatch")

        ic = IdentifierCollection([
            Identifier(12345678, IdentifierRole.FROM, IdentifierForm.RADIO),
            Identifier(1001, IdentifierRole.TO, IdentifierForm.TALKGROUP),
        ])

        enriched = manager.enrich(ic)

        assert enriched.get_alias(IdentifierRole.FROM) == "Engine 1"
        assert enriched.get_alias(IdentifierRole.TO) == "Fire Dispatch"

    def test_update_overwrites_alias(self):
        """Updating alias overwrites previous value."""
        manager = TalkerAliasManager()
        manager.update_alias(12345678, "Engine 1")
        manager.update_alias(12345678, "Engine 1 (Updated)")

        assert manager.get_alias(12345678) == "Engine 1 (Updated)"


# ============================================================================
# Identifier Collection Resolution Tests
# ============================================================================

class TestIdentifierCollectionResolution:
    """Test identifier resolution within collections."""

    def test_prefer_radio_for_from(self):
        """Prefer RADIO form for FROM identifier."""
        ic = IdentifierCollection([
            Identifier("Engine 1", IdentifierRole.FROM, IdentifierForm.ALIAS),
            Identifier(12345678, IdentifierRole.FROM, IdentifierForm.RADIO),
        ])

        from_id = ic.get_from_identifier()
        assert from_id.form == IdentifierForm.RADIO

    def test_prefer_talkgroup_for_to(self):
        """Prefer TALKGROUP form for TO identifier."""
        ic = IdentifierCollection([
            Identifier("Fire Dispatch", IdentifierRole.TO, IdentifierForm.ALIAS),
            Identifier(1001, IdentifierRole.TO, IdentifierForm.TALKGROUP),
        ])

        to_id = ic.get_to_identifier()
        assert to_id.form == IdentifierForm.TALKGROUP

    def test_get_alias_by_role(self):
        """Get alias identifier by role."""
        ic = IdentifierCollection([
            Identifier(12345678, IdentifierRole.FROM, IdentifierForm.RADIO),
            Identifier("Engine 1", IdentifierRole.FROM, IdentifierForm.ALIAS),
            Identifier(1001, IdentifierRole.TO, IdentifierForm.TALKGROUP),
            Identifier("Fire Dispatch", IdentifierRole.TO, IdentifierForm.ALIAS),
        ])

        assert ic.get_alias(IdentifierRole.FROM) == "Engine 1"
        assert ic.get_alias(IdentifierRole.TO) == "Fire Dispatch"


# ============================================================================
# Identifier Validation Tests
# ============================================================================

class TestIdentifierValidation:
    """Test identifier validation."""

    def test_talkgroup_valid_range(self):
        """Talkgroup IDs should be 16-bit (0-65535)."""
        valid_ids = [0, 1, 1000, 32768, 65535]
        for tgid in valid_ids:
            ident = Identifier(tgid, IdentifierRole.TO, IdentifierForm.TALKGROUP)
            assert 0 <= ident.value <= 65535

    def test_radio_id_valid_range(self):
        """Radio IDs should be 24-bit (0-16777215)."""
        valid_ids = [0, 1, 12345678, 16777215]
        for rid in valid_ids:
            ident = Identifier(rid, IdentifierRole.FROM, IdentifierForm.RADIO)
            assert 0 <= ident.value <= 16777215

    def test_system_id_valid_range(self):
        """System IDs should be 12-bit (0-4095)."""
        valid_ids = [0, 1, 0x123, 0xFFF]
        for sid in valid_ids:
            ident = Identifier(sid, IdentifierRole.ANY, IdentifierForm.SYSTEM)
            assert 0 <= ident.value <= 4095


# ============================================================================
# Bulk Alias Loading Tests
# ============================================================================

class TestBulkAliasLoading:
    """Test bulk alias loading from configuration."""

    def test_load_radio_aliases(self):
        """Load multiple radio aliases from dict."""
        manager = TalkerAliasManager()

        radio_aliases = {
            12345678: "Engine 1",
            12345679: "Engine 2",
            12345680: "Truck 1",
            12345681: "Rescue 1",
        }

        manager.load_from_config(radio_aliases=radio_aliases, talkgroup_aliases={})

        assert manager.get_alias(12345678) == "Engine 1"
        assert manager.get_alias(12345679) == "Engine 2"
        assert manager.get_alias(12345680) == "Truck 1"
        assert manager.get_alias(12345681) == "Rescue 1"

    def test_load_talkgroup_aliases(self):
        """Load multiple talkgroup aliases from dict."""
        manager = TalkerAliasManager()

        talkgroup_aliases = {
            1001: "Fire Dispatch",
            1002: "Fire TAC 1",
            1003: "Fire TAC 2",
            2001: "EMS Dispatch",
        }

        manager.load_from_config(radio_aliases={}, talkgroup_aliases=talkgroup_aliases)

        assert manager.get_talkgroup_alias(1001) == "Fire Dispatch"
        assert manager.get_talkgroup_alias(2001) == "EMS Dispatch"

    def test_load_mixed_aliases(self):
        """Load both radio and talkgroup aliases."""
        manager = TalkerAliasManager()

        manager.load_from_config(
            radio_aliases={12345678: "Engine 1"},
            talkgroup_aliases={1001: "Fire Dispatch"},
        )

        assert manager.get_alias(12345678) == "Engine 1"
        assert manager.get_talkgroup_alias(1001) == "Fire Dispatch"

    def test_stats_after_load(self):
        """Stats reflect loaded aliases."""
        manager = TalkerAliasManager()

        manager.load_from_config(
            radio_aliases={1: "A", 2: "B", 3: "C"},
            talkgroup_aliases={100: "X", 200: "Y"},
        )

        stats = manager.get_stats()
        assert stats["radioAliasCount"] == 3
        assert stats["talkgroupAliasCount"] == 2
