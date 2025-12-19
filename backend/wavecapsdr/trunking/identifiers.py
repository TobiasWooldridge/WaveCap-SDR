"""Identifier collection pattern for P25 trunking metadata.

Inspired by SDRTrunk's IdentifierCollection architecture for flexible
metadata management with immutable/mutable variants.

Reference: https://github.com/DSheirer/sdrtrunk
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class IdentifierRole(Enum):
    """Role of an identifier in a call event."""
    FROM = "from"       # Source (transmitting unit)
    TO = "to"           # Destination (talkgroup or target unit)
    ANY = "any"         # No specific role


class IdentifierForm(Enum):
    """Type/form of identifier."""
    RADIO = "radio"           # Radio unit ID
    TALKGROUP = "talkgroup"   # Talkgroup ID
    ALIAS = "alias"           # Human-readable alias
    PATCH = "patch"           # Patched talkgroup
    ENCRYPTION = "encryption" # Encryption key ID
    SITE = "site"             # Site identifier
    SYSTEM = "system"         # System identifier
    NAC = "nac"               # Network Access Code
    LOCATION = "location"     # GPS location


@dataclass(frozen=True)
class Identifier:
    """Immutable identifier with value, role, and form.

    Examples:
        Identifier(12345678, IdentifierRole.FROM, IdentifierForm.RADIO)
        Identifier(1001, IdentifierRole.TO, IdentifierForm.TALKGROUP)
        Identifier("Engine 1", IdentifierRole.FROM, IdentifierForm.ALIAS)
    """
    value: Any
    role: IdentifierRole = IdentifierRole.ANY
    form: IdentifierForm = IdentifierForm.RADIO

    def __hash__(self) -> int:
        return hash((self.value, self.role, self.form))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Identifier):
            return False
        return (self.value == other.value and
                self.role == other.role and
                self.form == other.form)

    def __repr__(self) -> str:
        return f"Identifier({self.value}, {self.role.name}, {self.form.name})"


class IdentifierCollection:
    """Immutable collection of identifiers for a call event.

    Provides type-safe queries for identifiers by role or form.
    Use MutableIdentifierCollection for building collections.

    Example:
        ic = IdentifierCollection([
            Identifier(12345, IdentifierRole.FROM, IdentifierForm.RADIO),
            Identifier(1001, IdentifierRole.TO, IdentifierForm.TALKGROUP),
        ])

        source = ic.get_from_identifier()  # Radio 12345
        tg = ic.get_to_identifier()        # Talkgroup 1001
    """

    def __init__(self, identifiers: Optional[List[Identifier]] = None,
                 timeslot: int = 0) -> None:
        self._identifiers: List[Identifier] = list(identifiers or [])
        self.timeslot = timeslot

    def get_identifiers(self) -> List[Identifier]:
        """Get all identifiers (copy)."""
        return list(self._identifiers)

    def get_identifiers_by_role(self, role: IdentifierRole) -> List[Identifier]:
        """Get identifiers with specific role."""
        return [i for i in self._identifiers if i.role == role]

    def get_identifiers_by_form(self, form: IdentifierForm) -> List[Identifier]:
        """Get identifiers with specific form."""
        return [i for i in self._identifiers if i.form == form]

    def get_from_identifier(self) -> Optional[Identifier]:
        """Get the FROM (source) identifier."""
        from_ids = self.get_identifiers_by_role(IdentifierRole.FROM)
        # Prefer RADIO over ALIAS
        for ident in from_ids:
            if ident.form == IdentifierForm.RADIO:
                return ident
        return from_ids[0] if from_ids else None

    def get_to_identifier(self) -> Optional[Identifier]:
        """Get the TO (destination) identifier."""
        to_ids = self.get_identifiers_by_role(IdentifierRole.TO)
        # Prefer TALKGROUP over RADIO for group calls
        for ident in to_ids:
            if ident.form == IdentifierForm.TALKGROUP:
                return ident
        return to_ids[0] if to_ids else None

    def get_radio_id(self) -> Optional[int]:
        """Get the source radio ID if present."""
        from_id = self.get_from_identifier()
        if from_id and from_id.form == IdentifierForm.RADIO:
            return int(from_id.value)
        return None

    def get_talkgroup_id(self) -> Optional[int]:
        """Get the talkgroup ID if present."""
        to_id = self.get_to_identifier()
        if to_id and to_id.form == IdentifierForm.TALKGROUP:
            return int(to_id.value)
        return None

    def get_alias(self, role: IdentifierRole = IdentifierRole.FROM) -> Optional[str]:
        """Get alias for specified role."""
        for ident in self._identifiers:
            if ident.role == role and ident.form == IdentifierForm.ALIAS:
                return str(ident.value)
        return None

    def has_identifier(self, identifier: Identifier) -> bool:
        """Check if collection contains identifier."""
        return identifier in self._identifiers

    def has_encryption(self) -> bool:
        """Check if collection has encryption identifier."""
        return any(i.form == IdentifierForm.ENCRYPTION for i in self._identifiers)

    def with_timeslot(self, timeslot: int) -> IdentifierCollection:
        """Create new collection with different timeslot."""
        return IdentifierCollection(self._identifiers, timeslot)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timeslot": self.timeslot,
            "from": self._identifier_to_dict(self.get_from_identifier()),
            "to": self._identifier_to_dict(self.get_to_identifier()),
            "fromAlias": self.get_alias(IdentifierRole.FROM),
            "toAlias": self.get_alias(IdentifierRole.TO),
            "hasEncryption": self.has_encryption(),
            "identifierCount": len(self._identifiers),
        }

    def _identifier_to_dict(self, ident: Optional[Identifier]) -> Optional[Dict[str, Any]]:
        if ident is None:
            return None
        return {
            "value": ident.value,
            "role": ident.role.value,
            "form": ident.form.value,
        }

    def __len__(self) -> int:
        return len(self._identifiers)

    def __repr__(self) -> str:
        return f"IdentifierCollection({len(self._identifiers)} ids, ts={self.timeslot})"


class MutableIdentifierCollection(IdentifierCollection):
    """Mutable variant for building identifier collections.

    Example:
        mic = MutableIdentifierCollection()
        mic.update(Identifier(12345, IdentifierRole.FROM, IdentifierForm.RADIO))
        mic.update(Identifier(1001, IdentifierRole.TO, IdentifierForm.TALKGROUP))

        # Convert to immutable
        ic = mic.to_immutable()
    """

    def update(self, identifier: Identifier) -> None:
        """Add or update identifier.

        If an identifier with same role and form exists, it's replaced.
        """
        # Remove existing with same role+form
        self._identifiers = [
            i for i in self._identifiers
            if not (i.role == identifier.role and i.form == identifier.form)
        ]
        self._identifiers.append(identifier)

    def remove(self, identifier: Identifier) -> bool:
        """Remove identifier if present."""
        try:
            self._identifiers.remove(identifier)
            return True
        except ValueError:
            return False

    def clear(self) -> None:
        """Remove all identifiers."""
        self._identifiers.clear()

    def to_immutable(self) -> IdentifierCollection:
        """Convert to immutable IdentifierCollection."""
        return IdentifierCollection(list(self._identifiers), self.timeslot)


class TalkerAliasManager:
    """Cache and enrich identifier collections with observed aliases.

    Aliases are learned from:
    - Talker alias headers in P25 calls
    - Radio ID to name mapping from configuration
    - LRRP registration data

    Inspired by SDRTrunk's TalkerAliasManager pattern.

    Example:
        manager = TalkerAliasManager()
        manager.update_alias(12345, "Engine 1")

        # Later, enrich a collection
        enriched = manager.enrich(original_collection)
    """

    def __init__(self) -> None:
        # radio_id -> alias_name
        self._radio_aliases: Dict[int, str] = {}
        # talkgroup_id -> alias_name
        self._talkgroup_aliases: Dict[int, str] = {}
        # Last update time for staleness
        self._last_update: float = 0.0

    def update_alias(self, radio_id: int, alias: str) -> None:
        """Update alias for a radio ID."""
        self._radio_aliases[radio_id] = alias
        self._last_update = time.time()

    def update_talkgroup_alias(self, tgid: int, alias: str) -> None:
        """Update alias for a talkgroup."""
        self._talkgroup_aliases[tgid] = alias
        self._last_update = time.time()

    def get_alias(self, radio_id: int) -> Optional[str]:
        """Get alias for radio ID."""
        return self._radio_aliases.get(radio_id)

    def get_talkgroup_alias(self, tgid: int) -> Optional[str]:
        """Get alias for talkgroup."""
        return self._talkgroup_aliases.get(tgid)

    def load_from_config(self, radio_aliases: Dict[int, str],
                         talkgroup_aliases: Dict[int, str]) -> None:
        """Bulk load aliases from configuration."""
        self._radio_aliases.update(radio_aliases)
        self._talkgroup_aliases.update(talkgroup_aliases)
        self._last_update = time.time()

    def enrich(self, ic: IdentifierCollection) -> IdentifierCollection:
        """Enrich collection with cached aliases.

        Returns new collection if aliases were added, original otherwise.
        """
        from_id = ic.get_from_identifier()
        to_id = ic.get_to_identifier()

        from_alias = None
        to_alias = None

        # Look up FROM alias
        if from_id and from_id.form == IdentifierForm.RADIO:
            from_alias = self._radio_aliases.get(int(from_id.value))

        # Look up TO alias
        if to_id:
            if to_id.form == IdentifierForm.TALKGROUP:
                to_alias = self._talkgroup_aliases.get(int(to_id.value))
            elif to_id.form == IdentifierForm.RADIO:
                to_alias = self._radio_aliases.get(int(to_id.value))

        # Return original if no aliases found
        if from_alias is None and to_alias is None:
            return ic

        # Build enriched collection
        mic = MutableIdentifierCollection(ic.get_identifiers(), ic.timeslot)

        if from_alias:
            mic.update(Identifier(from_alias, IdentifierRole.FROM, IdentifierForm.ALIAS))
        if to_alias:
            mic.update(Identifier(to_alias, IdentifierRole.TO, IdentifierForm.ALIAS))

        return mic.to_immutable()

    def get_stats(self) -> Dict[str, Any]:
        """Get alias cache statistics."""
        return {
            "radioAliasCount": len(self._radio_aliases),
            "talkgroupAliasCount": len(self._talkgroup_aliases),
            "lastUpdate": self._last_update,
        }
