"""Feature extraction from card descriptions.

Adds derived columns to the normalized dataset that downstream code can
filter on without re-parsing free text. Two tiers:

  - GENERIC features apply to both games (damage, block, AOE flag, status
    effects, mechanic keywords).
  - STS2-SPECIFIC features cover mechanics introduced in Slay the Spire 2
    that don't exist in STS1 (Orbs, Forge, Souls, Enchantments, Channel).

All extractors are best-effort regex over the description text. They fail
soft: missing or unparseable values become None / [] / False rather than
raising. This is fine because we always preserve the full original
payload in `raw_json`.
"""

from __future__ import annotations

import re
from typing import Any

# Status effects that can be applied — present in both games.
STATUS_EFFECTS: tuple[str, ...] = (
    "Vulnerable", "Weak", "Frail", "Poison",
    "Strength", "Dexterity",
)

# Mechanic keywords. The list is intentionally a superset of both games;
# keywords that don't match a card simply don't show up in `mechanics`.
MECHANIC_KEYWORDS: tuple[str, ...] = (
    "Exhaust", "Innate", "Ethereal", "Retain", "Eternal", "Unplayable",
    # STS2 additions
    "Channel", "Forge", "Soul", "Enchant", "Summon", "Procure",
)

# STS2 Orb types. Lightning and Frost exist for the Defect in STS1; the
# rest are STS2-specific. We extract any of them when seen.
ORB_TYPES: tuple[str, ...] = (
    "Lightning", "Frost", "Plasma", "Glass", "Dark", "Fuel",
)


# --- Generic extractors (both games) ---

_DAMAGE_RE = re.compile(r"\bDeal\s+(\d+)\s+damage", re.IGNORECASE)
_BLOCK_RE = re.compile(r"\bGain\s+(\d+)\s+Block", re.IGNORECASE)
_AOE_RE = re.compile(r"\bto\s+ALL\s+enemies\b")


def extract_damage(text: str | None) -> int | None:
    """Primary damage value if 'Deal N damage' is present.

    For multi-hit cards like 'Deal 5 damage twice', this returns 5
    (the per-hit value), not the total.
    """
    if not text:
        return None
    m = _DAMAGE_RE.search(text)
    return int(m.group(1)) if m else None


def extract_block(text: str | None) -> int | None:
    if not text:
        return None
    m = _BLOCK_RE.search(text)
    return int(m.group(1)) if m else None


def targets_all_enemies(text: str | None) -> bool:
    return bool(text) and bool(_AOE_RE.search(text))


def extract_status_effects_applied(text: str | None) -> list[dict[str, Any]]:
    """Find '(Apply|Gain) N <Status>' patterns. Returns a list of
    {effect, count} dicts to handle cards that apply multiple kinds."""
    if not text:
        return []
    results: list[dict[str, Any]] = []
    pattern = re.compile(
        r"\b(?:Apply|Gain)\s+(\d+)\s+(" + "|".join(STATUS_EFFECTS) + r")\b"
    )
    for m in pattern.finditer(text):
        results.append({"effect": m.group(2), "count": int(m.group(1))})
    return results


def extract_mechanics(text: str | None, declared_keywords: list[str] | None = None) -> list[str]:
    """Combine declared keywords from the API payload with any mechanics
    the description mentions. Deduplicates while preserving order."""
    found: list[str] = []
    seen: set[str] = set()

    for kw in (declared_keywords or []):
        if kw and kw not in seen:
            found.append(kw)
            seen.add(kw)

    if text:
        for kw in MECHANIC_KEYWORDS:
            if re.search(rf"\b{re.escape(kw)}\b", text) and kw not in seen:
                found.append(kw)
                seen.add(kw)

    return found


# --- STS2-specific extractors ---

_CHANNEL_RE = re.compile(
    r"\bChannel\s+(\d+)\s+(" + "|".join(ORB_TYPES) + r")\b"
)


def extract_orbs_channeled(text: str | None) -> list[dict[str, Any]]:
    """Find 'Channel N <Orb>' patterns. STS2 cards can channel multiple
    different orbs in one effect, so we return a list of {type, count}."""
    if not text:
        return []
    return [
        {"type": m.group(2), "count": int(m.group(1))}
        for m in _CHANNEL_RE.finditer(text)
    ]


def extract_orbs_referenced(text: str | None) -> list[str]:
    """Any orb types mentioned in the text, channeled or not.

    Useful for finding cards that *interact with* an orb type without
    necessarily channeling it (e.g. 'if you have Frost').
    """
    if not text:
        return []
    seen = []
    for orb in ORB_TYPES:
        if re.search(rf"\b{orb}\b", text) and orb not in seen:
            seen.append(orb)
    return seen


_FORGE_RE = re.compile(r"\bForge\s+(\d+)\b")
_SOULS_RE = re.compile(r"\bAdd\s+(\d+)\s+Souls?\b")


def extract_forge_value(text: str | None) -> int | None:
    if not text:
        return None
    m = _FORGE_RE.search(text)
    return int(m.group(1)) if m else None


def extract_souls_added(text: str | None) -> int | None:
    if not text:
        return None
    m = _SOULS_RE.search(text)
    return int(m.group(1)) if m else None


# --- Top-level entry points ---


def extract_features(card: dict[str, Any], game: str) -> dict[str, Any]:
    """Compute all derived features for one card.

    Returns a flat dict that can be merged into the normalized card row.
    STS2-only fields are always present in the schema (so the columns are
    consistent across games) but will be None / [] for STS1 cards.
    """
    desc = card.get("description") or ""
    desc_up = card.get("description_upgraded") or ""
    declared = card.get("keywords") or []
    if isinstance(declared, str):
        declared = [declared]

    features: dict[str, Any] = {
        # Generic
        "damage": extract_damage(desc),
        "damage_upgraded": extract_damage(desc_up),
        "block": extract_block(desc),
        "block_upgraded": extract_block(desc_up),
        "targets_all_enemies": targets_all_enemies(desc),
        "status_effects_applied": extract_status_effects_applied(desc),
        "mechanics": extract_mechanics(desc, declared),
        # STS2-specific (will be empty/None for STS1 cards in practice)
        "orbs_channeled": [],
        "orbs_referenced": [],
        "forge_value": None,
        "souls_added": None,
    }

    if game == "sts2":
        features["orbs_channeled"] = extract_orbs_channeled(desc)
        features["orbs_referenced"] = extract_orbs_referenced(desc)
        features["forge_value"] = extract_forge_value(desc)
        features["souls_added"] = extract_souls_added(desc)

    return features
