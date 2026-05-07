"""Normalize raw spire-archive card payloads into a stable schema.

These functions are pure (no I/O, no model loading) so they're trivially
unit-tested. The schema is what gets shipped as the HuggingFace dataset.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

import numpy as np

# Fields used for embedding (mechanics-relevant; no flavor text).
MECHANICS_FIELDS: tuple[str, ...] = (
    "name", "type", "rarity", "color", "cost",
    "description", "description_upgraded", "keywords",
)


def _first(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first present, non-None value among `keys`."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        return [value]
    return [str(value)]


def normalize_card(card: dict[str, Any], game: str) -> dict[str, Any]:
    """Map a heterogeneous API record into the stable column schema.

    The original payload is preserved as `raw_json` so downstream code
    can recover any field this function doesn't expose explicitly.
    """
    return {
        "game": game,
        "id": _first(card, "id", "card_id", "cardId", "key"),
        "name": _first(card, "name", "title"),
        "type": _first(card, "type", "card_type"),
        "rarity": _first(card, "rarity"),
        "color": _first(card, "color", "character", "class"),
        "cost": str(_first(card, "cost", "energy", default="")),
        "description": _first(card, "description", "text", "body", default=""),
        "description_upgraded": _first(
            card,
            "description_upgraded", "upgraded_description",
            "upgraded_text", "text_upgraded",
            default="",
        ),
        "keywords": json.dumps(_as_str_list(card.get("keywords")), ensure_ascii=False),
        "raw_json": json.dumps(card, ensure_ascii=False, sort_keys=True),
    }


def normalize_card_name_in_text(name: str | None, text: str | None) -> str:
    """Replace the card's own name in its description with `~`.

    Following the minimaxir/mtg-embeddings convention: card text shouldn't
    leak the card's name to the embedding model, otherwise vectors are
    dominated by name surface form rather than mechanics.
    """
    if not text:
        return ""
    if not name:
        return text
    return re.compile(re.escape(name), re.IGNORECASE).sub("~", text)


def build_card_document(row: dict[str, Any] | Any) -> str:
    """Produce the prettified-JSON string that gets embedded.

    Indentation is intentional — measurably improves embedding quality
    per minimaxir's writeup. Accepts both dicts and pandas Series.
    """
    def get(key: str, default: Any = None) -> Any:
        if hasattr(row, "get"):
            return row.get(key, default)
        return getattr(row, key, default)

    name = get("name") or ""
    doc: dict[str, Any] = {}
    for field in MECHANICS_FIELDS:
        val = get(field)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        # keywords may be stored as a JSON-string list in the DataFrame
        if field == "keywords" and isinstance(val, str):
            try:
                val = json.loads(val) if val else []
            except json.JSONDecodeError:
                val = [val]
        if field in ("description", "description_upgraded") and isinstance(val, str):
            val = normalize_card_name_in_text(name, val)
        if val == "" or val == []:
            continue
        doc[field] = val
    return json.dumps(doc, indent=2, ensure_ascii=False)


def build_documents(rows: Iterable[Any]) -> list[str]:
    return [build_card_document(r) for r in rows]
