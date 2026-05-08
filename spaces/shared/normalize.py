"""Card-document construction for embedding.

Vendored from src/sts_cards/normalize.py, keep in sync. Only the helpers
needed by Space 2 (Synergy Inspector) and Space 3 (Build Me a Deck) are
included; the full normalize_card() function lives in the parent package.
"""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np

# Fields the embedding model sees. Must match the parent package exactly,
# otherwise user-supplied cards get encoded with a different field set than
# the indexed cards and similarity scores stop being meaningful.
MECHANICS_FIELDS: tuple[str, ...] = (
    "name", "type", "rarity", "color", "cost",
    "description", "description_upgraded", "keywords",
)


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

    Indentation is intentional, measurably improves embedding quality
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
