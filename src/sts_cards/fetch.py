"""Fetch cards from spire-archive.com per game.

Now with:
  - HTTP-level caching via `requests-cache` (1h TTL by default, bypassable)
  - Retries with exponential backoff via `tenacity`
  - Derived feature columns (damage, block, mechanics, STS2 Orbs/Forge/Souls)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

from . import GAMES
from .cache import make_session
from .features import extract_features
from .normalize import normalize_card
from .provenance import DatasetProvenance, FetchProvenance, now_iso

log = logging.getLogger(__name__)

API_BASE = "https://spire-archive.com/api"
PAGE_SIZE = 100
REQUEST_TIMEOUT = 30
SLEEP_BETWEEN_PAGES = 0.2  # be polite to the API


@retry(
    retry=retry_if_exception_type((requests.RequestException,)),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
def _get(session: requests.Session, url: str, params: dict[str, Any]) -> Any:
    r = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


def _extract_items(payload: Any) -> list[dict[str, Any]]:
    """spire-archive may return a bare list or a wrapped dict — handle both."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("items", "results", "cards", "data"):
            if key in payload and isinstance(payload[key], list):
                return payload[key]
    raise ValueError(f"Unexpected response shape: {type(payload).__name__}")


def fetch_all_cards(
    game: str,
    *,
    lang: str = "en",
    use_cache: bool = True,
    cache_ttl_seconds: int = 3600,
) -> list[dict[str, Any]]:
    """Paginate through /api/{game}/cards until exhausted."""
    if game not in GAMES:
        raise ValueError(f"game must be one of {GAMES}, got {game!r}")

    session = make_session(use_cache=use_cache, ttl_seconds=cache_ttl_seconds)
    cards: list[dict[str, Any]] = []
    offset = 0
    pbar = tqdm(desc=game, unit="card")
    while True:
        page = _extract_items(_get(
            session,
            f"{API_BASE}/{game}/cards",
            {"lang": lang, "offset": offset, "limit": PAGE_SIZE},
        ))
        if not page:
            break
        cards.extend(page)
        pbar.update(len(page))
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
        # Only sleep when we actually hit the network (cache hits are free)
        if not getattr(session, "cache", None) or \
                getattr(getattr(session, "_last_request_response", None),
                        "from_cache", False) is False:
            time.sleep(SLEEP_BETWEEN_PAGES)
    pbar.close()
    return cards


def fetch_game(
    game: str,
    out_dir: Path,
    *,
    lang: str = "en",
    sts_game_version: str | None = None,
    use_cache: bool = True,
) -> Path:
    """Fetch one game's cards, normalize + extract features, write Parquet
    + provenance. Returns the path to the written Parquet.
    """
    log.info("Fetching %s cards (lang=%s, cache=%s)", game, lang, use_cache)
    raw = fetch_all_cards(game, lang=lang, use_cache=use_cache)

    # Normalize core schema, then merge in derived features
    rows = []
    for c in raw:
        row = normalize_card(c, game=game)
        feats = extract_features(c, game=game)
        # mechanics, status_effects_applied, orbs_* are list-typed — store
        # as JSON strings so the column dtype is consistent across all rows
        # (Parquet has no problem with native lists, but pandas occasionally
        # infers `object` dtype which leaks into downstream tools)
        row["damage"] = feats["damage"]
        row["damage_upgraded"] = feats["damage_upgraded"]
        row["block"] = feats["block"]
        row["block_upgraded"] = feats["block_upgraded"]
        row["targets_all_enemies"] = feats["targets_all_enemies"]
        row["status_effects_applied"] = json.dumps(
            feats["status_effects_applied"], ensure_ascii=False
        )
        row["mechanics"] = json.dumps(feats["mechanics"], ensure_ascii=False)
        row["orbs_channeled"] = json.dumps(
            feats["orbs_channeled"], ensure_ascii=False
        )
        row["orbs_referenced"] = json.dumps(
            feats["orbs_referenced"], ensure_ascii=False
        )
        row["forge_value"] = feats["forge_value"]
        row["souls_added"] = feats["souls_added"]
        rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{game}_cards.parquet"
    pd.DataFrame(rows).to_parquet(out_path, index=False)

    prov = DatasetProvenance(
        fetch=FetchProvenance(
            source=f"{API_BASE}/{game}/cards",
            source_fetched_at=now_iso(),
            game=game,
            language=lang,
            n_cards=len(rows),
            sts_game_version=sts_game_version,
        ),
        embed=None,
    )
    prov.write(out_dir / f"{game}_provenance.json")

    log.info("Wrote %d cards → %s", len(rows), out_path)
    return out_path
