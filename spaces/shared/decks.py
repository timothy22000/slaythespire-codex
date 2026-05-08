"""Deck construction for the Build Me a Deck Space.

Pure functions (no I/O, no model loading) so they're trivially testable.
Caller passes in the merged DataFrame + embedding matrix from `load_game()`
plus the encoded prompt vector from `encode_query()`.

Design lives at /Users/tmun/.claude/plans/do-i-need-to-prancy-stonebraker.md
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd


# Per-game playable character classes. Hardcoded because the parquet has
# pseudo-classes (event, quest, status, token) mixed into the `color` column
# and there's no schema field that distinguishes "drafted" from "inflicted."
DRAFTED_CLASSES: dict[str, list[str]] = {
    "sts1": ["ironclad", "silent", "defect", "watcher"],
    "sts2": ["ironclad", "silent", "defect", "necrobinder", "regent"],
}

# Starter decks per (game, character). Each entry is a list of (card_name, count)
# tuples. STS2 entries for necrobinder/regent are best-effort (the parquet stores
# card definitions, not deck lists; canonical multiplicity may shift across
# Early Access patches). Card-name lookups are case-insensitive and color-matched
# at draft time so character-specific Strike/Defend variants land correctly.
STARTERS: dict[tuple[str, str], list[tuple[str, int]]] = {
    ("sts1", "ironclad"):    [("Strike", 5), ("Defend", 4), ("Bash", 1)],
    ("sts1", "silent"):      [("Strike", 5), ("Defend", 5), ("Survivor", 1), ("Neutralize", 1)],
    ("sts1", "defect"):      [("Strike", 4), ("Defend", 4), ("Zap", 1), ("Dualcast", 1)],
    ("sts1", "watcher"):     [("Strike", 4), ("Defend", 4), ("Eruption", 1), ("Vigilance", 1)],
    ("sts2", "ironclad"):    [("Strike", 5), ("Defend", 4), ("Bash", 1)],
    ("sts2", "silent"):      [("Strike", 5), ("Defend", 5), ("Survivor", 1), ("Neutralize", 1)],
    ("sts2", "defect"):      [("Strike", 4), ("Defend", 4), ("Zap", 1), ("Dualcast", 1)],
    ("sts2", "necrobinder"): [("Strike", 4), ("Defend", 4), ("Bodyguard", 1), ("Unleash", 1)],
    ("sts2", "regent"):      [("Strike", 4), ("Defend", 4), ("Falling Star", 1), ("Venerate", 1)],
}

# Constraint targets used when `enforce_*` toggles are on. Calibrated against
# typical mid-act-2 deck composition and the actual cost distribution of the
# drafted pools (so the constraints don't fight the data).
TYPE_RATIO = {"Attack": 0.50, "Skill": 0.35, "Power": 0.15}
CURVE_RATIO = {"low": 0.30, "mid": 0.50, "high": 0.20}

# Maximum copies of any single named card. Real STS allows arbitrary copies
# during a run, but for a "buildable deck" presentation, 4 is the convention
# from CCGs and reads as obviously sensible.
DUP_CAP = 4


IncludeStartersOpt = Literal["None", "Strike + Defend only", "Full starter deck"]


@dataclass
class DeckPick:
    pool_idx: int
    name: str
    type_: str
    rarity: str
    color: str
    cost: str
    description: str
    similarity: float
    locked: bool


@dataclass
class DeckResult:
    picks: list[DeckPick]
    avg_sim_all: float
    avg_sim_picks: float            # similarity averaged over non-locked picks only
    type_count: dict[str, int]
    curve_count: dict[str, int]     # keys: "low", "mid", "high"
    avg_cost: float                 # weighted by number of copies; X-cost (-1) treated as 2
    top_keywords: list[tuple[str, int]]
    notes: list[str] = field(default_factory=list)


def filter_to_drafted_pool(df: pd.DataFrame, character: str) -> np.ndarray:
    """Return a boolean mask selecting cards a `character` can draft.

    Excludes statuses, curses, quests (filter on `type`, not `color`).
    Includes character-specific cards plus colorless.
    """
    mask = (
        df["color"].isin([character, "colorless"])
        & df["type"].isin(["Attack", "Skill", "Power"])
        & df["rarity"].isin(["Basic", "Common", "Uncommon", "Rare"])
    )
    return mask.to_numpy()


def _curve_bucket(cost_str) -> str:
    """Map a parquet cost string to a curve bucket. X-cost (-1) is mid."""
    s = str(cost_str)
    if s in ("0", "1"):
        return "low"
    if s in ("2", "-1"):
        return "mid"
    if s in ("3", "4", "5", "6", "7", "8", "9"):
        return "high"
    return "mid"  # fallback for "", shouldn't appear after pool filter


def _cost_for_avg(cost_str) -> float:
    """Numeric cost used in the avg-cost summary metric. X-cost counts as 2."""
    s = str(cost_str)
    if s == "-1":
        return 2.0  # X
    if s == "-2":
        return 0.0  # unplayable, shouldn't appear after pool filter
    try:
        return float(s)
    except ValueError:
        return 0.0


def _starter_row_idx(pool_df: pd.DataFrame, name: str, character: str) -> int | None:
    """Find the pool row matching a starter card name with the right color.

    Strike/Defend exist as one row per character class, pick the one whose
    color matches. Other starters are character-unique so the color match
    is redundant but still safe.
    """
    case = pool_df["name"].str.lower() == name.lower()
    char_match = pool_df[case & (pool_df["color"] == character)]
    if len(char_match):
        return int(char_match.index[0])
    # Fall back to any name match (e.g. colorless variants)
    any_match = pool_df[case]
    if len(any_match):
        return int(any_match.index[0])
    return None


def _extract_keywords(rows: pd.DataFrame, n: int = 3) -> list[tuple[str, int]]:
    """Take the most-common keywords across the picked rows."""
    counter: Counter[str] = Counter()
    for kw_str in rows["keywords"].fillna(""):
        if not isinstance(kw_str, str) or not kw_str.strip():
            continue
        try:
            parsed = json.loads(kw_str)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            for k in parsed:
                if isinstance(k, str) and k.strip():
                    counter[k] += 1
    return counter.most_common(n)


def build_deck(
    df: pd.DataFrame,
    emb: np.ndarray,
    *,
    game: str,
    character: str,
    query_vec: np.ndarray,
    deck_size: int = 20,
    include_starters: IncludeStartersOpt = "Full starter deck",
    allow_duplicates: bool = True,
    enforce_curve: bool = True,
    enforce_type_balance: bool = True,
) -> DeckResult:
    """Build a deck of `deck_size` cards by greedy similarity with constraints.

    See plan file for algorithm details. The caller must pass `query_vec`
    already encoded with the same model + instruction the indexed cards
    used; otherwise similarity scores are nonsense.
    """
    notes: list[str] = []

    # 1) Filter pool to drafted cards.
    mask = filter_to_drafted_pool(df, character)
    pool_df = df[mask].reset_index(drop=True)
    pool_emb = emb[mask]
    sims = pool_emb @ query_vec  # cosine similarity, both unit-norm

    if len(pool_df) == 0:
        return DeckResult(
            picks=[], avg_sim_all=0.0, avg_sim_picks=0.0,
            type_count={}, curve_count={}, avg_cost=0.0, top_keywords=[],
            notes=[f"No draftable cards found for character {character!r} in {game.upper()}."],
        )

    # Clamp deck size to pool size if needed.
    if deck_size > len(pool_df):
        notes.append(
            f"Pool only has {len(pool_df)} draftable cards; trimmed deck size from "
            f"{deck_size} to {len(pool_df)}."
        )
        deck_size = len(pool_df)

    # 2) Lock starters.
    locked_indices: list[int] = []
    starter_block = STARTERS.get((game, character), [])
    if include_starters != "None" and starter_block:
        wanted = (
            [(n, c) for (n, c) in starter_block if n in {"Strike", "Defend"}]
            if include_starters == "Strike + Defend only"
            else list(starter_block)
        )
        for (name, count) in wanted:
            idx = _starter_row_idx(pool_df, name, character)
            if idx is None:
                notes.append(f"Starter card {name!r} not in pool, slot replaced by prompt match.")
                continue
            locked_indices.extend([idx] * count)

    if len(locked_indices) > deck_size:
        return DeckResult(
            picks=[], avg_sim_all=0.0, avg_sim_picks=0.0,
            type_count={}, curve_count={}, avg_cost=0.0, top_keywords=[],
            notes=[
                f"Deck size {deck_size} is smaller than {character.title()}'s "
                f"{len(locked_indices)}-card starter. Increase deck size or set "
                f"Starters to 'Strike + Defend only'."
            ],
        )

    # 3) Compute targets for the non-locked slots.
    remaining = deck_size - len(locked_indices)
    type_target: dict[str, int] | None = None
    curve_target: dict[str, int] | None = None

    if enforce_type_balance and remaining > 0:
        a = round(TYPE_RATIO["Attack"] * remaining)
        s = round(TYPE_RATIO["Skill"] * remaining)
        p = remaining - a - s
        type_target = {"Attack": a, "Skill": s, "Power": max(0, p)}

    if enforce_curve and remaining > 0:
        lo = round(CURVE_RATIO["low"] * remaining)
        mid = round(CURVE_RATIO["mid"] * remaining)
        hi = remaining - lo - mid
        curve_target = {"low": lo, "mid": mid, "high": max(0, hi)}

    # 4) Greedy fill with feasibility check.
    candidates = sorted(range(len(pool_df)), key=lambda i: -sims[i])
    chosen_count_by_name: Counter[str] = Counter()
    type_count: Counter[str] = Counter()
    curve_count: Counter[str] = Counter()
    chosen_indices: list[int] = list(locked_indices)

    for idx in locked_indices:
        chosen_count_by_name[pool_df.iloc[idx]["name"]] += 1
        type_count[pool_df.iloc[idx]["type"]] += 1
        curve_count[_curve_bucket(pool_df.iloc[idx]["cost"])] += 1

    locked_names = {pool_df.iloc[i]["name"] for i in locked_indices}

    for ci in candidates:
        if remaining <= 0:
            break
        cname = pool_df.iloc[ci]["name"]
        ctype = pool_df.iloc[ci]["type"]
        cbucket = _curve_bucket(pool_df.iloc[ci]["cost"])

        # Duplicate gating
        if not allow_duplicates and cname in {pool_df.iloc[i]["name"] for i in chosen_indices} and cname not in locked_names:
            continue
        if chosen_count_by_name[cname] >= DUP_CAP:
            continue

        # Type-target feasibility
        if type_target is not None and type_count[ctype] >= type_target.get(ctype, 999):
            deficit = sum(max(0, type_target[t] - type_count[t]) for t in type_target)
            if deficit >= remaining:
                continue

        # Curve-target feasibility
        if curve_target is not None and curve_count[cbucket] >= curve_target.get(cbucket, 999):
            deficit = sum(max(0, curve_target[b] - curve_count[b]) for b in curve_target)
            if deficit >= remaining:
                continue

        chosen_indices.append(ci)
        chosen_count_by_name[cname] += 1
        type_count[ctype] += 1
        curve_count[cbucket] += 1
        remaining -= 1

    # 5) Safety net, fill any leftover slots from top-similarity ignoring constraints.
    if remaining > 0:
        for ci in candidates:
            if remaining <= 0:
                break
            cname = pool_df.iloc[ci]["name"]
            if chosen_count_by_name[cname] >= DUP_CAP:
                continue
            if not allow_duplicates and cname in {pool_df.iloc[i]["name"] for i in chosen_indices} and cname not in locked_names:
                continue
            if ci in chosen_indices and not allow_duplicates:
                continue
            chosen_indices.append(ci)
            chosen_count_by_name[cname] += 1
            type_count[pool_df.iloc[ci]["type"]] += 1
            curve_count[_curve_bucket(pool_df.iloc[ci]["cost"])] += 1
            remaining -= 1

    # 6) Render to DeckPick records.
    locked_set = Counter(locked_indices)  # multi-set: locked count per index
    locked_consumed: Counter[int] = Counter()
    picks: list[DeckPick] = []
    for ci in chosen_indices:
        is_locked = locked_consumed[ci] < locked_set.get(ci, 0)
        if is_locked:
            locked_consumed[ci] += 1
        row = pool_df.iloc[ci]
        d = row.get("description", "")
        desc_clean = "" if pd.isna(d) else str(d).strip()
        picks.append(DeckPick(
            pool_idx=ci,
            name=str(row["name"]),
            type_=str(row["type"]),
            rarity=str(row["rarity"]),
            color=str(row["color"]),
            cost=str(row["cost"]),
            description=desc_clean,
            similarity=float(sims[ci]),
            locked=is_locked,
        ))

    # 7) Summary metrics.
    chosen_rows = pool_df.iloc[chosen_indices].reset_index(drop=True)
    non_locked_sims = [p.similarity for p in picks if not p.locked]
    avg_sim_picks = float(sum(non_locked_sims) / len(non_locked_sims)) if non_locked_sims else 0.0
    avg_sim_all = float(sum(p.similarity for p in picks) / len(picks)) if picks else 0.0
    avg_cost = (
        float(sum(_cost_for_avg(p.cost) for p in picks) / len(picks)) if picks else 0.0
    )
    top_keywords = _extract_keywords(chosen_rows)

    return DeckResult(
        picks=picks,
        avg_sim_all=avg_sim_all,
        avg_sim_picks=avg_sim_picks,
        type_count=dict(type_count),
        curve_count=dict(curve_count),
        avg_cost=avg_cost,
        top_keywords=top_keywords,
        notes=notes,
    )
