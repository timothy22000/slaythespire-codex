"""Per-(type, cost) damage and block percentile baselines.

Used by the Synergy Inspector to flag user-submitted cards whose stats
sit outside the empirical distribution of the chosen game's cards.

Buckets with fewer than MIN_BUCKET_N cards fall back to a type-only
bucket (key (type, '*')) since per-(type, cost) percentiles aren't
meaningful at n=1 or n=2.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from .data import load_game

MIN_BUCKET_N = 5


def _stats_for(values: pd.Series) -> dict:
    arr = pd.to_numeric(values, errors="coerce").dropna()
    return {
        "n": int(len(arr)),
        "p50": float(arr.quantile(0.5)) if len(arr) else None,
        "p90": float(arr.quantile(0.9)) if len(arr) else None,
        "p95": float(arr.quantile(0.95)) if len(arr) else None,
    }


@lru_cache(maxsize=2)
def baselines(game: str) -> dict[tuple[str, str], dict]:
    """Return {(type, cost): {damage: stats, block: stats}}.

    Buckets with n < MIN_BUCKET_N omit per-bucket stats; type-only
    fallback under key (type, '*') is always present.
    """
    df, _ = load_game(game)
    out: dict[tuple[str, str], dict] = {}

    for t in df["type"].dropna().unique():
        t_df = df[df["type"] == t]
        # Type-only fallback bucket
        out[(t, "*")] = {
            "damage": _stats_for(t_df["damage"]),
            "block": _stats_for(t_df["block"]),
        }
        # Per-(type, cost) buckets
        for c in t_df["cost"].dropna().unique():
            tc_df = t_df[t_df["cost"] == c]
            d_stats = _stats_for(tc_df["damage"])
            b_stats = _stats_for(tc_df["block"])
            if d_stats["n"] >= MIN_BUCKET_N or b_stats["n"] >= MIN_BUCKET_N:
                out[(t, c)] = {"damage": d_stats, "block": b_stats}

    return out


# Costs where damage/block outlier checks don't apply.
SKIP_COST_OUTLIER = {"-1", "-2"}  # X-cost, Unplayable


def check_outliers(
    game: str,
    type_: str,
    cost: str,
    damage: float | None,
    block: float | None,
) -> list[dict]:
    """Return list of {severity, kind, message} warnings. Empty = clean."""
    warnings: list[dict] = []

    if cost in SKIP_COST_OUTLIER:
        if damage is not None or block is not None:
            label = "X-cost" if cost == "-1" else "unplayable"
            warnings.append({
                "severity": "info",
                "kind": "skip",
                "message": (
                    f"{label} cards skip the damage/block outlier check "
                    f"(stats here don't follow flat-cost distributions)."
                ),
            })
        return warnings

    if type_ not in {"Attack", "Skill"}:
        # Powers, Curses, Statuses, Quests don't have meaningful damage/block
        # baselines in the indexed data.
        return warnings

    bl = baselines(game)
    bucket = bl.get((type_, cost)) or bl.get((type_, "*"))
    if bucket is None:
        return warnings

    used_fallback = (type_, cost) not in bl
    bucket_label = (
        f"all {type_.lower()}s" if used_fallback
        else f"cost-{cost} {type_.lower()}s"
    )

    if damage is not None and damage > 0 and type_ == "Attack":
        s = bucket["damage"]
        if s["n"] >= MIN_BUCKET_N:
            if damage > s["p95"]:
                warnings.append({
                    "severity": "strong",
                    "kind": "damage",
                    "message": (
                        f"{bucket_label.capitalize()} rarely deal more than "
                        f"{s['p95']:.0f} damage (95th percentile across "
                        f"{s['n']} cards in {game.upper()}); yours deals {damage:.0f}."
                    ),
                })
            elif damage > s["p90"]:
                warnings.append({
                    "severity": "soft",
                    "kind": "damage",
                    "message": (
                        f"On the high end: only the top 10% of {bucket_label} "
                        f"deal {s['p90']:.0f}+ damage (n={s['n']})."
                    ),
                })
            elif damage < s["p50"] * 0.5:
                warnings.append({
                    "severity": "soft",
                    "kind": "damage",
                    "message": (
                        f"{bucket_label.capitalize()} median around "
                        f"{s['p50']:.0f} damage (n={s['n']}); this looks under-tuned."
                    ),
                })

    if block is not None and block > 0 and type_ == "Skill":
        s = bucket["block"]
        if s["n"] >= MIN_BUCKET_N:
            if block > s["p95"]:
                warnings.append({
                    "severity": "strong",
                    "kind": "block",
                    "message": (
                        f"{bucket_label.capitalize()} rarely give more than "
                        f"{s['p95']:.0f} block (95th percentile across "
                        f"{s['n']} cards in {game.upper()}); yours gives {block:.0f}."
                    ),
                })
            elif block > s["p90"]:
                warnings.append({
                    "severity": "soft",
                    "kind": "block",
                    "message": (
                        f"On the high end: only the top 10% of {bucket_label} "
                        f"give {s['p90']:.0f}+ block (n={s['n']})."
                    ),
                })
            elif block < s["p50"] * 0.5:
                warnings.append({
                    "severity": "soft",
                    "kind": "block",
                    "message": (
                        f"{bucket_label.capitalize()} median around "
                        f"{s['p50']:.0f} block (n={s['n']}); this looks under-tuned."
                    ),
                })

    return warnings
