"""Slay the Spire Synergy Inspector.

Custom-card design assistant for STS modders. Fill in a hypothetical card,
get back:
  - Tiered similarity verdict against the chosen game's existing cards
    (identical-ish / very similar / similar / novel)
  - Top-10 closest existing cards by cosine similarity
  - Statistical outlier check on damage/block per (type, cost) baseline

Encoder is the same Qwen3-Embedding-0.6B used to embed the indexed cards,
loaded locally (HF Inference API doesn't serve this model, verified 404).
Pre-warmed in a background thread at startup.
"""

from __future__ import annotations

import json
import threading
from typing import Any

import gradio as gr
import numpy as np
import pandas as pd

from shared.data import load_game, topk_similar
from shared.encoder import _model, encode_query
from shared.normalize import build_card_document
from shared.outliers import check_outliers


GAMES = {"Slay the Spire 1": "sts1", "Slay the Spire 2": "sts2"}
GAME_LABELS = {v: k for k, v in GAMES.items()}

# Cost dropdown choices: human-readable labels mapped to parquet values.
COST_DISPLAY_TO_RAW = {
    "0": "0", "1": "1", "2": "2", "3": "3", "4": "4", "5": "5",
    "X": "-1",
    "Unplayable": "-2",
    "Costless": "",
}

# Tiered similarity warning thresholds (calibrated empirically against the
# indexed corpus; see plan file for rationale).
SIM_TIER_IDENTICAL = 0.97   # red banner
SIM_TIER_VERY = 0.92        # amber banner
SIM_TIER_SIMILAR = 0.85     # neutral note
# below 0.85: green note (novel)


# ---------------------------------------------------------------------------
# Form-field option builders (per-game enums)
# ---------------------------------------------------------------------------

def _enum_choices(game: str, column: str) -> list[str]:
    df, _ = load_game(game)
    return sorted(df[column].dropna().unique().tolist())


def _form_options(game: str) -> dict:
    return {
        "type": _enum_choices(game, "type"),
        "rarity": _enum_choices(game, "rarity"),
        "color": _enum_choices(game, "color"),
    }


# ---------------------------------------------------------------------------
# Analyze handler
# ---------------------------------------------------------------------------

def _validate(name, type_, rarity, color, cost_display, description, keywords):
    """Return (ok, error_message). cost_display is the human label."""
    if not description.strip() and not keywords.strip():
        return False, (
            "Please provide a description or keywords. Without text describing "
            "what the card does, similarity scores aren't meaningful."
        )
    if cost_display not in COST_DISPLAY_TO_RAW:
        return False, f"Unknown cost: {cost_display!r}"
    return True, ""


def _build_doc_dict(name, type_, rarity, color, cost_display, description,
                    description_upgraded, keywords) -> dict[str, Any]:
    """Assemble a row dict matching the parquet schema for build_card_document.

    Caps applied: name 200 chars, description/description_upgraded 1500 chars
    each, keywords max 8 entries × 100 chars each. Caps protect the encoder
    and tokenizer from pathological user input without changing semantics for
    realistic card specs.
    """
    cost_raw = COST_DISPLAY_TO_RAW[cost_display]
    kw_list = [k.strip()[:100] for k in keywords.split(",") if k.strip()][:8]
    desc = (description or "").strip()[:1500]
    desc_up = (description_upgraded or "").strip()[:1500]
    name_clean = (name or "").strip()[:200] or "Custom Card"
    return {
        "name": name_clean,
        "type": type_,
        "rarity": rarity,
        "color": color,
        "cost": cost_raw,
        "description": desc,
        "description_upgraded": desc_up,
        "keywords": json.dumps(kw_list, ensure_ascii=False),
    }


def _similarity_banner(top_match_name: str, sim: float) -> tuple[str, str]:
    """Return (markdown, severity) for the similarity verdict banner."""
    if sim >= SIM_TIER_IDENTICAL:
        return (
            f"### 🔴 Near-identical to **{top_match_name}**  \n"
            f"_Cosine similarity: {sim:.3f}_ — your card is effectively a re-skin.",
            "strong",
        )
    if sim >= SIM_TIER_VERY:
        return (
            f"### 🟠 Strong overlap with **{top_match_name}**  \n"
            f"_Cosine similarity: {sim:.3f}_ — players may feel this is a "
            f"variant rather than a new card.",
            "soft",
        )
    if sim >= SIM_TIER_SIMILAR:
        return (
            f"### ⚪ Closest match: **{top_match_name}**  \n"
            f"_Cosine similarity: {sim:.3f}_ — distinct enough to feel original.",
            "neutral",
        )
    return (
        f"### 🟢 No close match  \n"
        f"_Closest is **{top_match_name}** at {sim:.3f}_ — your card occupies "
        f"its own niche.",
        "novel",
    )


def _outlier_banner(warnings: list[dict]) -> str:
    if not warnings:
        return ""
    lines = ["### Outlier check"]
    for w in warnings:
        emoji = {"strong": "⚠️", "soft": "⚡", "info": "ℹ️"}.get(w["severity"], "•")
        lines.append(f"- {emoji} {w['message']}")
    return "\n".join(lines)


def analyze(
    game_label: str,
    name: str,
    type_: str,
    rarity: str,
    color: str,
    cost_display: str,
    description: str,
    description_upgraded: str,
    keywords: str,
    damage,
    block,
    progress=gr.Progress(),
):
    """Main handler. Returns (sim_banner_md, outlier_banner_md, neighbors_df)."""
    progress(0, desc="Validating...")
    ok, err = _validate(name, type_, rarity, color, cost_display, description, keywords)
    if not ok:
        return f"### ❌ {err}", "", pd.DataFrame()

    game = GAMES[game_label]
    df, emb = load_game(game)

    # Color must be valid for the chosen game (Dropdown enforces this in normal
    # use; defensive check for completeness).
    if color not in df["color"].unique():
        return (
            f"### ❌ Color {color!r} not found in {game.upper()}; pick from the dropdown.",
            "",
            pd.DataFrame(),
        )

    progress(0.15, desc="Building card document...")
    doc_dict = _build_doc_dict(
        name, type_, rarity, color, cost_display,
        description, description_upgraded, keywords,
    )
    card_text = build_card_document(doc_dict)

    progress(0.25, desc="Encoding card (loading model on cold start)...")
    qv = encode_query(card_text)

    progress(0.7, desc="Searching nearest neighbors...")
    neighbors = topk_similar(df, emb, qv, k=10)

    progress(0.9, desc="Checking outlier stats...")
    cost_raw = COST_DISPLAY_TO_RAW[cost_display]
    dmg = float(damage) if damage not in (None, "") and not pd.isna(damage) else None
    blk = float(block) if block not in (None, "") and not pd.isna(block) else None
    warnings = check_outliers(game, type_, cost_raw, dmg, blk)

    progress(1.0, desc="Done")
    top = neighbors.iloc[0]
    sim_md, _ = _similarity_banner(top["name"], float(top["similarity"]))
    outlier_md = _outlier_banner(warnings)
    return sim_md, outlier_md, neighbors


# ---------------------------------------------------------------------------
# Game-toggle handler
# ---------------------------------------------------------------------------

def _cost_display_choices(game: str) -> list[str]:
    df, _ = load_game(game)
    cost_choices = sorted(df["cost"].dropna().unique().tolist())
    raw_to_display = {v: k for k, v in COST_DISPLAY_TO_RAW.items()}
    return [raw_to_display.get(c, c) for c in cost_choices if c in raw_to_display]


def _default_value(choices: list[str], preferred: str) -> str:
    """Pick `preferred` if it's in choices; else first choice; else preferred."""
    if preferred in choices:
        return preferred
    return choices[0] if choices else preferred


def _on_game_change(game_label: str, current_type, current_rarity, current_color):
    """Repopulate type/rarity/color/cost dropdowns; preserve current value
    if it's still valid in the new game, else reset to a default."""
    game = GAMES[game_label]
    opts = _form_options(game)
    cost_display_choices = _cost_display_choices(game)

    def pick(current, choices, default):
        return current if current in choices else default

    return (
        gr.Dropdown(choices=opts["type"], value=pick(current_type, opts["type"], "Attack")),
        gr.Dropdown(choices=opts["rarity"], value=pick(current_rarity, opts["rarity"], "Common")),
        gr.Dropdown(choices=opts["color"], value=pick(current_color, opts["color"], "ironclad")),
        gr.Dropdown(choices=cost_display_choices,
                    value=_default_value(cost_display_choices, "1")),
    )


# ---------------------------------------------------------------------------
# UI build
# ---------------------------------------------------------------------------

def make_demo() -> gr.Blocks:
    initial_game = "sts1"
    opts = _form_options(initial_game)
    cost_display_choices = _cost_display_choices(initial_game)

    with gr.Blocks(title="Slay the Spire Synergy Inspector") as demo:
        gr.Markdown(
            "# Slay the Spire Synergy Inspector\n"
            "Designing a custom Slay the Spire card? Drop the spec in here and "
            "find out what existing cards it overlaps with, plus a quick stats "
            "outlier check against the published distribution.\n\n"
            "Encodes via the same `Qwen/Qwen3-Embedding-0.6B` model used to "
            "produce the indexed embeddings. First analyze takes 30-60s while "
            "the model loads; subsequent ones are ~2s.\n\n"
            "Data: "
            "[STS1 cards](https://huggingface.co/datasets/t22000t/slay-the-spire-1-cards) · "
            "[STS1 embeddings](https://huggingface.co/datasets/t22000t/slay-the-spire-1-card-embeddings) · "
            "[STS2 cards](https://huggingface.co/datasets/t22000t/slay-the-spire-2-cards) · "
            "[STS2 embeddings](https://huggingface.co/datasets/t22000t/slay-the-spire-2-card-embeddings)"
        )

        with gr.Row():
            with gr.Column(scale=2):
                game = gr.Radio(
                    choices=list(GAMES.keys()),
                    value=GAME_LABELS[initial_game],
                    label="Compare against",
                )
                name = gr.Textbox(label="Name", placeholder="e.g. Phantom Strike", max_lines=1)
                type_ = gr.Dropdown(choices=opts["type"], value="Attack", label="Type")
                rarity = gr.Dropdown(choices=opts["rarity"], value="Common", label="Rarity")
                color = gr.Dropdown(choices=opts["color"], value="ironclad", label="Color / class")
                cost = gr.Dropdown(
                    choices=cost_display_choices,
                    value=_default_value(cost_display_choices, "1"),
                    label="Cost",
                    info="X = scales with energy; Unplayable = curse/status",
                )
                description = gr.Textbox(
                    label="Description",
                    placeholder="Deal 8 damage. Apply 2 Vulnerable.",
                    lines=4,
                    max_lines=8,
                )
                with gr.Accordion("Upgraded text (optional)", open=False):
                    description_upgraded = gr.Textbox(
                        label="Upgraded description",
                        placeholder="Deal 11 damage. Apply 3 Vulnerable.",
                        lines=2,
                    )
                keywords = gr.Textbox(
                    label="Keywords",
                    placeholder="comma-separated, e.g. Exhaust, Innate",
                )
                with gr.Accordion("Numeric stats (optional, for outlier check)", open=True):
                    damage = gr.Number(label="Damage", precision=0, minimum=0, maximum=99, value=None)
                    block = gr.Number(label="Block", precision=0, minimum=0, maximum=99, value=None)

                analyze_btn = gr.Button("Analyze", variant="primary")

            with gr.Column(scale=3):
                sim_banner = gr.Markdown("_Submit a card on the left to analyze._")
                outlier_banner = gr.Markdown("")
                gr.Markdown("### Closest existing cards")
                neighbors = gr.Dataframe(
                    headers=["similarity", "name", "type", "rarity", "cost", "color", "description"],
                    interactive=False,
                    wrap=True,
                )

        game.change(
            fn=_on_game_change,
            inputs=[game, type_, rarity, color],
            outputs=[type_, rarity, color, cost],
        )

        analyze_btn.click(
            fn=analyze,
            inputs=[game, name, type_, rarity, color, cost, description,
                    description_upgraded, keywords, damage, block],
            outputs=[sim_banner, outlier_banner, neighbors],
        )

        gr.Markdown(
            "---\n"
            "Built with [slaythespire-codex](https://github.com/timothy22000/slaythespire-codex). "
            "Outlier baselines and similarity thresholds are calibrated against the "
            "indexed corpus; cards far outside the existing distribution may register "
            "as 'novel' simply because nothing comparable exists."
        )

    return demo


# Pre-warm the encoder at module load so the first user click doesn't pay the
# 30-60s model load. Daemon thread = won't block process exit.
threading.Thread(target=_model, daemon=True).start()

demo = make_demo()

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
