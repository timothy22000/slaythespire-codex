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
import random
import threading
from pathlib import Path
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
COST_RAW_TO_DISPLAY = {v: k for k, v in COST_DISPLAY_TO_RAW.items()}

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


import html as _html


# Tier styling: (css class, headline, supporting copy template, dot color)
_TIER_STYLES = {
    "strong": ("synergy-tier-strong", "Near-identical match",
               "Your card is effectively a re-skin of an existing one."),
    "soft": ("synergy-tier-soft", "Strong overlap",
             "Players may feel this is a variant rather than a new card."),
    "neutral": ("synergy-tier-neutral", "Similar but distinct",
                "Distinct enough to feel original."),
    "novel": ("synergy-tier-novel", "No close match",
              "Your card occupies its own niche."),
}


def _similarity_severity(sim: float) -> str:
    if sim >= SIM_TIER_IDENTICAL:
        return "strong"
    if sim >= SIM_TIER_VERY:
        return "soft"
    if sim >= SIM_TIER_SIMILAR:
        return "neutral"
    return "novel"


def _similarity_banner_html(top_match: pd.Series, sim: float) -> str:
    """Render the verdict banner as HTML with the top match highlighted as a card."""
    sev = _similarity_severity(sim)
    cls, headline, blurb = _TIER_STYLES[sev]

    name = _html.escape(str(top_match.get("name", "Unknown")))
    type_ = _html.escape(str(top_match.get("type", "")))
    cost = _html.escape(str(top_match.get("cost", "")))
    color = _html.escape(str(top_match.get("color", "")))
    rarity = _html.escape(str(top_match.get("rarity", "")))
    desc = _html.escape(str(top_match.get("description", "")) or "")
    cost_label = "X" if cost == "-1" else ("Unplayable" if cost == "-2" else (cost or "—"))

    return f"""
<div class="synergy-banner {cls}">
  <div class="synergy-banner-headline">{headline}</div>
  <div class="synergy-banner-blurb">{blurb}</div>
  <div class="synergy-featured-card">
    <div class="synergy-featured-row">
      <div class="synergy-featured-name">{name}</div>
      <div class="synergy-featured-sim">cosine {sim:.3f}</div>
    </div>
    <div class="synergy-featured-meta">
      <span>{type_ or '—'}</span>
      <span class="synergy-divider">·</span>
      <span>{rarity or '—'}</span>
      <span class="synergy-divider">·</span>
      <span>{color or '—'}</span>
      <span class="synergy-divider">·</span>
      <span>cost {cost_label}</span>
    </div>
    <div class="synergy-featured-desc">{desc or '<i>(no description)</i>'}</div>
  </div>
</div>
""".strip()


def _outlier_banner_html(warnings: list[dict]) -> str:
    if not warnings:
        return ""
    cls_for = {"strong": "synergy-outlier-strong",
               "soft": "synergy-outlier-soft",
               "info": "synergy-outlier-info"}
    icon_for = {"strong": "⚠", "soft": "⚡", "info": "ℹ"}
    items = "".join(
        f'<div class="synergy-outlier-item {cls_for.get(w["severity"], "synergy-outlier-info")}">'
        f'<span class="synergy-outlier-icon">{icon_for.get(w["severity"], "•")}</span>'
        f'<span>{_html.escape(w["message"])}</span>'
        f'</div>'
        for w in warnings
    )
    return f'<div class="synergy-outliers"><div class="synergy-section-label">Outlier check</div>{items}</div>'


def _empty_state_html() -> str:
    return """
<div class="synergy-empty">
  <div class="synergy-empty-icon">✨</div>
  <div class="synergy-empty-title">Submit a card on the left to analyze</div>
  <div class="synergy-empty-tip">
    New here? Click <b>🎲 Randomize</b> to load an existing card and see what
    a result looks like, then edit it to test your own ideas.
  </div>
</div>
""".strip()


def _error_banner_html(message: str) -> str:
    return (
        f'<div class="synergy-error">'
        f'<span class="synergy-error-icon">✕</span>'
        f'<span>{_html.escape(message)}</span>'
        f'</div>'
    )


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
        return _error_banner_html(err), "", pd.DataFrame()

    game = GAMES[game_label]
    df, emb = load_game(game)

    if color not in df["color"].unique():
        return (
            _error_banner_html(
                f"Color {color!r} not found in {game.upper()}; pick from the dropdown."
            ),
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

    # The featured card needs the full row (color, rarity, etc.); the dataframe
    # only carries a projection. Look up the full row by id where possible.
    top = neighbors.iloc[0]
    full_top = df[df["name"] == top["name"]].iloc[0] if (df["name"] == top["name"]).any() else top

    progress(1.0, desc="Done")
    sim_html = _similarity_banner_html(full_top, float(top["similarity"]))
    outlier_html = _outlier_banner_html(warnings)
    return sim_html, outlier_html, neighbors


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
# Randomize + file-upload handlers
# ---------------------------------------------------------------------------

# Form-field outputs in the order the handlers return values.
# Must match the `outputs=` list on the corresponding event wirings.
FORM_FIELD_KEYS = (
    "name", "type_", "rarity", "color", "cost",
    "description", "description_upgraded", "keywords",
    "damage", "block",
)


def _parse_keywords(raw: Any) -> str:
    """Parquet stores keywords as JSON string lists. Return comma-separated."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    if isinstance(raw, list):
        return ", ".join(str(x) for x in raw)
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return ""
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return ", ".join(str(x) for x in parsed)
        except json.JSONDecodeError:
            pass
        return s  # already comma-separated text
    return str(raw)


def _cost_raw_to_display(raw: Any) -> str:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return "1"
    raw_s = str(raw)
    return COST_RAW_TO_DISPLAY.get(raw_s, raw_s if raw_s in COST_DISPLAY_TO_RAW else "1")


def randomize(game_label: str):
    """Pick a random card from the chosen game and return form values."""
    game = GAMES[game_label]
    df, _ = load_game(game)
    row = df.sample(1, random_state=random.randint(0, 1 << 30)).iloc[0]

    def _opt(val, default=""):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return default
        return val

    return (
        _opt(row.get("name"), ""),
        _opt(row.get("type"), "Attack"),
        _opt(row.get("rarity"), "Common"),
        _opt(row.get("color"), "ironclad"),
        _cost_raw_to_display(row.get("cost")),
        _opt(row.get("description"), ""),
        _opt(row.get("description_upgraded"), ""),
        _parse_keywords(row.get("keywords")),
        float(row["damage"]) if pd.notna(row.get("damage")) else None,
        float(row["block"]) if pd.notna(row.get("block")) else None,
    )


def load_card_file(file_obj, game_label: str):
    """Populate form fields from an uploaded file.

    .json: parsed as a dict; recognised keys map to form fields. Cost can be
           int or string; gets converted to the display label.
    .txt/.md: file contents become the `description` field. Other fields
              left unchanged.
    Anything else: same fallback as .txt (treat as raw description).

    Returns ten gr.update() payloads matching FORM_FIELD_KEYS.
    """
    if file_obj is None:
        return tuple(gr.update() for _ in FORM_FIELD_KEYS)

    path = Path(file_obj.name if hasattr(file_obj, "name") else str(file_obj))
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return tuple(gr.update() for _ in FORM_FIELD_KEYS)

    suffix = path.suffix.lower()

    if suffix == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            game = GAMES[game_label]
            df, _ = load_game(game)

            def _coerce_enum(field, value, default):
                if value is None:
                    return gr.update()
                allowed = set(df[field].dropna().unique())
                return value if value in allowed else default

            kws = data.get("keywords")
            if isinstance(kws, list):
                kw_text = ", ".join(str(k) for k in kws)
            else:
                kw_text = _parse_keywords(kws)

            return (
                gr.update(value=str(data.get("name", "")) if data.get("name") is not None else gr.update()),
                gr.update(value=_coerce_enum("type", data.get("type"), "Attack")) if data.get("type") is not None else gr.update(),
                gr.update(value=_coerce_enum("rarity", data.get("rarity"), "Common")) if data.get("rarity") is not None else gr.update(),
                gr.update(value=_coerce_enum("color", data.get("color"), "ironclad")) if data.get("color") is not None else gr.update(),
                gr.update(value=_cost_raw_to_display(data.get("cost"))) if data.get("cost") is not None else gr.update(),
                gr.update(value=str(data.get("description", ""))) if data.get("description") is not None else gr.update(),
                gr.update(value=str(data.get("description_upgraded", ""))) if data.get("description_upgraded") is not None else gr.update(),
                gr.update(value=kw_text) if data.get("keywords") is not None else gr.update(),
                gr.update(value=float(data["damage"])) if data.get("damage") is not None else gr.update(),
                gr.update(value=float(data["block"])) if data.get("block") is not None else gr.update(),
            )

    # .txt / .md / unrecognized: dump the file content into the description
    # field. Trim to the same 1500-char hard cap the analyze handler enforces
    # so the user sees what will actually be used.
    desc_text = text.strip()[:1500]
    return (
        gr.update(),  # name
        gr.update(),  # type_
        gr.update(),  # rarity
        gr.update(),  # color
        gr.update(),  # cost
        gr.update(value=desc_text),  # description
        gr.update(),  # description_upgraded
        gr.update(),  # keywords
        gr.update(),  # damage
        gr.update(),  # block
    )


# ---------------------------------------------------------------------------
# UI build
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
.gradio-container { max-width: 1240px !important; margin: 0 auto !important; }

/* Header */
.synergy-hero {
  padding: 18px 0 6px 0;
  border-bottom: 1px solid var(--border-color-primary);
  margin-bottom: 16px;
}
.synergy-hero h1 {
  margin: 0 0 6px 0;
  font-size: 26px;
  letter-spacing: -0.01em;
}
.synergy-hero p {
  margin: 0;
  color: var(--body-text-color-subdued);
  font-size: 14px;
  line-height: 1.5;
  max-width: 70ch;
}

/* Section labels */
.synergy-section-label {
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--body-text-color-subdued);
  margin: 8px 0 6px 2px;
}

/* Verdict banner */
.synergy-banner {
  padding: 18px 22px;
  border-radius: 12px;
  border-left-width: 6px;
  border-left-style: solid;
  margin-bottom: 14px;
  background-clip: padding-box;
}
.synergy-tier-strong { border-left-color: #dc2626; background: #fef2f2; }
.synergy-tier-soft   { border-left-color: #ea580c; background: #fff7ed; }
.synergy-tier-neutral{ border-left-color: #6b7280; background: #f9fafb; }
.synergy-tier-novel  { border-left-color: #16a34a; background: #f0fdf4; }
.dark .synergy-tier-strong { background: rgba(220,38,38,0.10); }
.dark .synergy-tier-soft   { background: rgba(234,88,12,0.10); }
.dark .synergy-tier-neutral{ background: rgba(107,114,128,0.10); }
.dark .synergy-tier-novel  { background: rgba(22,163,74,0.10); }

.synergy-banner-headline { font-size: 18px; font-weight: 600; color: var(--body-text-color); }
.synergy-banner-blurb    { font-size: 14px; color: var(--body-text-color-subdued); margin: 4px 0 14px 0; }

/* Featured top-match card */
.synergy-featured-card {
  background: var(--background-fill-primary);
  border: 1px solid var(--border-color-primary);
  border-radius: 10px;
  padding: 14px 16px;
}
.synergy-featured-row { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; }
.synergy-featured-name { font-size: 18px; font-weight: 600; }
.synergy-featured-sim {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px;
  color: var(--body-text-color-subdued);
  background: var(--background-fill-secondary);
  padding: 3px 10px;
  border-radius: 999px;
  white-space: nowrap;
}
.synergy-featured-meta {
  display: flex; gap: 6px; flex-wrap: wrap;
  margin: 6px 0 10px 0;
  font-size: 13px; color: var(--body-text-color-subdued);
}
.synergy-divider { opacity: 0.4; }
.synergy-featured-desc {
  font-size: 14px; line-height: 1.5;
  padding-top: 8px;
  border-top: 1px solid var(--border-color-accent-subdued);
}

/* Outlier callouts */
.synergy-outliers { margin-bottom: 16px; }
.synergy-outlier-item {
  display: flex; align-items: flex-start; gap: 8px;
  padding: 10px 14px;
  border-radius: 8px;
  margin-bottom: 6px;
  font-size: 14px; line-height: 1.5;
  border-left-width: 4px; border-left-style: solid;
}
.synergy-outlier-strong { border-left-color: #dc2626; background: #fef2f2; }
.synergy-outlier-soft   { border-left-color: #ca8a04; background: #fefce8; }
.synergy-outlier-info   { border-left-color: #0284c7; background: #f0f9ff; }
.dark .synergy-outlier-strong { background: rgba(220,38,38,0.10); }
.dark .synergy-outlier-soft   { background: rgba(202,138,4,0.10); }
.dark .synergy-outlier-info   { background: rgba(2,132,199,0.10); }
.synergy-outlier-icon { flex: 0 0 auto; font-size: 16px; line-height: 1.4; }

/* Empty state */
.synergy-empty {
  padding: 48px 24px;
  text-align: center;
  background: var(--background-fill-secondary);
  border: 2px dashed var(--border-color-primary);
  border-radius: 14px;
}
.synergy-empty-icon { font-size: 28px; margin-bottom: 10px; }
.synergy-empty-title { font-size: 16px; font-weight: 600; margin-bottom: 6px; }
.synergy-empty-tip   { font-size: 14px; color: var(--body-text-color-subdued); max-width: 48ch; margin: 0 auto; line-height: 1.5; }

/* Error banner */
.synergy-error {
  display: flex; align-items: center; gap: 10px;
  padding: 14px 18px; border-radius: 10px;
  background: #fef2f2; color: #991b1b;
  border-left: 4px solid #dc2626;
  font-size: 14px;
}
.synergy-error-icon {
  font-weight: 700; font-size: 14px;
  width: 22px; height: 22px; border-radius: 50%;
  background: #dc2626; color: white;
  display: flex; align-items: center; justify-content: center;
}
.dark .synergy-error { background: rgba(220,38,38,0.10); color: #fca5a5; }

/* Form group spacing */
.synergy-form-group { margin-bottom: 4px; }

/* Footer */
.synergy-footer {
  margin-top: 24px;
  padding-top: 16px;
  border-top: 1px solid var(--border-color-primary);
  font-size: 12px;
  color: var(--body-text-color-subdued);
}
"""


def make_demo() -> gr.Blocks:
    initial_game = "sts1"
    opts = _form_options(initial_game)
    cost_display_choices = _cost_display_choices(initial_game)

    with gr.Blocks(
        title="Slay the Spire Synergy Inspector",
        css=CUSTOM_CSS,
    ) as demo:
        gr.HTML(
            '<div class="synergy-hero">'
            '<h1>Slay the Spire Synergy Inspector</h1>'
            '<p>Designing a custom card? Drop a spec in and find out which '
            'existing cards it overlaps with, plus a quick stats outlier check '
            'against the published distribution. Encodes via the same Qwen3 '
            'embedding model used for the indexed corpus, so similarity scores '
            'are directly comparable across cards.</p>'
            '</div>'
        )

        with gr.Row(equal_height=False):
            # ---------- LEFT: form ----------
            with gr.Column(scale=2):
                game = gr.Radio(
                    choices=list(GAMES.keys()),
                    value=GAME_LABELS[initial_game],
                    label="Compare against",
                )

                gr.HTML('<div class="synergy-section-label">Quick start</div>')
                with gr.Row():
                    randomize_btn = gr.Button("🎲 Randomize", size="sm", scale=1)
                    upload_file = gr.File(
                        label="Upload card",
                        file_types=[".json", ".txt", ".md"],
                        file_count="single",
                        type="filepath",
                        height=84,
                        scale=2,
                    )
                with gr.Accordion("Upload format", open=False):
                    gr.Markdown(
                        "**JSON file** with any of these keys (all optional):\n"
                        "```json\n"
                        "{\n"
                        '  "name": "Phantom Strike",\n'
                        '  "type": "Attack",\n'
                        '  "rarity": "Common",\n'
                        '  "color": "ironclad",\n'
                        '  "cost": "1",\n'
                        '  "description": "Deal 8 damage. Apply 2 Vulnerable.",\n'
                        '  "description_upgraded": "Deal 11 damage. Apply 3 Vulnerable.",\n'
                        '  "keywords": ["Exhaust"],\n'
                        '  "damage": 8,\n'
                        '  "block": null\n'
                        "}\n"
                        "```\n"
                        "Missing fields keep their current form values. Cost values "
                        "of `\"-1\"`/`-1` map to *X*; `\"-2\"`/`-2` to *Unplayable*; "
                        "`\"\"` to *Costless*.\n\n"
                        "**`.txt` or `.md` file:** the entire content goes into the "
                        "Description field (trimmed to 1500 chars)."
                    )

                gr.HTML('<div class="synergy-section-label">Card basics</div>')
                with gr.Group():
                    name = gr.Textbox(label="Name", placeholder="e.g. Phantom Strike", max_lines=1)
                    with gr.Row():
                        type_ = gr.Dropdown(choices=opts["type"], value="Attack", label="Type")
                        cost = gr.Dropdown(
                            choices=cost_display_choices,
                            value=_default_value(cost_display_choices, "1"),
                            label="Cost",
                            info="X scales with energy",
                        )
                    with gr.Row():
                        rarity = gr.Dropdown(choices=opts["rarity"], value="Common", label="Rarity")
                        color = gr.Dropdown(choices=opts["color"], value="ironclad", label="Class")

                gr.HTML('<div class="synergy-section-label">Card text</div>')
                with gr.Group():
                    description = gr.Textbox(
                        label="Description",
                        placeholder="Deal 8 damage. Apply 2 Vulnerable.",
                        lines=4,
                        max_lines=8,
                    )
                    keywords = gr.Textbox(
                        label="Keywords",
                        placeholder="comma-separated, e.g. Exhaust, Innate",
                    )
                    with gr.Accordion("Upgraded text (optional)", open=False):
                        description_upgraded = gr.Textbox(
                            label="Upgraded description",
                            placeholder="Deal 11 damage. Apply 3 Vulnerable.",
                            lines=2,
                        )

                gr.HTML('<div class="synergy-section-label">Stats <span style="opacity:0.6;font-weight:400;text-transform:none;">(optional, for outlier check)</span></div>')
                with gr.Group():
                    with gr.Row():
                        damage = gr.Number(label="Damage", precision=0, minimum=0, maximum=99, value=None)
                        block = gr.Number(label="Block", precision=0, minimum=0, maximum=99, value=None)

                analyze_btn = gr.Button("Analyze →", variant="primary", size="lg")

            # ---------- RIGHT: result ----------
            with gr.Column(scale=3):
                gr.HTML('<div class="synergy-section-label">Verdict</div>')
                sim_banner = gr.HTML(_empty_state_html())
                outlier_banner = gr.HTML("")
                gr.HTML('<div class="synergy-section-label">Closest existing cards</div>')
                neighbors = gr.Dataframe(
                    headers=["similarity", "name", "type", "rarity", "cost", "color", "description"],
                    interactive=False,
                    wrap=True,
                    row_count=(0, "dynamic"),
                )

        # ---------- Event wiring ----------
        game.change(
            fn=_on_game_change,
            inputs=[game, type_, rarity, color],
            outputs=[type_, rarity, color, cost],
        )

        form_outputs = [name, type_, rarity, color, cost,
                        description, description_upgraded, keywords,
                        damage, block]

        randomize_btn.click(
            fn=randomize,
            inputs=game,
            outputs=form_outputs,
        )

        upload_file.change(
            fn=load_card_file,
            inputs=[upload_file, game],
            outputs=form_outputs,
        )

        analyze_btn.click(
            fn=analyze,
            inputs=[game, name, type_, rarity, color, cost, description,
                    description_upgraded, keywords, damage, block],
            outputs=[sim_banner, outlier_banner, neighbors],
        )

        gr.HTML(
            '<div class="synergy-footer">'
            'Built with <a href="https://github.com/timothy22000/slaythespire-codex">slaythespire-codex</a>. '
            'Outlier baselines and similarity thresholds are calibrated against the indexed '
            'corpus; cards far outside the existing distribution may register as "novel" '
            'simply because nothing comparable exists. '
            'Data: '
            '<a href="https://huggingface.co/datasets/t22000t/slay-the-spire-1-cards">STS1 cards</a> · '
            '<a href="https://huggingface.co/datasets/t22000t/slay-the-spire-1-card-embeddings">STS1 embeddings</a> · '
            '<a href="https://huggingface.co/datasets/t22000t/slay-the-spire-2-cards">STS2 cards</a> · '
            '<a href="https://huggingface.co/datasets/t22000t/slay-the-spire-2-card-embeddings">STS2 embeddings</a>.'
            '</div>'
        )

    return demo


# Pre-warm the encoder at module load so the first user click doesn't pay the
# 30-60s model load. Daemon thread = won't block process exit.
threading.Thread(target=_model, daemon=True).start()

demo = make_demo()

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
