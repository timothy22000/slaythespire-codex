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

import base64
import json
import os
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
    cost_label = "X" if cost == "-1" else ("Unplayable" if cost == "-2" else (cost or "-"))

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
      <span>{type_ or '-'}</span>
      <span class="synergy-divider">·</span>
      <span>{rarity or '-'}</span>
      <span class="synergy-divider">·</span>
      <span>{color or '-'}</span>
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


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
IMAGE_MIME = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}
TEXT_EXTS = {".txt", ".md"}
JSON_EXTS = {".json"}
CSV_EXTS = {".csv"}

VISION_EXTRACTION_PROMPT = """You are looking at a screenshot of a Slay the Spire card. Extract its details and return ONLY a JSON object with the fields below. Use null for any field you cannot determine confidently. Do not include markdown fences or any prose, just the JSON.

{
  "name": "<exact card name as shown>",
  "type": "Attack" | "Skill" | "Power" | "Status" | "Curse" | "Quest" | null,
  "rarity": "Basic" | "Common" | "Uncommon" | "Rare" | "Special" | null,
  "color": "ironclad" | "silent" | "defect" | "watcher" | "necrobinder" | "regent" | "colorless" | "curse" | null,
  "cost": "0" | "1" | "2" | "3" | "X" | "Unplayable" | "Costless" | null,
  "description": "<exact card description text>",
  "description_upgraded": null,
  "keywords": ["..."] or null,
  "damage": <integer or null, primary damage value if 'Deal N damage' appears>,
  "block": <integer or null, primary block value if 'Gain N block' appears>
}

Color guide: card border color signals the character class.
  Red = ironclad. Green = silent. Blue = defect. Purple = watcher.
  Black/bone = necrobinder. Gold = regent. Gray = colorless. Black-with-skull = curse.
"""


def _form_updates_from_card_dict(data: dict, game_label: str):
    """Build ten gr.update() payloads from a card-shaped dict.

    Used by both the .json upload path and the screenshot extraction path.
    Order matches FORM_FIELD_KEYS:
      name, type_, rarity, color, cost,
      description, description_upgraded, keywords, damage, block.
    """
    game = GAMES[game_label]
    df, _ = load_game(game)

    def _opt_str(key):
        v = data.get(key)
        return gr.update(value=str(v)) if v is not None else gr.update()

    def _enum_update(field):
        v = data.get(field)
        if v is None:
            return gr.update()
        allowed = set(df[field].dropna().unique())
        # If the model returned a value that isn't in the per-game enum
        # (e.g. STS1 user uploaded an STS2-only color), drop it silently
        # rather than corrupt the form.
        return gr.update(value=v) if v in allowed else gr.update()

    def _num_update(key):
        v = data.get(key)
        if v is None or (isinstance(v, str) and not v.strip()):
            return gr.update()
        try:
            return gr.update(value=float(v))
        except (TypeError, ValueError):
            return gr.update()

    kws = data.get("keywords")
    if isinstance(kws, list):
        kw_update = gr.update(value=", ".join(str(k) for k in kws))
    elif kws is not None:
        kw_update = gr.update(value=_parse_keywords(kws))
    else:
        kw_update = gr.update()

    return (
        _opt_str("name"),
        _enum_update("type"),
        _enum_update("rarity"),
        _enum_update("color"),
        gr.update(value=_cost_raw_to_display(data.get("cost"))) if data.get("cost") is not None else gr.update(),
        _opt_str("description"),
        _opt_str("description_upgraded"),
        kw_update,
        _num_update("damage"),
        _num_update("block"),
    )


VISION_MODEL = "Qwen/Qwen2.5-VL-72B-Instruct"


def _extract_card_from_image(image_path: Path) -> dict:
    """Call a vision model via HF Inference Providers to extract card fields.
    Returns a card-shaped dict (same keys as the JSON-upload format).
    Raises with a clear message if HF_TOKEN isn't set."""
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN isn't set in this Space's secrets. The Space owner needs "
            "to add it under Settings → Variables and secrets to enable "
            "screenshot extraction. The token must have inference provider access."
        )

    from huggingface_hub import InferenceClient
    client = InferenceClient(provider="auto", api_key=hf_token)

    suffix = image_path.suffix.lower().lstrip(".")
    mime = IMAGE_MIME.get(suffix, "image/png")
    image_bytes = image_path.read_bytes()
    img_b64 = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:{mime};base64,{img_b64}"

    resp = client.chat.completions.create(
        model=VISION_MODEL,
        max_tokens=600,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": VISION_EXTRACTION_PROMPT},
            ],
        }],
    )

    text = (resp.choices[0].message.content or "").strip()
    # Strip markdown code fences defensively, in case the model wrapped its output.
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1:]
        text = text.rsplit("```", 1)[0].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Vision model returned invalid JSON: {e}. Raw output: {text[:200]!r}"
        )


def _load_csv_first_row(path: Path) -> dict | None:
    """Read a CSV and return the first row as a dict.

    Uses pandas' python engine with `sep=None` to sniff `,`/`;`/`\\t`.
    Warns if the file has multiple rows.
    """
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        return None

    if df.empty:
        return None

    if len(df) > 1:
        gr.Info(f"CSV has {len(df)} rows; using only the first.")

    row = df.iloc[0].to_dict()
    # Normalise keys to lowercase for a forgiving match against the schema.
    return {str(k).strip().lower(): v for k, v in row.items()}


def load_card_file(file_obj, game_label: str):
    """Populate form fields from an uploaded file.

    .json:               parsed as a dict; recognised keys map to form fields.
    .csv:                first row's columns map to form fields (header row
                         expected, lowercase keys; same field names as JSON).
    .png/.jpg/.webp:     Vision model (Qwen2.5-VL-72B via HF) extracts card
                         fields and populates the form.
    .txt/.md/anything:   file content goes into the description field.

    Returns ten gr.update() payloads matching FORM_FIELD_KEYS. Surfaces
    failures via gr.Warning rather than throwing.
    """
    if file_obj is None:
        return tuple(gr.update() for _ in FORM_FIELD_KEYS)

    path = Path(file_obj.name if hasattr(file_obj, "name") else str(file_obj))
    suffix = path.suffix.lower()

    # Image path: extract via vision model.
    if suffix in IMAGE_EXTS:
        try:
            data = _extract_card_from_image(path)
        except Exception as e:
            gr.Warning(f"Screenshot extraction failed: {e}")
            return tuple(gr.update() for _ in FORM_FIELD_KEYS)
        if not isinstance(data, dict):
            gr.Warning("Vision model didn't return a card-shaped JSON object.")
            return tuple(gr.update() for _ in FORM_FIELD_KEYS)
        recognized_name = data.get("name")
        if recognized_name:
            gr.Info(f"Extracted card details from screenshot (recognized as {recognized_name!r}).")
        else:
            gr.Info("Extracted card details from screenshot.")
        return _form_updates_from_card_dict(data, game_label)

    # CSV path
    if suffix in CSV_EXTS:
        data = _load_csv_first_row(path)
        if data is None:
            gr.Warning("Couldn't parse CSV. Expected a header row with column names like `name`, `type`, `cost`, `description`.")
            return tuple(gr.update() for _ in FORM_FIELD_KEYS)
        return _form_updates_from_card_dict(data, game_label)

    # JSON / text path
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        gr.Warning("Couldn't read uploaded file.")
        return tuple(gr.update() for _ in FORM_FIELD_KEYS)

    if suffix in JSON_EXTS:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            return _form_updates_from_card_dict(data, game_label)
        gr.Warning("JSON file did not contain a card-shaped object; dropping content into Description.")

    # .txt / .md / unrecognized / failed-JSON fallback: dump content into
    # the description field, trimmed to the same 1500-char cap the analyze
    # handler enforces.
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

/* Quick-start button row: matched pair of clean, equal-width buttons. */
.synergy-quick-row { gap: 10px !important; align-items: stretch !important; }
.synergy-quick-btn,
.synergy-quick-btn button,
.synergy-quick-btn label {
  height: 56px !important;
  min-height: 56px !important;
  border-radius: 12px !important;
  font-size: 15px !important;
  font-weight: 500 !important;
  letter-spacing: 0.01em;
  transition: transform 0.06s ease, box-shadow 0.12s ease, background 0.12s ease;
}
.synergy-quick-btn button:hover,
.synergy-quick-btn label:hover {
  transform: translateY(-1px);
  box-shadow: 0 2px 6px rgba(0,0,0,0.06);
}
.synergy-quick-btn button:active,
.synergy-quick-btn label:active {
  transform: translateY(0);
  box-shadow: none;
}

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
                with gr.Row(elem_classes="synergy-quick-row"):
                    randomize_btn = gr.Button(
                        "🎲  Randomize",
                        variant="secondary",
                        scale=1,
                        elem_classes="synergy-quick-btn",
                    )
                    upload_btn = gr.UploadButton(
                        "📁  Upload card",
                        file_types=[
                            ".json", ".csv", ".txt", ".md",
                            ".png", ".jpg", ".jpeg", ".webp",
                        ],
                        file_count="single",
                        variant="secondary",
                        scale=1,
                        elem_classes="synergy-quick-btn",
                    )
                with gr.Accordion("Upload formats", open=False):
                    gr.Markdown(
                        "Drop one file into the upload widget; the format is "
                        "detected from the extension.\n\n"
                        "**📸 Screenshot** (`.png` / `.jpg` / `.jpeg` / `.webp`): "
                        "vision model (Qwen2.5-VL-72B via HF Inference Providers) "
                        "reads the card's name, cost, type, description, and "
                        "stats, then fills the form. Takes ~3-5 seconds. "
                        "Requires `HF_TOKEN` to be set in the Space's secrets "
                        "- without it, this path returns a clear error.\n\n"
                        "**JSON** (`.json`), fields all optional:\n"
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
                        "```\n\n"
                        "**CSV** (`.csv`), header row required, same field "
                        "names as the JSON keys above. Only the first row is "
                        "used; multi-row files surface a one-line notice.\n\n"
                        "Cost values of `\"-1\"`/`-1` map to *X*; `\"-2\"`/`-2` "
                        "to *Unplayable*; `\"\"` to *Costless*. Missing fields "
                        "keep their current form values.\n\n"
                        "**`.txt` or `.md`:** the entire content goes into the "
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

        upload_btn.upload(
            fn=load_card_file,
            inputs=[upload_btn, game],
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
