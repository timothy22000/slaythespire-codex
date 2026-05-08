"""Slay the Spire — Build Me a Deck.

Text-prompt-driven deck synthesis. User describes a playstyle ("a deck that
wins by stacking poison and exhausting itself"), picks a character, gets back
a 20-card deck whose cards are scored by cosine similarity against the prompt
embedding. Optional starter-deck lock-in, optional curve and type-balance
enforcement, fit-summary panel, and an honesty layer when scores are weak.

Encoder: Qwen3-Embedding-0.6B, loaded locally (HF Inference API doesn't serve
the model). Pre-warmed in a background thread at startup.
"""

from __future__ import annotations

import html as _html
import threading
from typing import Any

import gradio as gr
import numpy as np
import pandas as pd

from shared.data import load_game
from shared.encoder import _model, encode_query
from shared.decks import (
    DRAFTED_CLASSES,
    DeckResult,
    build_deck,
)
from shared.bosses import generate_strategy


GAMES = {"Slay the Spire 1": "sts1", "Slay the Spire 2": "sts2"}
GAME_LABELS = {v: k for k, v in GAMES.items()}

# Honesty bands for avg_sim_picks (non-locked similarity).
HONESTY_GREEN = 0.65
HONESTY_NEUTRAL = 0.55
HONESTY_AMBER = 0.45

EXAMPLE_PROMPTS = [
    "Stack poison and exhaust the deck",
    "Lightning orbs and frost defense",
    "Infinite scaling block",
    "Draw the entire deck in one turn",
]


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _empty_state_html() -> str:
    return """
<div class="bmd-empty">
  <div class="bmd-empty-icon">⚔</div>
  <div class="bmd-empty-title">No deck yet</div>
  <div class="bmd-empty-tip">
    Describe a playstyle on the left, or click one of the example chips to
    get started. The algorithm picks cards by semantic similarity to your
    prompt, with optional starter cards locked in.
  </div>
</div>
""".strip()


def _error_banner_html(message: str) -> str:
    return (
        f'<div class="bmd-error">'
        f'<span class="bmd-error-icon">!</span>'
        f'<span>{_html.escape(message)}</span>'
        f'</div>'
    )


def _honesty_band(avg_sim: float) -> tuple[str, str] | None:
    """Return (cls, message) for the honesty banner, or None if avg_sim ≥ green threshold."""
    if avg_sim >= HONESTY_GREEN:
        return None
    if avg_sim >= HONESTY_NEUTRAL:
        return None  # neutral band has no warning banner; the quality banner says enough
    if avg_sim >= HONESTY_AMBER:
        return ("bmd-honesty-amber",
                f"Average fit: {avg_sim:.2f}. The embeddings found a partial match. "
                f"Try replacing playstyle words ('aggressive', 'control', 'glass cannon') "
                f"with mechanic names ('poison', 'block', 'exhaust', 'lightning'). "
                f"The model was trained on card text, not strategy guides.")
    return ("bmd-honesty-red",
            f"Average fit: {avg_sim:.2f}. The embeddings didn't find strong matches "
            f"for this prompt. Two common reasons: (1) the prompt names a mechanic "
            f"that doesn't exist for this character (e.g. 'lightning' on Ironclad — "
            f"Lightning is Defect-only), or (2) the prompt is too abstract. Try a "
            f"one-line description of mechanics, e.g. 'Apply weak and vulnerable, "
            f"then deal damage.'")


def _quality_label(avg_sim: float) -> str:
    if avg_sim >= HONESTY_GREEN:
        return "Strong fit"
    if avg_sim >= HONESTY_NEUTRAL:
        return "Moderate fit"
    if avg_sim >= HONESTY_AMBER:
        return "Weak fit"
    return "Poor fit"


def _quality_class(avg_sim: float) -> str:
    if avg_sim >= HONESTY_GREEN:
        return "bmd-quality-green"
    if avg_sim >= HONESTY_NEUTRAL:
        return "bmd-quality-neutral"
    return "bmd-quality-soft"


def _type_strip_html(type_count: dict[str, int], total: int) -> str:
    if total == 0:
        return ""
    a = type_count.get("Attack", 0)
    s = type_count.get("Skill", 0)
    p = type_count.get("Power", 0)
    pa, ps, pp = (100 * a / total, 100 * s / total, 100 * p / total)
    return (
        '<div class="bmd-strip-row">'
        '<span class="bmd-strip-label">Type</span>'
        '<div class="bmd-strip-bar">'
        f'<span class="bmd-strip-attack" style="width:{pa:.1f}%" title="{a} Attacks"></span>'
        f'<span class="bmd-strip-skill" style="width:{ps:.1f}%" title="{s} Skills"></span>'
        f'<span class="bmd-strip-power" style="width:{pp:.1f}%" title="{p} Powers"></span>'
        '</div>'
        f'<span class="bmd-strip-readout">{a}A · {s}S · {p}P</span>'
        '</div>'
    )


def _curve_strip_html(picks: list[Any]) -> str:
    """Histogram of cost 0..5+, derived from the picks list."""
    buckets = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5+": 0, "X": 0}
    for p in picks:
        c = p.cost
        if c == "-1":
            buckets["X"] += 1
        elif c == "-2":
            continue
        else:
            try:
                n = int(c)
            except (ValueError, TypeError):
                continue
            if n >= 5:
                buckets["5+"] += 1
            else:
                buckets[str(n)] += 1
    max_h = max(buckets.values()) or 1
    bars = "".join(
        f'<span class="bmd-curve-col" title="{label}: {count}">'
        f'<span class="bmd-curve-bar" style="height:{(count/max_h)*36:.1f}px"></span>'
        f'<span class="bmd-curve-label">{label}</span>'
        f'</span>'
        for label, count in buckets.items()
    )
    return (
        '<div class="bmd-strip-row">'
        '<span class="bmd-strip-label">Curve</span>'
        f'<div class="bmd-curve-grid">{bars}</div>'
        '</div>'
    )


def _quality_banner_html(result: DeckResult) -> str:
    if not result.picks:
        return ""
    avg = result.avg_sim_picks if any(not p.locked for p in result.picks) else result.avg_sim_all
    label = _quality_label(avg)
    cls = _quality_class(avg)
    n_picks = len(result.picks)
    n_locked = sum(1 for p in result.picks if p.locked)

    keywords_html = ""
    if result.top_keywords:
        kw_pills = "".join(
            f'<span class="bmd-kw-pill">{_html.escape(k)} <span class="bmd-kw-count">×{c}</span></span>'
            for k, c in result.top_keywords
        )
        keywords_html = f'<div class="bmd-kw-row">Top themes: {kw_pills}</div>'

    return (
        f'<div class="bmd-banner {cls}">'
        f'<div class="bmd-banner-headline">{label} · avg similarity {avg:.3f}</div>'
        f'<div class="bmd-banner-blurb">'
        f'{n_picks} cards ({n_locked} locked starters, {n_picks - n_locked} prompt-driven). '
        f'Avg cost {result.avg_cost:.1f}.'
        f'</div>'
        f'{keywords_html}'
        '<div class="bmd-strip-group">'
        f'{_type_strip_html(result.type_count, n_picks)}'
        f'{_curve_strip_html(result.picks)}'
        '</div>'
        '</div>'
    )


def _honesty_banner_html(result: DeckResult) -> str:
    # Don't fire when starters exactly fill the deck — there's no prompt-driven
    # signal to honest about. avg_sim_picks would be 0.0 and trip the red band
    # spuriously.
    if not any(not p.locked for p in result.picks):
        return ""
    band = _honesty_band(result.avg_sim_picks)
    if band is None:
        return ""
    cls, message = band
    return f'<div class="bmd-honesty-banner {cls}">{_html.escape(message)}</div>'


def _card_cell_html(p: Any, max_sim: float) -> str:
    badge = '<span class="bmd-card-locked">LOCKED</span>' if p.locked else ""
    cost = "X" if p.cost == "-1" else (p.cost or "—")
    desc_raw = (p.description or "").strip()
    desc_html = _html.escape(desc_raw) if desc_raw else "<i>(no description)</i>"
    sim_pct = max(0.0, min(1.0, p.similarity / max_sim if max_sim > 0 else 0.0)) * 100
    cell_cls = "bmd-card"
    if p.locked:
        cell_cls += " bmd-card-locked-bg"
    if p.similarity >= 0.70 and not p.locked:
        cell_cls += " bmd-card-strong"
    return (
        f'<div class="{cell_cls}">'
        f'<div class="bmd-card-row1">'
        f'<span class="bmd-card-name">{_html.escape(p.name)}</span>'
        f'{badge}'
        f'</div>'
        f'<div class="bmd-card-meta">{_html.escape(p.type_)} · {_html.escape(p.rarity)} · {_html.escape(p.color)}</div>'
        f'<div class="bmd-card-cost">Cost {_html.escape(cost)}</div>'
        f'<div class="bmd-card-desc">{desc_html}</div>'
        f'<div class="bmd-card-sim-row">'
        f'<div class="bmd-card-sim-bar"><span style="width:{sim_pct:.1f}%"></span></div>'
        f'<span class="bmd-card-sim-score">{p.similarity:.3f}</span>'
        f'</div>'
        f'</div>'
    )


def _deck_grid_html(result: DeckResult) -> str:
    if not result.picks:
        return ""
    sims = [p.similarity for p in result.picks if not p.locked]
    max_sim = max(sims) if sims else max((p.similarity for p in result.picks), default=1.0)
    cells = "".join(_card_cell_html(p, max_sim) for p in result.picks)
    return f'<div class="bmd-grid">{cells}</div>'


def _notes_html(notes: list[str]) -> str:
    if not notes:
        return ""
    items = "".join(f'<li>{_html.escape(n)}</li>' for n in notes)
    return f'<div class="bmd-notes"><div class="bmd-section-label">Notes</div><ul>{items}</ul></div>'


def _deck_to_dataframe(result: DeckResult) -> pd.DataFrame:
    """Mirror the deck as a Dataframe for the optional 'See full table' view."""
    return pd.DataFrame([
        {
            "similarity": round(p.similarity, 4),
            "locked": "✓" if p.locked else "",
            "name": p.name,
            "type": p.type_,
            "rarity": p.rarity,
            "color": p.color,
            "cost": "X" if p.cost == "-1" else p.cost,
            "description": p.description,
        }
        for p in result.picks
    ])


# ---------------------------------------------------------------------------
# Build handler
# ---------------------------------------------------------------------------

def _validate(prompt: str) -> tuple[bool, str]:
    if not prompt.strip():
        return False, "Describe a playstyle in the prompt box. Try one of the chips above."
    return True, ""


def build(
    game_label: str,
    character: str,
    prompt: str,
    deck_size: int,
    include_starters: str,
    allow_duplicates: bool,
    enforce_curve: bool,
    enforce_type_balance: bool,
    progress=gr.Progress(),
):
    """Returns (honesty_html, quality_html, deck_grid_html, notes_html,
                full_table_df, strategy_state_dict).

    The strategy_state_dict is consumed by the follow-up `_generate_strategy`
    handler that calls the LLM. Splitting build vs strategy keeps the deck
    rendering instant and the slower LLM call as a streaming follow-up.
    """
    progress(0, desc="Validating...")
    ok, err = _validate(prompt)
    if not ok:
        return _error_banner_html(err), "", "", "", pd.DataFrame(), {}

    game = GAMES[game_label]

    progress(0.05, desc="Loading data...")
    df, emb = load_game(game)

    if character not in df["color"].unique():
        return (
            _error_banner_html(
                f"Character {character!r} not found in {game.upper()}. Pick one from the dropdown."
            ),
            "", "", "", pd.DataFrame(), {},
        )

    progress(0.15, desc="Encoding prompt (loading model on cold start)...")
    qv = encode_query(prompt)

    progress(0.65, desc="Selecting cards...")
    result = build_deck(
        df, emb,
        game=game, character=character, query_vec=qv,
        deck_size=int(deck_size),
        include_starters=include_starters,  # type: ignore[arg-type]
        allow_duplicates=allow_duplicates,
        enforce_curve=enforce_curve,
        enforce_type_balance=enforce_type_balance,
    )

    progress(0.95, desc="Rendering...")
    if not result.picks and result.notes:
        return _error_banner_html(result.notes[0]), "", "", "", pd.DataFrame(), {}

    # Stash the deck state so the follow-up strategy handler can pick it up
    # without re-running the encoder + selection pass.
    state = {
        "game": game,
        "character": character,
        "user_prompt": prompt,
        "result": result,
    }

    return (
        _honesty_banner_html(result),
        _quality_banner_html(result),
        _deck_grid_html(result),
        _notes_html(result.notes),
        _deck_to_dataframe(result),
        state,
    )


def _strategy_loading_html() -> str:
    return (
        '<div class="bmd-strategy-loading">'
        '<span class="bmd-spinner"></span>'
        '<span>Asking Qwen2.5-72B for a strategic readout against this game\'s bosses…</span>'
        '</div>'
    )


def _strategy_html(markdown: str) -> str:
    """Render the LLM's markdown response inside the strategy panel."""
    return (
        '<div class="bmd-strategy">'
        '<div class="bmd-section-label">Strategy reasoning</div>'
        '<div class="bmd-strategy-body">'
        # Gradio's gr.HTML doesn't auto-render markdown; we render via a
        # nested gr.Markdown in the layout. This wrapper is just a styled
        # container; the inner markdown is set via a separate output.
        '</div>'
        '</div>'
    )


def render_strategy(state: dict, progress=gr.Progress()):
    """LLM follow-up that runs after `build` populates the deck state."""
    if not state or not state.get("result"):
        return ""
    progress(0.1, desc="Calling reasoning model...")
    try:
        text = generate_strategy(
            game=state["game"],
            character=state["character"],
            user_prompt=state["user_prompt"],
            result=state["result"],
        )
    except Exception as e:
        return (
            f'> ⚠️ Strategy generation failed: `{type(e).__name__}: {e}`. '
            f'Deck is still valid; the reasoning step is best-effort.'
        )
    progress(1.0, desc="Done")
    if text is None:
        return (
            "> ℹ️ Strategy reasoning is gated on `HF_TOKEN`. The Space owner "
            "can add that secret under **Settings → Variables and secrets** "
            "to enable LLM-generated boss-matchup analysis."
        )
    return text


# ---------------------------------------------------------------------------
# Game-toggle handler
# ---------------------------------------------------------------------------

def _on_game_change(game_label: str, current_character: str):
    game = GAMES[game_label]
    classes = DRAFTED_CLASSES[game]
    value = current_character if current_character in classes else classes[0]
    return gr.Dropdown(choices=classes, value=value)


def _on_chip_click(text: str):
    """Return the example prompt string to fill the prompt textbox."""
    return text


# ---------------------------------------------------------------------------
# UI build
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; }

/* Header */
.bmd-hero {
  padding: 18px 0 6px 0;
  border-bottom: 1px solid var(--border-color-primary);
  margin-bottom: 16px;
}
.bmd-hero h1 { margin: 0 0 6px 0; font-size: 26px; letter-spacing: -0.01em; }
.bmd-hero p {
  margin: 0; color: var(--body-text-color-subdued);
  font-size: 14px; line-height: 1.5; max-width: 75ch;
}

.bmd-section-label {
  font-size: 12px; font-weight: 600;
  letter-spacing: 0.06em; text-transform: uppercase;
  color: var(--body-text-color-subdued);
  margin: 8px 0 6px 2px;
}

/* Quick-start chip row */
.bmd-chip-row { gap: 6px !important; flex-wrap: wrap !important; }
.bmd-chip,
.bmd-chip button {
  font-size: 12px !important;
  padding: 6px 10px !important;
  border-radius: 999px !important;
  font-weight: 400 !important;
  white-space: nowrap;
}

/* Build button */
.bmd-build-btn,
.bmd-build-btn button {
  height: 56px !important;
  border-radius: 12px !important;
  font-size: 15px !important;
  font-weight: 600 !important;
  letter-spacing: 0.02em;
}

/* Empty state */
.bmd-empty {
  padding: 56px 24px;
  text-align: center;
  background: var(--background-fill-secondary);
  border: 2px dashed var(--border-color-primary);
  border-radius: 14px;
}
.bmd-empty-icon { font-size: 32px; margin-bottom: 10px; }
.bmd-empty-title { font-size: 16px; font-weight: 600; margin-bottom: 6px; }
.bmd-empty-tip { font-size: 14px; color: var(--body-text-color-subdued); max-width: 50ch; margin: 0 auto; line-height: 1.5; }

/* Error */
.bmd-error {
  display: flex; align-items: center; gap: 10px;
  padding: 14px 18px; border-radius: 10px;
  background: #fef2f2; color: #991b1b;
  border-left: 4px solid #dc2626;
  font-size: 14px;
}
.bmd-error-icon {
  font-weight: 700; font-size: 14px;
  width: 22px; height: 22px; border-radius: 50%;
  background: #dc2626; color: white;
  display: flex; align-items: center; justify-content: center;
}
.dark .bmd-error { background: rgba(220,38,38,0.10); color: #fca5a5; }

/* Honesty banner */
.bmd-honesty-banner {
  padding: 14px 18px; border-radius: 10px;
  border-left: 4px solid; margin-bottom: 14px;
  font-size: 14px; line-height: 1.5;
}
.bmd-honesty-amber { background: #fefce8; color: #854d0e; border-left-color: #ca8a04; }
.bmd-honesty-red   { background: #fef2f2; color: #991b1b; border-left-color: #dc2626; }
.dark .bmd-honesty-amber { background: rgba(202,138,4,0.10); color: #fde68a; }
.dark .bmd-honesty-red   { background: rgba(220,38,38,0.10); color: #fca5a5; }

/* Quality banner */
.bmd-banner {
  padding: 18px 22px;
  border-radius: 12px;
  border-left-width: 6px; border-left-style: solid;
  margin-bottom: 14px;
}
.bmd-quality-green   { border-left-color: #16a34a; background: #f0fdf4; }
.bmd-quality-neutral { border-left-color: #6b7280; background: #f9fafb; }
.bmd-quality-soft    { border-left-color: #ea580c; background: #fff7ed; }
.dark .bmd-quality-green   { background: rgba(22,163,74,0.10); }
.dark .bmd-quality-neutral { background: rgba(107,114,128,0.10); }
.dark .bmd-quality-soft    { background: rgba(234,88,12,0.10); }

.bmd-banner-headline { font-size: 17px; font-weight: 600; }
.bmd-banner-blurb { font-size: 13px; color: var(--body-text-color-subdued); margin: 4px 0 12px 0; }

/* Keyword pills */
.bmd-kw-row { font-size: 13px; margin-bottom: 12px; color: var(--body-text-color-subdued); }
.bmd-kw-pill {
  display: inline-block; padding: 3px 10px; margin-right: 6px;
  background: var(--background-fill-primary); border: 1px solid var(--border-color-primary);
  border-radius: 999px; font-size: 12px; color: var(--body-text-color);
}
.bmd-kw-count { color: var(--body-text-color-subdued); font-variant-numeric: tabular-nums; }

/* Composition strip */
.bmd-strip-group { display: flex; flex-direction: column; gap: 8px; }
.bmd-strip-row {
  display: flex; align-items: center; gap: 10px;
  font-size: 12px; color: var(--body-text-color-subdued);
}
.bmd-strip-label {
  flex: 0 0 56px; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.05em;
}
.bmd-strip-bar {
  flex: 1; height: 14px; border-radius: 4px;
  background: var(--background-fill-secondary);
  overflow: hidden; display: flex;
}
.bmd-strip-attack { background: #dc2626; height: 100%; display: inline-block; }
.bmd-strip-skill  { background: #2563eb; height: 100%; display: inline-block; }
.bmd-strip-power  { background: #9333ea; height: 100%; display: inline-block; }
.bmd-strip-readout { font-variant-numeric: tabular-nums; min-width: 80px; text-align: right; }

.bmd-curve-grid {
  flex: 1; display: flex; gap: 6px; align-items: flex-end;
  height: 56px; padding-bottom: 4px;
}
.bmd-curve-col {
  flex: 1; display: flex; flex-direction: column;
  align-items: center; justify-content: flex-end; gap: 3px;
  min-width: 0;
}
.bmd-curve-bar {
  width: 100%; background: var(--body-text-color-subdued); opacity: 0.55;
  border-radius: 2px 2px 0 0; min-height: 2px;
}
.bmd-curve-label { font-size: 10px; color: var(--body-text-color-subdued); }

/* Card grid */
.bmd-grid {
  display: grid; gap: 12px;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
}
.bmd-card {
  background: var(--background-fill-primary);
  border: 1px solid var(--border-color-primary);
  border-radius: 10px;
  padding: 12px 14px;
  display: flex; flex-direction: column;
  transition: border-color 0.15s ease, box-shadow 0.15s ease;
}
.bmd-card:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
.bmd-card-strong { border-color: rgba(22,163,74,0.45); }
.bmd-card-locked-bg { background: var(--background-fill-secondary); }

.bmd-card-row1 {
  display: flex; align-items: baseline; justify-content: space-between;
  gap: 8px; margin-bottom: 4px;
}
.bmd-card-name { font-size: 15px; font-weight: 600; }
.bmd-card-locked {
  font-size: 9.5px; font-weight: 600; letter-spacing: 0.06em;
  padding: 2px 7px; border-radius: 999px;
  background: #fde68a; color: #92400e;
}
.dark .bmd-card-locked { background: rgba(253,230,138,0.15); color: #fde68a; }
.bmd-card-meta {
  font-size: 11.5px; color: var(--body-text-color-subdued);
  margin-bottom: 4px;
  text-transform: capitalize;
}
.bmd-card-cost {
  font-size: 11.5px; color: var(--body-text-color-subdued);
  margin-bottom: 6px;
  font-variant-numeric: tabular-nums;
}
.bmd-card-desc {
  font-size: 13px; line-height: 1.4;
  margin: 0 0 10px 0;
  flex-grow: 1;
}
.bmd-card-sim-row {
  display: flex; align-items: center; gap: 8px;
  margin-top: auto; padding-top: 8px;
  border-top: 1px solid var(--border-color-accent-subdued);
}
.bmd-card-sim-bar {
  flex: 1; height: 6px; border-radius: 3px;
  background: var(--background-fill-secondary);
  overflow: hidden;
}
.bmd-card-sim-bar > span {
  display: block; height: 100%;
  background: linear-gradient(90deg, #6366f1, #ec4899);
}
.bmd-card-sim-score {
  font-size: 11px; color: var(--body-text-color-subdued);
  font-variant-numeric: tabular-nums;
  min-width: 36px; text-align: right;
}

/* Notes */
.bmd-notes {
  margin-top: 14px;
  padding: 10px 14px;
  border-left: 3px solid var(--border-color-primary);
  background: var(--background-fill-secondary);
  border-radius: 0 6px 6px 0;
  font-size: 13px;
}
.bmd-notes ul { margin: 4px 0 0 0; padding-left: 18px; }

/* Strategy panel */
.bmd-strategy-body {
  background: var(--background-fill-primary);
  border: 1px solid var(--border-color-primary);
  border-radius: 10px;
  padding: 16px 20px;
  font-size: 14px;
  line-height: 1.6;
}
.bmd-strategy-body h3 {
  margin: 14px 0 6px 0 !important;
  font-size: 14px !important;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--body-text-color);
}
.bmd-strategy-body h3:first-child { margin-top: 0 !important; }
.bmd-strategy-body ul { margin: 4px 0 8px 18px !important; padding: 0 !important; }
.bmd-strategy-body li { margin-bottom: 4px; }
.bmd-strategy-body p { margin: 4px 0 8px 0; }
.bmd-strategy-body strong { color: var(--body-text-color); }

/* Footer */
.bmd-footer {
  margin-top: 24px; padding-top: 16px;
  border-top: 1px solid var(--border-color-primary);
  font-size: 12px; color: var(--body-text-color-subdued);
}
"""


def make_demo() -> gr.Blocks:
    initial_game = "sts1"
    initial_classes = DRAFTED_CLASSES[initial_game]

    with gr.Blocks(title="Slay the Spire — Build Me a Deck", css=CUSTOM_CSS) as demo:
        gr.HTML(
            '<div class="bmd-hero">'
            '<h1>Slay the Spire — Build Me a Deck</h1>'
            '<p>Describe a playstyle. Get a deck that matches. The algorithm '
            'encodes your prompt with the same Qwen3 model used for the '
            'indexed cards, then picks cards by cosine similarity to the prompt '
            'with optional starter-deck lock-in, mana-curve correction, and '
            'attack/skill/power balance.</p>'
            '</div>'
        )

        with gr.Row(equal_height=False):
            # ---------- LEFT: form ----------
            with gr.Column(scale=2):
                game = gr.Radio(
                    choices=list(GAMES.keys()),
                    value=GAME_LABELS[initial_game],
                    label="Game",
                )
                character = gr.Dropdown(
                    choices=initial_classes,
                    value="ironclad",
                    label="Character",
                )

                gr.HTML('<div class="bmd-section-label">Quick start — example prompts</div>')
                with gr.Row(elem_classes="bmd-chip-row"):
                    chip_btns = [
                        gr.Button(p, size="sm", elem_classes="bmd-chip")
                        for p in EXAMPLE_PROMPTS
                    ]

                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="A deck that wins by stacking poison and exhausting itself",
                    lines=3,
                    max_lines=6,
                )

                deck_size = gr.Slider(
                    minimum=10, maximum=30, value=20, step=1, label="Deck size",
                )
                include_starters = gr.Radio(
                    choices=["Full starter deck", "Strike + Defend only", "None"],
                    value="Full starter deck",
                    label="Starters",
                    info="Locks the character's starting cards in place; remaining slots come from the prompt.",
                )

                with gr.Accordion("Advanced", open=False):
                    allow_duplicates = gr.Checkbox(
                        value=True, label="Allow duplicate copies (max 4 per name)",
                    )
                    enforce_curve = gr.Checkbox(
                        value=True, label="Enforce mana curve (30% low / 50% mid / 20% high)",
                    )
                    enforce_type_balance = gr.Checkbox(
                        value=True, label="Enforce type balance (50% Attack / 35% Skill / 15% Power)",
                    )

                build_btn = gr.Button(
                    "Build deck →", variant="primary", size="lg",
                    elem_classes="bmd-build-btn",
                )

            # ---------- RIGHT: results ----------
            with gr.Column(scale=3):
                gr.HTML('<div class="bmd-section-label">Result</div>')
                honesty_banner = gr.HTML("")
                quality_banner = gr.HTML(_empty_state_html())
                deck_grid = gr.HTML("")
                notes_panel = gr.HTML("")
                with gr.Accordion("See full table", open=False):
                    full_table = gr.Dataframe(
                        headers=["similarity", "locked", "name", "type", "rarity", "color", "cost", "description"],
                        interactive=False, wrap=True, row_count=(0, "dynamic"),
                    )

                gr.HTML('<div class="bmd-section-label" id="strategy-section">Strategy & boss matchups</div>')
                strategy_md = gr.Markdown(
                    "_Strategy reasoning will appear here after the deck builds. "
                    "Uses Qwen2.5-72B via HF Inference Providers to map the deck "
                    "onto each act's bosses for the chosen class._",
                    elem_classes="bmd-strategy-body",
                )

                # Hidden state passed from `build` to `render_strategy`.
                deck_state = gr.State({})

        # Event wiring
        game.change(
            fn=_on_game_change, inputs=[game, character], outputs=character,
        )

        for btn, text in zip(chip_btns, EXAMPLE_PROMPTS):
            btn.click(fn=_on_chip_click, inputs=gr.State(text), outputs=prompt)

        build_btn.click(
            fn=build,
            inputs=[game, character, prompt, deck_size, include_starters,
                    allow_duplicates, enforce_curve, enforce_type_balance],
            outputs=[honesty_banner, quality_banner, deck_grid, notes_panel,
                     full_table, deck_state],
        ).then(
            # Show a placeholder while the LLM runs, then replace with the result.
            fn=lambda: "_Generating strategy reasoning… (this takes 5-15s)_",
            inputs=None,
            outputs=strategy_md,
        ).then(
            fn=render_strategy,
            inputs=deck_state,
            outputs=strategy_md,
        )

        gr.HTML(
            '<div class="bmd-footer">'
            'Built with <a href="https://github.com/timothy22000/slaythespire-codex">slaythespire-codex</a>. '
            'Greedy similarity-based selection with constraint-feasibility checks; not an optimization solver. '
            "Decks are aspirational — actual STS runs build decks card-by-card from card-reward draws. "
            'Data: '
            '<a href="https://huggingface.co/datasets/t22000t/slay-the-spire-1-cards">STS1 cards</a> · '
            '<a href="https://huggingface.co/datasets/t22000t/slay-the-spire-1-card-embeddings">STS1 embeddings</a> · '
            '<a href="https://huggingface.co/datasets/t22000t/slay-the-spire-2-cards">STS2 cards</a> · '
            '<a href="https://huggingface.co/datasets/t22000t/slay-the-spire-2-card-embeddings">STS2 embeddings</a>.'
            '</div>'
        )

    return demo


# Pre-warm the encoder so the first user click doesn't pay the 30-60s model load.
threading.Thread(target=_model, daemon=True).start()

demo = make_demo()

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
