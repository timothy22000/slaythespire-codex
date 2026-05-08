"""Slay the Spire: Build Me a Deck.

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
  <div class="bmd-empty-eyebrow">No deck yet</div>
  <div class="bmd-empty-title">Describe a playstyle to begin</div>
  <div class="bmd-empty-tip">
    Type a prompt above or pick a starter prompt to populate one. The algorithm
    encodes your prompt with the same Qwen3 model that produced the indexed
    embeddings, then selects cards by similarity with type and curve constraints.
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
            f"that doesn't exist for this character (e.g. 'lightning' on Ironclad, "
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
    # Don't fire when starters exactly fill the deck, there's no prompt-driven
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
    badge = '<span class="bmd-card-locked">Starter</span>' if p.locked else ""
    cost = "X" if p.cost == "-1" else (p.cost or "-")
    desc_raw = (p.description or "").strip()
    desc_html = _html.escape(desc_raw) if desc_raw else "<span class=\"bmd-card-nodesc\">(no description)</span>"
    sim_pct = max(0.0, min(1.0, p.similarity / max_sim if max_sim > 0 else 0.0)) * 100
    cell_cls = "bmd-card"
    if p.locked:
        cell_cls += " bmd-card-locked-bg"
    if p.similarity >= 0.70 and not p.locked:
        cell_cls += " bmd-card-strong"
    color_attr = _html.escape(str(p.color or "colorless").lower())
    return (
        f'<div class="{cell_cls}" data-char="{color_attr}">'
        f'<div class="bmd-card-row1">'
        f'<span class="bmd-card-name">{_html.escape(p.name)}</span>'
        f'<span class="bmd-card-cost">{_html.escape(cost)}</span>'
        f'</div>'
        f'<div class="bmd-card-meta">'
        f'<span>{_html.escape(p.type_)}</span>'
        f'<span class="bmd-card-meta-sep">·</span>'
        f'<span>{_html.escape(p.rarity)}</span>'
        f'<span class="bmd-card-meta-sep">·</span>'
        f'<span>{_html.escape(p.color)}</span>'
        f'{badge}'
        f'</div>'
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
/* ============================================================
   Design tokens. Single accent (amber). Stone neutrals. No
   decorative gradients. Spacing on a 4px grid.
   ============================================================ */
:root {
  --bmd-accent: #d97706;          /* amber-600, slightly more vibrant */
  --bmd-accent-soft: #fef3c7;     /* amber-100 */
  --bmd-accent-ring: rgba(217,119,6,0.22);
  --bmd-success: #16a34a;
  --bmd-success-soft: #dcfce7;
  --bmd-warning: #d97706;
  --bmd-warning-soft:#fef3c7;
  --bmd-danger:  #dc2626;
  --bmd-danger-soft:#fee2e2;
  --bmd-info:    #2563eb;
  --bmd-info-soft:#dbeafe;
  --bmd-fg:      #1c1917;
  --bmd-fg-muted:#57534e;
  --bmd-fg-soft: #78716c;
  --bmd-surface: #ffffff;
  --bmd-surface-2:#fafaf9;
  --bmd-border:  #e7e5e4;
  --bmd-border-strong:#d6d3d1;

  /* Canonical STS character colors */
  --color-ironclad:    #dc2626;
  --color-silent:      #16a34a;
  --color-defect:      #2563eb;
  --color-watcher:     #9333ea;
  --color-necrobinder: #475569;
  --color-regent:      #ca8a04;
  --color-colorless:   #78716c;
  --color-curse:       #44403c;
}
.dark, .gradio-container.dark {
  --bmd-accent: #f59e0b;
  --bmd-accent-soft: rgba(245,158,11,0.14);
  --bmd-accent-ring: rgba(245,158,11,0.32);
  --bmd-success: #4ade80;
  --bmd-success-soft: rgba(74,222,128,0.14);
  --bmd-warning: #fbbf24;
  --bmd-warning-soft:rgba(251,191,36,0.14);
  --bmd-danger:  #f87171;
  --bmd-danger-soft:rgba(248,113,113,0.14);
  --bmd-info:    #60a5fa;
  --bmd-info-soft:rgba(96,165,250,0.14);
  --bmd-fg:      #f5f5f4;
  --bmd-fg-muted:#a8a29e;
  --bmd-fg-soft: #78716c;
  --bmd-surface: #1c1917;
  --bmd-surface-2:#292524;
  --bmd-border:  #292524;
  --bmd-border-strong:#44403c;

  --color-ironclad:    #f87171;
  --color-silent:      #4ade80;
  --color-defect:      #60a5fa;
  --color-watcher:     #c084fc;
  --color-necrobinder: #94a3b8;
  --color-regent:      #fbbf24;
  --color-colorless:   #a8a29e;
  --color-curse:       #78716c;
}

.gradio-container {
  max-width: 860px !important;
  margin: 0 auto !important;
  font-feature-settings: "ss01", "cv11";
}

/* ---------- Header ---------- */
.bmd-hero {
  padding: 32px 0 22px 0;
}
.bmd-hero h1 {
  margin: 0 0 4px 0;
  font-size: 30px;
  font-weight: 700;
  letter-spacing: -0.022em;
  color: var(--bmd-fg);
  line-height: 1.15;
}
.bmd-hero p {
  margin: 0;
  color: var(--bmd-fg-muted);
  font-size: 15px;
  line-height: 1.55;
  max-width: 62ch;
}
.bmd-hero-eyebrow {
  display: inline-block;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.10em;
  text-transform: uppercase;
  color: var(--bmd-accent);
  margin-bottom: 8px;
}

/* ---------- Input card (Apple-style chat box) ---------- */
.bmd-input-wrap { margin-bottom: 14px; }

.bmd-input-card {
  border: 1px solid var(--bmd-border) !important;
  border-radius: 20px !important;
  background: var(--bmd-surface) !important;
  box-shadow: 0 1px 2px rgba(0,0,0,0.03);
  padding: 18px 20px 14px 20px !important;
  transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
  margin-bottom: 16px;
}
.bmd-input-card:hover {
  border-color: var(--bmd-border-strong) !important;
}
.bmd-input-card:focus-within {
  border-color: var(--bmd-border-strong) !important;
  box-shadow: 0 8px 24px -8px rgba(0,0,0,0.10), 0 2px 6px -2px rgba(0,0,0,0.04);
}

.bmd-prompt-textarea textarea {
  border: none !important;
  background: transparent !important;
  resize: none !important;
  font-size: 17px !important;
  line-height: 1.5 !important;
  padding: 6px 2px !important;
  box-shadow: none !important;
  color: var(--bmd-fg) !important;
  font-feature-settings: normal;
  letter-spacing: -0.005em;
}
.bmd-prompt-textarea textarea::placeholder {
  color: var(--bmd-fg-soft);
}
.bmd-prompt-textarea textarea:focus {
  outline: none !important;
  box-shadow: none !important;
}

.bmd-input-controls {
  gap: 10px !important;
  align-items: center !important;
  margin-top: 12px !important;
  padding-top: 0;
  border-top: none;
}
.bmd-spacer { flex: 1; }

/* Hero Build button — large standalone CTA below input */
.bmd-hero-cta { margin: 0 0 18px 0; }
.bmd-hero-cta button {
  width: 100% !important;
  background: var(--bmd-accent) !important;
  color: white !important;
  border: none !important;
  border-radius: 16px !important;
  padding: 18px 24px !important;
  font-size: 16px !important;
  font-weight: 600 !important;
  letter-spacing: -0.005em;
  min-height: 56px !important;
  box-shadow: 0 1px 2px rgba(217,119,6,0.20),
              0 6px 16px -4px rgba(217,119,6,0.30);
  transition: transform 0.08s ease, box-shadow 0.18s ease, filter 0.18s ease;
  cursor: pointer;
}
.bmd-hero-cta button:hover {
  filter: brightness(1.04);
  box-shadow: 0 1px 2px rgba(217,119,6,0.25),
              0 10px 24px -4px rgba(217,119,6,0.40);
  transform: translateY(-1px);
}
.bmd-hero-cta button:active {
  transform: translateY(0);
  box-shadow: 0 1px 2px rgba(217,119,6,0.20),
              0 4px 10px -4px rgba(217,119,6,0.30);
}
.dark .bmd-hero-cta button {
  box-shadow: 0 1px 2px rgba(245,158,11,0.30),
              0 6px 16px -4px rgba(245,158,11,0.20);
}
.dark .bmd-hero-cta button:hover {
  box-shadow: 0 1px 2px rgba(245,158,11,0.35),
              0 10px 24px -4px rgba(245,158,11,0.30);
}

/* Game pills, segmented-control style */
.bmd-game-pills {
  border: none !important;
  background: transparent !important;
  padding: 0 !important;
}
.bmd-game-pills > .wrap,
.bmd-game-pills .form,
.bmd-game-pills .wrap-inner {
  background: var(--bmd-surface-2) !important;
  border-radius: 8px !important;
  padding: 3px !important;
  display: inline-flex !important;
  gap: 0 !important;
  border: 1px solid var(--bmd-border) !important;
}
.bmd-game-pills label {
  font-size: 12.5px !important;
  font-weight: 500 !important;
  padding: 5px 12px !important;
  border-radius: 6px !important;
  border: none !important;
  margin: 0 !important;
  cursor: pointer;
  color: var(--bmd-fg-muted);
  transition: color 0.15s ease, background 0.15s ease;
  min-height: 28px;
  display: inline-flex;
  align-items: center;
}
.bmd-game-pills label:hover { color: var(--bmd-fg); }
.bmd-game-pills label.selected,
.bmd-game-pills label[data-testid*="selected"],
.bmd-game-pills input:checked + label,
.bmd-game-pills label:has(input:checked) {
  background: var(--bmd-surface) !important;
  color: var(--bmd-fg) !important;
  box-shadow: 0 1px 2px rgba(0,0,0,0.05);
  font-weight: 600 !important;
}
.bmd-game-pills input[type="radio"] { display: none; }

/* Character dropdown — visually heavier per design ask, character-themed */
.bmd-char-dropdown {
  font-size: 13.5px !important;
  --char-color: var(--bmd-accent);
  --char-color-soft: var(--bmd-accent-soft);
}
.bmd-char-dropdown:has(input[value="ironclad"]),
.bmd-char-dropdown[data-char="ironclad"] {
  --char-color: var(--color-ironclad);
}
.bmd-char-dropdown:has(input[value="silent"]),
.bmd-char-dropdown[data-char="silent"] {
  --char-color: var(--color-silent);
}
.bmd-char-dropdown:has(input[value="defect"]),
.bmd-char-dropdown[data-char="defect"] {
  --char-color: var(--color-defect);
}
.bmd-char-dropdown:has(input[value="watcher"]),
.bmd-char-dropdown[data-char="watcher"] {
  --char-color: var(--color-watcher);
}
.bmd-char-dropdown:has(input[value="necrobinder"]),
.bmd-char-dropdown[data-char="necrobinder"] {
  --char-color: var(--color-necrobinder);
}
.bmd-char-dropdown:has(input[value="regent"]),
.bmd-char-dropdown[data-char="regent"] {
  --char-color: var(--color-regent);
}

.bmd-char-dropdown .wrap,
.bmd-char-dropdown > div > div {
  background: color-mix(in srgb, var(--char-color) 8%, var(--bmd-surface)) !important;
  border: 1.5px solid var(--char-color) !important;
  border-radius: 8px !important;
  padding: 2px 6px 2px 4px !important;
  min-height: 36px !important;
  transition: box-shadow 0.15s ease, background 0.2s ease, border-color 0.2s ease;
}
.bmd-char-dropdown:hover .wrap,
.bmd-char-dropdown:focus-within .wrap {
  box-shadow: 0 0 0 4px color-mix(in srgb, var(--char-color) 22%, transparent);
}
.bmd-char-dropdown input,
.bmd-char-dropdown select,
.bmd-char-dropdown .single-select {
  font-weight: 600 !important;
  color: var(--char-color) !important;
  text-transform: capitalize;
  padding-left: 8px !important;
}
.bmd-char-dropdown::before {
  content: "";
  position: absolute;
  width: 9px; height: 9px;
  border-radius: 50%;
  background: var(--char-color);
  margin: 14px 0 0 12px;
  z-index: 2;
  pointer-events: none;
  box-shadow: 0 0 0 2px var(--bmd-surface);
}

/* Inline Build button */
.bmd-build-btn-inline button {
  background: var(--bmd-accent) !important;
  color: white !important;
  border: 1px solid var(--bmd-accent) !important;
  border-radius: 8px !important;
  padding: 8px 16px !important;
  font-size: 13.5px !important;
  font-weight: 600 !important;
  letter-spacing: 0.005em;
  min-height: 36px;
  transition: filter 0.15s ease, transform 0.06s ease;
}
.bmd-build-btn-inline button:hover { filter: brightness(0.95); }
.bmd-build-btn-inline button:active { transform: translateY(1px); }

/* Quick-start suggestion cards */
.bmd-chip-label {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.10em;
  color: var(--bmd-fg-soft);
  margin: 18px 0 8px 2px;
  font-weight: 600;
}
.bmd-chip-row {
  display: grid !important;
  grid-template-columns: repeat(2, 1fr) !important;
  gap: 8px !important;
  margin-bottom: 16px;
}
@media (max-width: 540px) {
  .bmd-chip-row { grid-template-columns: 1fr !important; }
}
.bmd-chip { width: 100% !important; }
.bmd-chip button {
  width: 100% !important;
  text-align: left !important;
  font-size: 13px !important;
  padding: 12px 14px !important;
  border-radius: 10px !important;
  font-weight: 400 !important;
  white-space: normal !important;
  word-break: break-word;
  line-height: 1.4 !important;
  min-height: 52px !important;
  height: auto !important;
  background: var(--bmd-surface) !important;
  border: 1px solid var(--bmd-border) !important;
  color: var(--bmd-fg) !important;
  transition: background 0.15s ease, border-color 0.15s ease, transform 0.06s ease;
  display: flex !important;
  align-items: center !important;
  position: relative;
  padding-left: 32px !important;
}
.bmd-chip button::before {
  content: "›";
  position: absolute;
  left: 12px;
  top: 50%;
  transform: translateY(-50%);
  color: var(--bmd-accent);
  font-size: 18px;
  font-weight: 600;
  line-height: 1;
  transition: transform 0.15s ease;
}
.bmd-chip button:hover {
  background: var(--bmd-accent-soft) !important;
  border-color: var(--bmd-accent) !important;
}
.bmd-chip button:hover::before { transform: translateY(-50%) translateX(2px); }
.bmd-chip button:active { transform: translateY(1px); }

/* Options card (always visible) */
.bmd-options-label {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.10em;
  color: var(--bmd-fg-soft);
  margin: 8px 0 8px 2px;
  font-weight: 600;
}
.bmd-options-card {
  border: 1px solid var(--bmd-border) !important;
  border-radius: 12px !important;
  background: var(--bmd-surface-2) !important;
  padding: 14px 16px !important;
  margin-bottom: 4px;
}
.bmd-options-card label {
  font-size: 13px !important;
  color: var(--bmd-fg) !important;
}
.bmd-options-toggles {
  gap: 12px !important;
  flex-wrap: wrap !important;
  margin-top: 4px;
}
.bmd-options-toggles > * { flex: 1 1 200px !important; min-width: 180px !important; }

/* Results wrapper */
.bmd-results-wrap { margin-top: 32px; }

.bmd-section-label {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.10em;
  text-transform: uppercase;
  color: var(--bmd-fg-soft);
  margin: 16px 0 8px 2px;
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

/* Deck-pending empty state */
.bmd-empty {
  padding: 32px 28px;
  background: var(--bmd-surface-2);
  border: 1px solid var(--bmd-border);
  border-radius: 8px;
}
.bmd-empty-eyebrow {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.10em;
  text-transform: uppercase;
  color: var(--bmd-fg-soft);
  margin-bottom: 8px;
}
.bmd-empty-title {
  font-size: 16px;
  font-weight: 600;
  letter-spacing: -0.005em;
  color: var(--bmd-fg);
  margin-bottom: 4px;
}
.bmd-empty-tip {
  font-size: 14px;
  color: var(--bmd-fg-muted);
  max-width: 56ch;
  line-height: 1.6;
}

/* Error */
.bmd-error {
  display: flex; align-items: center; gap: 12px;
  padding: 14px 18px;
  border: 1px solid var(--bmd-border);
  border-left: 3px solid var(--bmd-danger);
  border-radius: 4px;
  background: var(--bmd-surface);
  color: var(--bmd-fg);
  font-size: 14px;
  line-height: 1.55;
}
.bmd-error-icon {
  flex: 0 0 auto;
  font-weight: 700; font-size: 12px;
  width: 20px; height: 20px;
  border-radius: 50%;
  background: var(--bmd-danger); color: white;
  display: flex; align-items: center; justify-content: center;
}

/* Honesty banner */
.bmd-honesty-banner {
  padding: 14px 18px;
  border: 1px solid var(--bmd-border);
  border-left: 4px solid;
  border-radius: 6px;
  margin-bottom: 14px;
  font-size: 14px;
  line-height: 1.55;
  color: var(--bmd-fg);
}
.bmd-honesty-amber {
  border-left-color: var(--bmd-warning);
  background: var(--bmd-warning-soft);
}
.bmd-honesty-red {
  border-left-color: var(--bmd-danger);
  background: var(--bmd-danger-soft);
}

/* Quality banner */
.bmd-banner {
  padding: 18px 22px;
  border: 1px solid var(--bmd-border);
  border-left: 4px solid;
  border-radius: 6px;
  margin-bottom: 16px;
}
.bmd-quality-green {
  border-left-color: var(--bmd-success);
  background: var(--bmd-success-soft);
}
.bmd-quality-neutral {
  border-left-color: var(--bmd-info);
  background: var(--bmd-info-soft);
}
.bmd-quality-soft {
  border-left-color: var(--bmd-warning);
  background: var(--bmd-warning-soft);
}

.bmd-banner-headline {
  font-size: 16px;
  font-weight: 600;
  letter-spacing: -0.005em;
  color: var(--bmd-fg);
  font-variant-numeric: tabular-nums;
}
.bmd-banner-blurb {
  font-size: 13px;
  color: var(--bmd-fg-muted);
  margin: 4px 0 14px 0;
}

/* Keyword pills */
.bmd-kw-row {
  font-size: 13px;
  margin-bottom: 14px;
  color: var(--bmd-fg-muted);
}
.bmd-kw-pill {
  display: inline-block;
  padding: 4px 11px;
  margin-right: 6px;
  background: var(--bmd-surface);
  border: 1px solid var(--bmd-accent);
  border-radius: 999px;
  font-size: 12px;
  font-weight: 500;
  color: var(--bmd-accent);
  letter-spacing: 0.005em;
}
.bmd-kw-count {
  color: var(--bmd-fg-soft);
  font-variant-numeric: tabular-nums;
  font-size: 11px;
  margin-left: 2px;
}

/* Composition strip */
.bmd-strip-group { display: flex; flex-direction: column; gap: 10px; }
.bmd-strip-row {
  display: flex; align-items: center; gap: 12px;
  font-size: 12px; color: var(--bmd-fg-muted);
}
.bmd-strip-label {
  flex: 0 0 52px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-size: 10.5px;
  color: var(--bmd-fg-soft);
}
.bmd-strip-bar {
  flex: 1;
  height: 8px;
  border-radius: 4px;
  background: var(--bmd-surface-2);
  overflow: hidden;
  display: flex;
}
.bmd-strip-attack { background: #dc2626; opacity: 0.85; height: 100%; display: inline-block; }
.bmd-strip-skill  { background: #1d4ed8; opacity: 0.85; height: 100%; display: inline-block; }
.bmd-strip-power  { background: #7e22ce; opacity: 0.85; height: 100%; display: inline-block; }
.bmd-strip-readout {
  font-variant-numeric: tabular-nums;
  min-width: 84px;
  text-align: right;
  font-size: 12px;
  color: var(--bmd-fg);
}

.bmd-curve-grid {
  flex: 1; display: flex; gap: 6px; align-items: flex-end;
  height: 48px; padding-bottom: 4px;
}
.bmd-curve-col {
  flex: 1; display: flex; flex-direction: column;
  align-items: center; justify-content: flex-end; gap: 4px;
  min-width: 0;
}
.bmd-curve-bar {
  width: 100%;
  background: var(--bmd-fg-soft);
  opacity: 0.55;
  border-radius: 2px 2px 0 0;
  min-height: 2px;
}
.bmd-curve-label {
  font-size: 10px;
  color: var(--bmd-fg-soft);
  font-variant-numeric: tabular-nums;
}

/* Card grid */
.bmd-grid {
  display: grid;
  gap: 12px;
  grid-template-columns: repeat(auto-fit, minmax(232px, 1fr));
  margin-bottom: 16px;
}
.bmd-card {
  position: relative;
  background: var(--bmd-surface);
  border: 1px solid var(--bmd-border);
  border-left: 3px solid var(--bmd-fg-soft);
  border-radius: 8px;
  padding: 14px 16px 14px 14px;
  display: flex; flex-direction: column;
  transition: border-color 0.15s ease, transform 0.15s ease, box-shadow 0.15s ease;
}
.bmd-card[data-char="ironclad"]    { border-left-color: var(--color-ironclad); }
.bmd-card[data-char="silent"]      { border-left-color: var(--color-silent); }
.bmd-card[data-char="defect"]      { border-left-color: var(--color-defect); }
.bmd-card[data-char="watcher"]     { border-left-color: var(--color-watcher); }
.bmd-card[data-char="necrobinder"] { border-left-color: var(--color-necrobinder); }
.bmd-card[data-char="regent"]      { border-left-color: var(--color-regent); }
.bmd-card[data-char="colorless"]   { border-left-color: var(--color-colorless); }
.bmd-card[data-char="curse"]       { border-left-color: var(--color-curse); }
.bmd-card:hover {
  border-color: var(--bmd-border-strong);
  transform: translateY(-1px);
  box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.bmd-card-strong {
  border-top-color: var(--bmd-accent);
  border-right-color: var(--bmd-accent);
  border-bottom-color: var(--bmd-accent);
}
.bmd-card-locked-bg { background: var(--bmd-surface-2); }

.bmd-card-row1 {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 4px;
}
.bmd-card-name {
  font-size: 15px;
  font-weight: 600;
  letter-spacing: -0.005em;
  color: var(--bmd-fg);
  line-height: 1.3;
  flex: 1 1 auto;
  min-width: 0;
}
.bmd-card-cost {
  flex: 0 0 auto;
  display: inline-flex;
  align-items: center; justify-content: center;
  min-width: 22px; height: 22px;
  padding: 0 7px;
  font-size: 11.5px;
  font-weight: 700;
  color: var(--bmd-fg);
  background: var(--bmd-surface-2);
  border: 1px solid var(--bmd-border);
  border-radius: 6px;
  font-variant-numeric: tabular-nums;
  margin-top: 1px;
}
.bmd-card-meta {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 5px;
  font-size: 11.5px;
  color: var(--bmd-fg-soft);
  margin-bottom: 10px;
  text-transform: capitalize;
  letter-spacing: 0.005em;
}
.bmd-card-meta-sep { color: var(--bmd-border-strong); }
.bmd-card-locked {
  margin-left: auto;
  font-size: 9.5px;
  font-weight: 700;
  letter-spacing: 0.10em;
  padding: 2px 7px;
  border-radius: 3px;
  background: var(--bmd-accent-soft);
  color: var(--bmd-accent);
  text-transform: uppercase;
  white-space: nowrap;
}
.bmd-card-desc {
  font-size: 13px;
  line-height: 1.5;
  margin: 0 0 12px 0;
  color: var(--bmd-fg);
  flex-grow: 1;
}
.bmd-card-nodesc { color: var(--bmd-fg-soft); font-style: italic; }
.bmd-card-sim-row {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: auto;
  padding-top: 10px;
  border-top: 1px solid var(--bmd-border);
}
.bmd-card-sim-bar {
  flex: 1;
  height: 4px;
  border-radius: 2px;
  background: var(--bmd-surface-2);
  overflow: hidden;
}
.bmd-card-sim-bar > span {
  display: block;
  height: 100%;
  background: var(--bmd-accent);
  opacity: 0.85;
}
.bmd-card-sim-score {
  font-size: 11px;
  color: var(--bmd-fg-soft);
  font-variant-numeric: tabular-nums;
  min-width: 36px;
  text-align: right;
}

/* Notes */
.bmd-notes {
  margin-top: 10px;
  padding: 12px 16px;
  border: 1px solid var(--bmd-border);
  border-left: 3px solid var(--bmd-fg-soft);
  background: var(--bmd-surface-2);
  border-radius: 4px;
  font-size: 13px;
  color: var(--bmd-fg-muted);
  line-height: 1.55;
}
.bmd-notes ul { margin: 4px 0 0 0; padding-left: 18px; }
.bmd-notes li { margin-bottom: 3px; }

/* ===== Strategy section: prominent, top-of-results ===== */
.bmd-strategy-empty {
  padding: 28px 24px;
  background: var(--bmd-surface);
  border: 1px solid var(--bmd-border);
  border-left: 3px solid var(--bmd-accent);
  border-radius: 4px;
  margin-bottom: 20px;
}
.bmd-strategy-empty-eyebrow {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.10em;
  text-transform: uppercase;
  color: var(--bmd-accent);
  margin-bottom: 8px;
}
.bmd-strategy-empty-title {
  font-size: 17px;
  font-weight: 600;
  letter-spacing: -0.005em;
  color: var(--bmd-fg);
  margin-bottom: 4px;
}
.bmd-strategy-empty-tip {
  font-size: 14px;
  color: var(--bmd-fg-muted);
  max-width: 60ch;
  line-height: 1.6;
}

.bmd-strategy-header {
  padding: 16px 22px 14px 22px;
  background: var(--bmd-surface);
  border: 1px solid var(--bmd-border);
  border-left: 3px solid var(--bmd-accent);
  border-radius: 4px 4px 0 0;
  border-bottom: none;
}
.bmd-strategy-eyebrow {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.10em;
  text-transform: uppercase;
  color: var(--bmd-accent);
  margin-bottom: 4px;
}
.bmd-strategy-title {
  font-size: 17px;
  font-weight: 600;
  letter-spacing: -0.005em;
  color: var(--bmd-fg);
}

.bmd-strategy-body {
  background: var(--bmd-surface);
  border: 1px solid var(--bmd-border);
  border-left: 3px solid var(--bmd-accent);
  border-top: none;
  border-radius: 0 0 4px 4px;
  padding: 4px 22px 20px 22px;
  font-size: 14.5px;
  line-height: 1.7;
  color: var(--bmd-fg);
  margin-bottom: 24px;
}
.bmd-strategy-body h3,
.bmd-strategy-body h4 {
  margin: 22px 0 8px 0 !important;
  font-size: 11px !important;
  letter-spacing: 0.10em !important;
  text-transform: uppercase !important;
  color: var(--bmd-fg-muted) !important;
  font-weight: 600 !important;
  border: none !important;
}
.bmd-strategy-body h3:first-child,
.bmd-strategy-body h4:first-child { margin-top: 8px !important; }
.bmd-strategy-body ul { margin: 6px 0 14px 18px !important; padding: 0 !important; }
.bmd-strategy-body li { margin-bottom: 6px; line-height: 1.6; }
.bmd-strategy-body p { margin: 6px 0 12px 0; }
.bmd-strategy-body strong { color: var(--bmd-fg); font-weight: 600; }
.bmd-strategy-body em { color: var(--bmd-fg-muted); }
.bmd-strategy-body code {
  font-size: 13px;
  background: var(--bmd-surface-2);
  padding: 1px 6px;
  border-radius: 3px;
  font-feature-settings: normal;
}

/* Footer */
.bmd-footer {
  margin-top: 32px;
  padding-top: 18px;
  border-top: 1px solid var(--bmd-border);
  font-size: 12px;
  color: var(--bmd-fg-soft);
  line-height: 1.6;
}
.bmd-footer a {
  color: var(--bmd-fg-muted);
  text-decoration: underline;
  text-decoration-color: var(--bmd-border-strong);
  text-underline-offset: 2px;
}
.bmd-footer a:hover { color: var(--bmd-accent); text-decoration-color: var(--bmd-accent); }
"""


def make_demo() -> gr.Blocks:
    initial_game = "sts1"
    initial_classes = DRAFTED_CLASSES[initial_game]

    with gr.Blocks(title="Slay the Spire: Build Me a Deck", css=CUSTOM_CSS) as demo:
        gr.HTML(
            '<div class="bmd-hero">'
            '<div class="bmd-hero-eyebrow">slaythespire-codex</div>'
            '<h1>Build me a deck</h1>'
            '<p>Describe a playstyle in plain English. The algorithm encodes the '
            "prompt with the same Qwen3 model that produced the indexed cards, "
            "then picks cards by similarity, balanced for mana curve and "
            "attack/skill/power mix.</p>"
            '</div>'
        )

        # ---------- INPUT CARD (Apple-style chat box) ----------
        with gr.Column(elem_classes="bmd-input-wrap"):
            with gr.Group(elem_classes="bmd-input-card"):
                prompt = gr.Textbox(
                    placeholder="Describe a playstyle. e.g. 'A deck that stacks poison and exhausts itself.'",
                    lines=3,
                    max_lines=8,
                    show_label=False,
                    container=False,
                    elem_classes="bmd-prompt-textarea",
                )
                with gr.Row(elem_classes="bmd-input-controls"):
                    game = gr.Radio(
                        choices=list(GAMES.keys()),
                        value=GAME_LABELS[initial_game],
                        show_label=False,
                        container=False,
                        elem_classes="bmd-game-pills",
                        scale=0,
                    )
                    character = gr.Dropdown(
                        choices=initial_classes,
                        value="ironclad",
                        show_label=False,
                        container=False,
                        elem_classes="bmd-char-dropdown",
                        scale=0,
                        min_width=160,
                    )
                    gr.HTML('<div class="bmd-spacer"></div>')

            # Hero Build CTA — pulled out of the input card so it dominates
            with gr.Column(elem_classes="bmd-hero-cta"):
                build_btn = gr.Button(
                    "Build deck  →",
                    variant="primary",
                )

            # Quick-start chip row, like ChatGPT's "try one of these"
            gr.HTML('<div class="bmd-chip-label">Try a starting prompt</div>')
            with gr.Row(elem_classes="bmd-chip-row"):
                chip_btns = [
                    gr.Button(p, size="sm", elem_classes="bmd-chip")
                    for p in EXAMPLE_PROMPTS
                ]

            # Options shown by default (no accordion)
            gr.HTML('<div class="bmd-options-label">Options</div>')
            with gr.Group(elem_classes="bmd-options-card"):
                deck_size = gr.Slider(
                    minimum=10, maximum=30, value=20, step=1, label="Deck size",
                )
                include_starters = gr.Radio(
                    choices=["Full starter deck", "Strike + Defend only", "None"],
                    value="Full starter deck",
                    label="Starters",
                    info="Locks the character's starting cards in place; remaining slots come from the prompt.",
                )
                with gr.Row(elem_classes="bmd-options-toggles"):
                    allow_duplicates = gr.Checkbox(
                        value=True, label="Allow duplicates (max 4 per card)",
                    )
                    enforce_curve = gr.Checkbox(
                        value=True, label="Enforce mana curve",
                    )
                    enforce_type_balance = gr.Checkbox(
                        value=True, label="Enforce type balance",
                    )

        # ---------- RESULTS (full-width below input) ----------
        with gr.Column(elem_classes="bmd-results-wrap"):
            # Strategy section pulled to the TOP of results, prominent styling.
            strategy_section = gr.HTML(
                '<div class="bmd-strategy-empty">'
                '<div class="bmd-strategy-empty-eyebrow">Strategy & boss matchups</div>'
                '<div class="bmd-strategy-empty-title">Per-act readout populates here</div>'
                '<div class="bmd-strategy-empty-tip">'
                "Once a deck is built, this section explains what the deck wants to do "
                "each turn, the strongest card synergies it carries, and how it handles "
                "each boss in each act for the chosen class. Generated by Qwen2.5-72B "
                "from the actual cards in the deck."
                '</div></div>'
            )
            strategy_md = gr.Markdown(
                "",
                elem_classes="bmd-strategy-body",
                visible=False,
            )

            # Honesty + quality banner
            honesty_banner = gr.HTML("")
            quality_banner = gr.HTML(_empty_state_html())

            # Deck grid + notes
            deck_grid = gr.HTML("")
            notes_panel = gr.HTML("")

            # Full table accordion (collapsed)
            with gr.Accordion("See full table", open=False):
                full_table = gr.Dataframe(
                    headers=["similarity", "locked", "name", "type", "rarity", "color", "cost", "description"],
                    interactive=False, wrap=True, row_count=(0, "dynamic"),
                )

            # Hidden state passed from `build` to `render_strategy`.
            deck_state = gr.State({})

        # Event wiring
        game.change(
            fn=_on_game_change, inputs=[game, character], outputs=character,
        )

        for btn, text in zip(chip_btns, EXAMPLE_PROMPTS):
            btn.click(fn=_on_chip_click, inputs=gr.State(text), outputs=prompt)

        def _show_strategy_loading():
            # Replace empty-state with the section header + loading copy,
            # and reveal the markdown component.
            return (
                '<div class="bmd-strategy-header">'
                '<div class="bmd-strategy-eyebrow">Strategy & boss matchups</div>'
                '<div class="bmd-strategy-title">Per-act readout for this deck</div>'
                '</div>',
                gr.update(value="_Generating strategic readout… (5-15s)_", visible=True),
            )

        def _show_strategy_result(state):
            text = render_strategy(state)
            return gr.update(value=text, visible=True)

        build_btn.click(
            fn=build,
            inputs=[game, character, prompt, deck_size, include_starters,
                    allow_duplicates, enforce_curve, enforce_type_balance],
            outputs=[honesty_banner, quality_banner, deck_grid, notes_panel,
                     full_table, deck_state],
        ).then(
            fn=_show_strategy_loading,
            inputs=None,
            outputs=[strategy_section, strategy_md],
        ).then(
            fn=_show_strategy_result,
            inputs=deck_state,
            outputs=strategy_md,
        )

        gr.HTML(
            '<div class="bmd-footer">'
            'Built with <a href="https://github.com/timothy22000/slaythespire-codex">slaythespire-codex</a>. '
            'Greedy similarity-based selection with constraint-feasibility checks; not an optimization solver. '
            "Decks are aspirational, actual STS runs build decks card-by-card from card-reward draws. "
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
