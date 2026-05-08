"""Slay the Spire Archetype Map.

Interactive UMAP scatter of every card in STS1 + STS2, colored by character.
Hover for card details; pick a card from the dropdown to see its 5 nearest
neighbors in embedding space.

Data is loaded from the published HF datasets:
  t22000t/slay-the-spire-{1,2}-cards
  t22000t/slay-the-spire-{1,2}-card-embeddings
"""

from __future__ import annotations

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from shared.data import load_game, topk_similar


GAMES = ("sts1", "sts2")
GAME_LABELS = {"sts1": "Slay the Spire 1", "sts2": "Slay the Spire 2"}


def _truncate(s: str, n: int = 80) -> str:
    if not isinstance(s, str):
        return ""
    return s if len(s) <= n else s[: n - 1] + "…"


HIGHLIGHT_K = 10  # how many neighbors to highlight on the scatter


def render_scatter(
    game: str,
    color_filter: list[str],
    type_filter: list[str],
    highlight_name: str | None = None,
) -> go.Figure:
    df, emb = load_game(game)
    mask = pd.Series(True, index=df.index)
    if color_filter:
        mask &= df["color"].isin(color_filter)
    if type_filter:
        mask &= df["type"].isin(type_filter)
    sub = df.loc[mask].copy()
    sub["short_desc"] = sub["description"].fillna("").map(lambda s: _truncate(s, 80))

    if len(sub) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No cards match the current filters.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )
        fig.update_layout(height=600, title=f"{GAME_LABELS[game]}: 0 cards")
        return fig

    title = f"{GAME_LABELS[game]}: {len(sub)} of {len(df)} cards"
    if highlight_name:
        title += f" · highlighting top-{HIGHLIGHT_K} similar to “{highlight_name}”"

    fig = px.scatter(
        sub,
        x="umap_x",
        y="umap_y",
        color="color",
        custom_data=["name", "type", "rarity", "cost", "short_desc"],
        title=title,
        height=600,
    )
    base_opacity = 0.85 if not highlight_name else 0.3
    fig.update_traces(
        marker=dict(size=8, line=dict(width=0.5, color="white"), opacity=base_opacity),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "%{customdata[1]} · %{customdata[2]} · cost %{customdata[3]}<br>"
            "%{customdata[4]}<extra></extra>"
        ),
    )

    if highlight_name:
        match = df.index[df["name"] == highlight_name]
        if len(match):
            picked_idx = int(match[0])
            picked = df.iloc[picked_idx]
            # Similarity of picked vs every card in the *filtered* pool
            sims_full = emb @ emb[picked_idx]
            sub_sims = sub.assign(_sim=sims_full[sub.index.values])
            neighbors = (
                sub_sims[sub_sims.index != picked_idx]
                .nlargest(HIGHLIGHT_K, "_sim")
            )

            # Neighbors trace, color-graded by similarity
            if len(neighbors):
                fig.add_trace(
                    go.Scatter(
                        x=neighbors["umap_x"],
                        y=neighbors["umap_y"],
                        mode="markers",
                        marker=dict(
                            size=10,
                            color=neighbors["_sim"],
                            colorscale="Viridis",
                            cmin=float(neighbors["_sim"].min()),
                            cmax=float(neighbors["_sim"].max()),
                            showscale=True,
                            colorbar=dict(
                                title=dict(text="Cosine similarity", side="top"),
                                orientation="h",
                                x=0.5, y=-0.08,
                                xanchor="center", yanchor="top",
                                len=0.45, thickness=12,
                            ),
                            line=dict(width=1.5, color="white"),
                        ),
                        customdata=neighbors[["name", "type", "cost", "_sim"]].values,
                        hovertemplate=(
                            "<b>%{customdata[0]}</b><br>"
                            "%{customdata[1]} · cost %{customdata[2]}<br>"
                            "sim=%{customdata[3]:.3f}<extra></extra>"
                        ),
                        name=f"Top {HIGHLIGHT_K} neighbors",
                        showlegend=False,
                    )
                )

            # Picked card on top, gold star, always visible even if filtered out
            fig.add_trace(
                go.Scatter(
                    x=[picked["umap_x"]],
                    y=[picked["umap_y"]],
                    mode="markers",
                    marker=dict(
                        symbol="star",
                        size=16,
                        color="gold",
                        line=dict(width=1.5, color="black"),
                    ),
                    hovertemplate=f"<b>{picked['name']}</b> (picked)<extra></extra>",
                    name="Picked card",
                    showlegend=False,
                )
            )

    fig.update_layout(
        legend=dict(
            title=dict(text="Character", font=dict(size=12, color="#475569")),
            font=dict(size=11, color="#334155"),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(title=None, showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(title=None, showgrid=False, zeroline=False, showticklabels=False),
        title=dict(font=dict(size=13, color="#475569"), x=0.01, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family='-apple-system, "SF Pro Text", system-ui, sans-serif', color="#334155"),
        # Bottom margin grows when a colorbar is present so the horizontal
        # colorbar has room without clipping the plot area.
        margin=dict(l=10, r=10, t=50, b=80 if highlight_name else 10),
    )
    return fig


def card_choices(game: str) -> list[str]:
    df, _ = load_game(game)
    return sorted(df["name"].dropna().unique().tolist())


import html as _html


def _empty_card_html() -> str:
    return (
        '<div class="arch-empty">'
        '<div class="arch-empty-eyebrow">No selection</div>'
        '<div class="arch-empty-title">Pick a card to see its neighbors</div>'
        '<div class="arch-empty-tip">'
        "Use the dropdown to choose any card. The map highlights the top-10 nearest "
        "neighbors in the full 1024-D embedding space, color-graded by cosine similarity."
        "</div>"
        "</div>"
    )


def _selected_card_html(row) -> str:
    name = _html.escape(str(row.get("name", "")))
    type_ = _html.escape(str(row.get("type", "")))
    rarity = _html.escape(str(row.get("rarity", "")))
    color = _html.escape(str(row.get("color", "")))
    cost = _html.escape(str(row.get("cost", "")))
    desc = _html.escape(str(row.get("description") or ""))
    cost_label = "X" if cost == "-1" else ("Unplayable" if cost == "-2" else (cost or "-"))
    desc_html = desc or '<span class="arch-card-nodesc">No description on file</span>'
    return f"""
<div class="arch-selected-card">
  <div class="arch-selected-eyebrow">Selected card</div>
  <div class="arch-selected-row">
    <div class="arch-selected-name">{name}</div>
    <div class="arch-selected-cost">cost {cost_label}</div>
  </div>
  <div class="arch-selected-meta">
    <span>{type_ or '-'}</span>
    <span class="arch-divider">·</span>
    <span>{rarity or '-'}</span>
    <span class="arch-divider">·</span>
    <span class="arch-meta-color">{color or '-'}</span>
  </div>
  <div class="arch-selected-desc">{desc_html}</div>
</div>
""".strip()


def render_neighbors(game: str, card_name: str | None) -> tuple[str, pd.DataFrame]:
    if not card_name:
        return _empty_card_html(), pd.DataFrame()
    df, emb = load_game(game)
    matches = df.index[df["name"] == card_name]
    if len(matches) == 0:
        return (
            f'<div class="arch-empty"><div class="arch-empty-title">No card named '
            f'{_html.escape(card_name)!r} in {game.upper()}.</div></div>',
            pd.DataFrame(),
        )
    idx = int(matches[0])
    selected = df.iloc[idx]
    nn = topk_similar(df, emb, emb[idx], k=6, exclude_idx=idx)
    return _selected_card_html(selected), nn


CUSTOM_CSS = """
/* ============================================================
   Archetype Map - slate accent on stone neutrals.
   Single accent. No decorative gradients. 4px spacing grid.
   ============================================================ */
:root {
  --arch-accent: #334155;          /* slate-700 */
  --arch-accent-soft: #f1f5f9;     /* slate-100 */
  --arch-accent-ring: rgba(51,65,85,0.18);
  --arch-fg: #1c1917;
  --arch-fg-muted: #475569;        /* slate-600 */
  --arch-fg-soft: #94a3b8;         /* slate-400 */
  --arch-surface: #ffffff;
  --arch-surface-2: #f8fafc;       /* slate-50 */
  --arch-border: #e2e8f0;          /* slate-200 */
  --arch-border-strong: #cbd5e1;   /* slate-300 */
}
.dark, .gradio-container.dark {
  --arch-accent: #cbd5e1;          /* slate-300 */
  --arch-accent-soft: rgba(203,213,225,0.10);
  --arch-accent-ring: rgba(203,213,225,0.28);
  --arch-fg: #f1f5f9;
  --arch-fg-muted: #94a3b8;
  --arch-fg-soft: #64748b;
  --arch-surface: #0f172a;         /* slate-900 */
  --arch-surface-2: #1e293b;       /* slate-800 */
  --arch-border: #1e293b;
  --arch-border-strong: #334155;
}

.gradio-container {
  max-width: 1240px !important;
  margin: 0 auto !important;
  font-feature-settings: "ss01", "cv11";
}

/* ---------- Hero ---------- */
.arch-hero {
  padding: 32px 4px 24px 4px;
  border-bottom: 1px solid var(--arch-border);
  margin-bottom: 20px;
}
.arch-hero-eyebrow {
  display: inline-block;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.10em;
  text-transform: uppercase;
  color: var(--arch-accent);
  margin-bottom: 10px;
}
.arch-hero h1 {
  margin: 0 0 6px 0;
  font-size: 30px;
  font-weight: 700;
  letter-spacing: -0.022em;
  color: var(--arch-fg);
  line-height: 1.15;
}
.arch-hero p {
  margin: 0 0 14px 0;
  color: var(--arch-fg-muted);
  font-size: 15px;
  line-height: 1.55;
  max-width: 72ch;
}
.arch-hero-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  font-size: 12.5px;
  color: var(--arch-fg-soft);
}
.arch-hero-meta a {
  color: var(--arch-fg-muted);
  text-decoration: none;
  border-bottom: 1px dotted var(--arch-border-strong);
  padding-bottom: 1px;
  transition: color 0.15s ease, border-color 0.15s ease;
}
.arch-hero-meta a:hover {
  color: var(--arch-accent);
  border-bottom-color: var(--arch-accent);
}
.arch-hero-meta-sep { color: var(--arch-border-strong); }

/* ---------- Tabs ---------- */
.arch-tabs button.selected,
.arch-tabs .selected {
  border-bottom-color: var(--arch-accent) !important;
  color: var(--arch-fg) !important;
}
.arch-tabs button {
  font-size: 13.5px !important;
  font-weight: 500 !important;
  letter-spacing: 0.005em;
  color: var(--arch-fg-muted) !important;
}

/* ---------- Section labels ---------- */
.arch-section-label {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.10em;
  text-transform: uppercase;
  color: var(--arch-fg-soft);
  margin: 16px 0 8px 2px;
}
.arch-section-label-first { margin-top: 4px; }

/* ---------- Sidebar filters ---------- */
.arch-sidebar { padding: 4px 12px 4px 0; }
.arch-sidebar .form,
.arch-sidebar fieldset {
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
}
.arch-sidebar label {
  font-size: 13px !important;
  color: var(--arch-fg) !important;
}
.arch-sidebar .wrap label {
  border: 1px solid var(--arch-border) !important;
  background: var(--arch-surface) !important;
  border-radius: 6px !important;
  padding: 5px 10px !important;
  margin: 2px 4px 2px 0 !important;
  font-size: 12px !important;
  font-weight: 500 !important;
  color: var(--arch-fg-muted) !important;
  transition: border-color 0.15s ease, color 0.15s ease, background 0.15s ease;
  min-height: 28px;
  display: inline-flex;
  align-items: center;
}
.arch-sidebar .wrap label:hover {
  border-color: var(--arch-border-strong) !important;
  color: var(--arch-fg) !important;
}
.arch-sidebar .wrap label:has(input:checked) {
  background: var(--arch-accent-soft) !important;
  border-color: var(--arch-accent) !important;
  color: var(--arch-accent) !important;
  font-weight: 600 !important;
}
.arch-sidebar input[type="checkbox"] { display: none; }

/* Card picker - visually heavier so it stands out as the primary action */
.arch-card-picker .wrap,
.arch-card-picker > div > div {
  background: var(--arch-accent-soft) !important;
  border: 1px solid var(--arch-accent) !important;
  border-radius: 8px !important;
  min-height: 40px !important;
  transition: box-shadow 0.15s ease;
}
.arch-card-picker:hover .wrap,
.arch-card-picker:focus-within .wrap {
  box-shadow: 0 0 0 4px var(--arch-accent-ring);
}
.arch-card-picker input,
.arch-card-picker .single-select {
  font-weight: 600 !important;
  color: var(--arch-accent) !important;
  font-size: 13.5px !important;
}

/* ---------- Plot wrapper ---------- */
.arch-plot {
  border: 1px solid var(--arch-border);
  border-radius: 12px;
  background: var(--arch-surface);
  padding: 12px 8px 4px 8px;
  overflow: hidden;
}

/* ---------- Selected-card panel ---------- */
.arch-selected-card,
.arch-empty {
  background: var(--arch-surface);
  border: 1px solid var(--arch-border);
  border-radius: 12px;
  padding: 18px 22px;
  margin-top: 16px;
}
.arch-empty {
  border-left: 3px solid var(--arch-accent);
}
.arch-empty-eyebrow,
.arch-selected-eyebrow {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.10em;
  text-transform: uppercase;
  color: var(--arch-accent);
  margin-bottom: 8px;
}
.arch-empty-title {
  font-size: 16px;
  font-weight: 600;
  letter-spacing: -0.005em;
  color: var(--arch-fg);
  margin-bottom: 4px;
}
.arch-empty-tip {
  font-size: 14px;
  color: var(--arch-fg-muted);
  line-height: 1.6;
  max-width: 64ch;
}
.arch-selected-row {
  display: flex; align-items: baseline; justify-content: space-between;
  gap: 14px;
}
.arch-selected-name {
  font-size: 18px;
  font-weight: 600;
  letter-spacing: -0.01em;
  color: var(--arch-fg);
}
.arch-selected-cost {
  font-size: 12px;
  font-variant-numeric: tabular-nums;
  color: var(--arch-fg-muted);
  background: var(--arch-surface-2);
  border: 1px solid var(--arch-border);
  border-radius: 999px;
  padding: 3px 11px;
  white-space: nowrap;
}
.arch-selected-meta {
  display: flex; gap: 6px; flex-wrap: wrap;
  margin: 6px 0 12px 0;
  font-size: 13px;
  color: var(--arch-fg-muted);
  text-transform: capitalize;
}
.arch-divider { color: var(--arch-border-strong); }
.arch-meta-color { color: var(--arch-fg); font-weight: 500; }
.arch-selected-desc {
  font-size: 14px;
  line-height: 1.55;
  padding-top: 10px;
  border-top: 1px solid var(--arch-border);
  color: var(--arch-fg);
}
.arch-card-nodesc {
  color: var(--arch-fg-soft);
  font-style: italic;
}

/* ---------- Neighbors table ---------- */
.arch-neighbors-wrap { margin-top: 18px; }
.arch-neighbors-wrap table {
  border: 1px solid var(--arch-border) !important;
  border-radius: 8px !important;
  overflow: hidden;
  font-size: 13px;
}
.arch-neighbors-wrap thead th {
  background: var(--arch-surface-2) !important;
  color: var(--arch-fg-muted) !important;
  font-weight: 600 !important;
  font-size: 11px !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase;
  padding: 10px 12px !important;
  border-bottom: 1px solid var(--arch-border) !important;
}
.arch-neighbors-wrap tbody td {
  padding: 10px 12px !important;
  border-top: 1px solid var(--arch-border) !important;
  color: var(--arch-fg);
}
.arch-neighbors-wrap tbody tr:hover td {
  background: var(--arch-surface-2) !important;
}

/* ---------- Footer ---------- */
.arch-footer {
  margin-top: 32px;
  padding-top: 16px;
  border-top: 1px solid var(--arch-border);
  font-size: 12px;
  color: var(--arch-fg-soft);
  line-height: 1.6;
}
.arch-footer a {
  color: var(--arch-fg-muted);
  text-decoration: none;
  border-bottom: 1px dotted var(--arch-border-strong);
  padding-bottom: 1px;
  transition: color 0.15s ease, border-color 0.15s ease;
}
.arch-footer a:hover {
  color: var(--arch-accent);
  border-bottom-color: var(--arch-accent);
}
"""


def make_demo() -> gr.Blocks:
    df1, _ = load_game("sts1")
    df2, _ = load_game("sts2")
    sts1_colors = sorted(df1["color"].dropna().unique().tolist())
    sts1_types = sorted(df1["type"].dropna().unique().tolist())
    sts2_colors = sorted(df2["color"].dropna().unique().tolist())
    sts2_types = sorted(df2["type"].dropna().unique().tolist())

    with gr.Blocks(title="Slay the Spire Archetype Map", css=CUSTOM_CSS) as demo:
        gr.HTML(
            '<div class="arch-hero">'
            '<span class="arch-hero-eyebrow">Embedding atlas</span>'
            '<h1>Archetype Map</h1>'
            '<p>Every card in <b>Slay the Spire 1 + 2</b> projected to 2D via UMAP '
            'over Qwen3-Embedding-0.6B vectors. Cards close together play similarly. '
            'Pick a card to see its nearest neighbors in the full 1024-D embedding '
            'space, not just on the 2D projection.</p>'
            '<div class="arch-hero-meta">'
            '<a href="https://huggingface.co/datasets/t22000t/slay-the-spire-1-cards">STS1 cards</a>'
            '<span class="arch-hero-meta-sep">·</span>'
            '<a href="https://huggingface.co/datasets/t22000t/slay-the-spire-1-card-embeddings">STS1 embeddings</a>'
            '<span class="arch-hero-meta-sep">·</span>'
            '<a href="https://huggingface.co/datasets/t22000t/slay-the-spire-2-cards">STS2 cards</a>'
            '<span class="arch-hero-meta-sep">·</span>'
            '<a href="https://huggingface.co/datasets/t22000t/slay-the-spire-2-card-embeddings">STS2 embeddings</a>'
            '<span class="arch-hero-meta-sep">·</span>'
            '<a href="https://huggingface.co/collections/t22000t/slaythespire-codex">collection</a>'
            '</div>'
            '</div>'
        )

        with gr.Tabs(elem_classes="arch-tabs"):
            for game, colors, types in (
                ("sts1", sts1_colors, sts1_types),
                ("sts2", sts2_colors, sts2_types),
            ):
                with gr.Tab(GAME_LABELS[game]):
                    with gr.Row():
                        with gr.Column(scale=1, min_width=220, elem_classes="arch-sidebar"):
                            gr.HTML('<div class="arch-section-label arch-section-label-first">Filter by class</div>')
                            color_filter = gr.CheckboxGroup(
                                choices=colors,
                                show_label=False,
                                value=colors,
                            )
                            gr.HTML('<div class="arch-section-label">Filter by type</div>')
                            type_filter = gr.CheckboxGroup(
                                choices=types,
                                show_label=False,
                                value=types,
                            )
                            gr.HTML('<div class="arch-section-label">Highlight neighbors</div>')
                            card_picker = gr.Dropdown(
                                choices=card_choices(game),
                                show_label=False,
                                value=None,
                                allow_custom_value=False,
                                container=False,
                                elem_classes="arch-card-picker",
                            )
                        with gr.Column(scale=4):
                            plot = gr.Plot(
                                value=render_scatter(game, colors, types),
                                show_label=False,
                                elem_classes="arch-plot",
                            )
                            detail_html = gr.HTML(_empty_card_html())
                            with gr.Column(elem_classes="arch-neighbors-wrap"):
                                gr.HTML('<div class="arch-section-label">Nearest neighbors</div>')
                                neighbors_table = gr.Dataframe(
                                    headers=["similarity", "name", "type", "rarity", "cost", "color", "description"],
                                    interactive=False,
                                    wrap=True,
                                    show_label=False,
                                )

                    def _on_filter_change(c, t, name, g=game):
                        return render_scatter(g, c, t, highlight_name=name)

                    def _on_pick_change(c, t, name, g=game):
                        fig = render_scatter(g, c, t, highlight_name=name)
                        html_, nn = render_neighbors(g, name)
                        return fig, html_, nn

                    color_filter.change(
                        fn=_on_filter_change,
                        inputs=[color_filter, type_filter, card_picker],
                        outputs=plot,
                    )
                    type_filter.change(
                        fn=_on_filter_change,
                        inputs=[color_filter, type_filter, card_picker],
                        outputs=plot,
                    )
                    card_picker.change(
                        fn=_on_pick_change,
                        inputs=[color_filter, type_filter, card_picker],
                        outputs=[plot, detail_html, neighbors_table],
                    )

        gr.HTML(
            '<div class="arch-footer">'
            'Built with <a href="https://github.com/timothy22000/slaythespire-codex">slaythespire-codex</a>. '
            'UMAP coordinates were precomputed and shipped with the embeddings dataset; '
            'the nearest-neighbor search runs on the full 1024-D embedding (cosine similarity).'
            '</div>'
        )

    return demo


demo = make_demo()

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
