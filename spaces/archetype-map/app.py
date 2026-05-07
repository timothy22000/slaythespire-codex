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


def render_scatter(
    game: str,
    color_filter: list[str],
    type_filter: list[str],
) -> go.Figure:
    df, _ = load_game(game)
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

    fig = px.scatter(
        sub,
        x="umap_x",
        y="umap_y",
        color="color",
        custom_data=["name", "type", "rarity", "cost", "short_desc"],
        title=f"{GAME_LABELS[game]}: {len(sub)} of {len(df)} cards",
        height=600,
    )
    fig.update_traces(
        marker=dict(size=8, line=dict(width=0.5, color="white"), opacity=0.85),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "%{customdata[1]} · %{customdata[2]} · cost %{customdata[3]}<br>"
            "%{customdata[4]}<extra></extra>"
        ),
    )
    fig.update_layout(
        legend=dict(title="Character"),
        xaxis=dict(title=None, showgrid=False, zeroline=False),
        yaxis=dict(title=None, showgrid=False, zeroline=False),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def card_choices(game: str) -> list[str]:
    df, _ = load_game(game)
    return sorted(df["name"].dropna().unique().tolist())


def render_neighbors(game: str, card_name: str | None) -> tuple[str, pd.DataFrame]:
    if not card_name:
        return "", pd.DataFrame()
    df, emb = load_game(game)
    matches = df.index[df["name"] == card_name]
    if len(matches) == 0:
        return f"_No card named {card_name!r} in {game.upper()}._", pd.DataFrame()
    idx = int(matches[0])
    selected = df.iloc[idx]
    header = (
        f"### {selected['name']}\n"
        f"**{selected['type']} · {selected['rarity']} · "
        f"{selected['color']} · cost {selected['cost']}**\n\n"
        f"> {selected.get('description') or '_(no description)_'}"
    )
    nn = topk_similar(df, emb, emb[idx], k=6, exclude_idx=idx)
    return header, nn


def make_demo() -> gr.Blocks:
    df1, _ = load_game("sts1")
    df2, _ = load_game("sts2")
    sts1_colors = sorted(df1["color"].dropna().unique().tolist())
    sts1_types = sorted(df1["type"].dropna().unique().tolist())
    sts2_colors = sorted(df2["color"].dropna().unique().tolist())
    sts2_types = sorted(df2["type"].dropna().unique().tolist())

    with gr.Blocks(title="Slay the Spire Archetype Map") as demo:
        gr.Markdown(
            "# Slay the Spire Archetype Map\n"
            "Every card in **Slay the Spire 1 + 2** projected to 2D via UMAP "
            "over Qwen3-Embedding-0.6B vectors. Cards close together play "
            "similarly. Pick a card to see its nearest neighbors in the full "
            "1024-D embedding space (not just on the 2D projection).\n\n"
            "Data: "
            "[STS1 cards](https://huggingface.co/datasets/t22000t/slay-the-spire-1-cards) · "
            "[STS1 embeddings](https://huggingface.co/datasets/t22000t/slay-the-spire-1-card-embeddings) · "
            "[STS2 cards](https://huggingface.co/datasets/t22000t/slay-the-spire-2-cards) · "
            "[STS2 embeddings](https://huggingface.co/datasets/t22000t/slay-the-spire-2-card-embeddings)"
        )

        with gr.Tabs():
            for game, colors, types in (
                ("sts1", sts1_colors, sts1_types),
                ("sts2", sts2_colors, sts2_types),
            ):
                with gr.Tab(GAME_LABELS[game]):
                    with gr.Row():
                        with gr.Column(scale=1, min_width=200):
                            color_filter = gr.CheckboxGroup(
                                choices=colors,
                                label="Character / class",
                                value=colors,
                            )
                            type_filter = gr.CheckboxGroup(
                                choices=types,
                                label="Card type",
                                value=types,
                            )
                            card_picker = gr.Dropdown(
                                choices=card_choices(game),
                                label="Pick a card to see neighbors",
                                value=None,
                                allow_custom_value=False,
                            )
                        with gr.Column(scale=4):
                            plot = gr.Plot(
                                value=render_scatter(game, colors, types),
                            )
                            with gr.Accordion("Selected card + neighbors", open=True):
                                detail_md = gr.Markdown("_Pick a card from the dropdown._")
                                neighbors_table = gr.Dataframe(
                                    headers=["similarity", "name", "type", "rarity", "cost", "color", "description"],
                                    interactive=False,
                                    wrap=True,
                                )

                    color_filter.change(
                        fn=lambda c, t, g=game: render_scatter(g, c, t),
                        inputs=[color_filter, type_filter],
                        outputs=plot,
                    )
                    type_filter.change(
                        fn=lambda c, t, g=game: render_scatter(g, c, t),
                        inputs=[color_filter, type_filter],
                        outputs=plot,
                    )
                    card_picker.change(
                        fn=lambda name, g=game: render_neighbors(g, name),
                        inputs=card_picker,
                        outputs=[detail_md, neighbors_table],
                    )

        gr.Markdown(
            "---\n"
            "Built with [slaythespire-codex](https://github.com/timothy22000/slaythespire-codex). "
            "UMAP coordinates were precomputed and shipped with the embeddings dataset; "
            "the nearest-neighbor search runs on the full 1024-D embedding (cosine similarity)."
        )

    return demo


demo = make_demo()

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
