"""UMAP 2D projection for one game's embeddings.

Each game gets its own coordinate system because their mechanics
differ — STS2 has Orbs/Forge/Souls that don't exist in STS1.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def project_2d(
    in_path: Path,
    *,
    out_html: Path | None = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    seed: int = 42,
) -> Path:
    """Add umap_x, umap_y columns to a per-game embedding Parquet and
    optionally write an interactive HTML scatter plot.
    """
    import umap

    df = pd.read_parquet(in_path)
    log.info("UMAP on %d cards (%s)", len(df), in_path)

    n_neighbors = min(n_neighbors, max(2, len(df) - 1))
    emb = np.vstack(df["embedding"].values).astype(np.float32)

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric="cosine",
        random_state=seed,
    )
    coords = reducer.fit_transform(emb)
    df["umap_x"] = coords[:, 0]
    df["umap_y"] = coords[:, 1]
    df.to_parquet(in_path, index=False)
    log.info("Updated %s with umap_x/umap_y", in_path)

    # Mirror the UMAP coords into the slim embeddings file so the embeddings
    # HF repo carries them too.
    slim_path = in_path.parent / in_path.name.replace(
        "_cards_with_embeddings.parquet", "_embeddings.parquet"
    )
    if slim_path.exists():
        slim = pd.read_parquet(slim_path)
        # Join on id — both files come from the same source dataframe, so
        # ordering is identical, but we use a real merge for safety.
        slim = slim.merge(df[["id", "umap_x", "umap_y"]], on="id", how="left")
        slim.to_parquet(slim_path, index=False)
        log.info("Updated %s with umap_x/umap_y", slim_path)

    if out_html is not None:
        import plotly.express as px
        fig = px.scatter(
            df, x="umap_x", y="umap_y", color="color",
            hover_data=["name", "type", "rarity", "cost", "description"],
            title=f"{in_path.stem} — UMAP 2D",
            width=1200, height=800, opacity=0.8,
        )
        fig.update_traces(marker=dict(size=9, line=dict(width=0.5, color="white")))
        out_html.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(out_html)
        log.info("Plot → %s", out_html)

    return in_path
