"""Load the four published HuggingFace datasets and merge cards onto embeddings.

Uses huggingface_hub.hf_hub_download to fetch parquets directly, bypassing
the `datasets` library's hashing path (which references
`transformers.PreTrainedTokenizerBase` via lazy lookup; that lookup fails
on HF Spaces when transformers is in sys.modules even at known-good versions).

Files cache to ~/.cache/huggingface/hub; subsequent calls hit disk.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

REPOS = {
    "sts1": (
        "t22000t/slay-the-spire-1-cards",
        "t22000t/slay-the-spire-1-card-embeddings",
    ),
    "sts2": (
        "t22000t/slay-the-spire-2-cards",
        "t22000t/slay-the-spire-2-card-embeddings",
    ),
}


@lru_cache(maxsize=2)
def load_game(game: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (cards_df_with_umap, embeddings_matrix) for one game.

    cards_df_with_umap has all 22 metadata columns plus umap_x, umap_y, embedding.
    embeddings_matrix is shape (n_cards, 1024), float32, unit-normalized.
    """
    if game not in REPOS:
        raise ValueError(f"game must be one of {list(REPOS)}, got {game!r}")

    cards_repo, emb_repo = REPOS[game]
    cards_path = hf_hub_download(
        repo_id=cards_repo, filename="cards.parquet", repo_type="dataset",
    )
    embs_path = hf_hub_download(
        repo_id=emb_repo, filename="embeddings.parquet", repo_type="dataset",
    )

    cards = pd.read_parquet(cards_path)
    embs = pd.read_parquet(embs_path)

    df = cards.merge(
        embs[["id", "embedding", "umap_x", "umap_y"]],
        on="id",
        how="inner",
    )
    emb = np.vstack(df["embedding"].values).astype(np.float32)
    return df, emb


def topk_similar(
    df: pd.DataFrame,
    emb: np.ndarray,
    query_vec: np.ndarray,
    k: int = 10,
    exclude_idx: int | None = None,
) -> pd.DataFrame:
    """Top-k cosine-similar rows. Vectors must be unit-normalized.

    Vendored from src/sts_cards/search.py.
    """
    sims = emb @ query_vec
    if exclude_idx is not None:
        sims[exclude_idx] = -np.inf

    k = min(k, len(sims))
    top_idx = np.argpartition(-sims, k - 1)[:k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]

    columns = [c for c in ("name", "type", "rarity", "cost", "color", "description") if c in df.columns]
    out = df.iloc[top_idx][columns].copy()
    out.insert(0, "similarity", sims[top_idx].round(4))
    return out.reset_index(drop=True)
