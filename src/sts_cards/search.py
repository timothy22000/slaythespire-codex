"""Nearest-neighbor search within a single game's embeddings.

Since each game is its own dataset, search operates on one Parquet at
a time. For cross-game lookups, load both files in your application
code — the embeddings share the same model+instruction so vectors are
directly comparable.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .provenance import DatasetProvenance

log = logging.getLogger(__name__)


def load_index(parquet_path: Path) -> tuple[pd.DataFrame, np.ndarray, DatasetProvenance | None]:
    """Load a per-game embeddings Parquet plus its provenance."""
    df = pd.read_parquet(parquet_path)
    if "embedding" not in df.columns:
        raise ValueError(f"{parquet_path} has no `embedding` column — run embed first")
    emb = np.vstack(df["embedding"].values).astype(np.float32)

    prov_path = parquet_path.parent / parquet_path.name.replace(
        "_cards_with_embeddings.parquet", "_provenance.json"
    )
    prov = DatasetProvenance.read(prov_path) if prov_path.exists() else None
    return df, emb, prov


def topk_similar(
    df: pd.DataFrame,
    emb: np.ndarray,
    query_vec: np.ndarray,
    k: int = 10,
    exclude_idx: int | None = None,
) -> pd.DataFrame:
    """Return the top-k most similar rows by cosine similarity.

    Embeddings are unit-normalized, so dot product == cosine similarity.
    """
    sims = emb @ query_vec
    if exclude_idx is not None:
        sims[exclude_idx] = -np.inf

    k = min(k, len(sims))
    top_idx = np.argpartition(-sims, k - 1)[:k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]

    columns = ["name", "type", "rarity", "cost", "description"]
    out = df.iloc[top_idx][columns].copy()
    out.insert(0, "similarity", sims[top_idx].round(4))
    return out.reset_index(drop=True)


def search_by_card(
    df: pd.DataFrame, emb: np.ndarray, name: str, k: int = 10,
) -> pd.DataFrame:
    """Find cards similar to a named card in the same index."""
    matches = df.index[df["name"].str.lower() == name.lower()]
    if len(matches) == 0:
        raise ValueError(f"No card named {name!r} in this index")
    idx = int(matches[0])
    return topk_similar(df, emb, emb[idx], k=k, exclude_idx=idx)


def search_by_text(
    df: pd.DataFrame, emb: np.ndarray, text: str, prov: DatasetProvenance | None,
    k: int = 10,
) -> pd.DataFrame:
    """Encode a free-form query with the same model+instruction the
    cards were embedded with, then search."""
    if prov is None or prov.embed is None:
        raise RuntimeError(
            "Cannot run text query without provenance — query model and "
            "instruction must match the indexed cards. Run embed first."
        )

    from sentence_transformers import SentenceTransformer
    log.info("Encoding query with %s", prov.embed.model_id)
    model = SentenceTransformer(prov.embed.model_id, trust_remote_code=True)

    encode_kwargs: dict = {
        "normalize_embeddings": True,
        "convert_to_numpy": True,
    }
    if prov.embed.task_instruction:
        encode_kwargs["prompt"] = f"Instruct: {prov.embed.task_instruction}\nText: "

    query_vec = model.encode([text], **encode_kwargs)[0]

    # Match Matryoshka truncation if applied at index time
    if prov.embed.matryoshka_dim and query_vec.shape[0] != emb.shape[1]:
        query_vec = query_vec[: prov.embed.matryoshka_dim]
        query_vec = query_vec / np.linalg.norm(query_vec)

    return topk_similar(df, emb, query_vec, k=k)
