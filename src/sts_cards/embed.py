"""Embed a single game's cards. One game = one Parquet output."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import DEFAULT_MODEL, DEFAULT_TASK_INSTRUCTION
from .normalize import build_card_document
from .provenance import DatasetProvenance, EmbedProvenance, get_st_version, now_iso

log = logging.getLogger(__name__)


def embed_game(
    in_path: Path,
    out_dir: Path,
    *,
    game: str,
    model_id: str = DEFAULT_MODEL,
    task_instruction: str = DEFAULT_TASK_INSTRUCTION,
    matryoshka_dim: int | None = None,
    batch_size: int = 16,
    device: str | None = None,
) -> Path:
    """Embed one game's cards. Reads `{game}_cards.parquet`, writes
    `{game}_cards_with_embeddings.parquet` plus updated provenance."""
    # Lazy imports — these are heavy and only needed at embed time
    from sentence_transformers import SentenceTransformer
    import torch

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    log.info("Embedding %s | model=%s | device=%s", game, model_id, device)

    df = pd.read_parquet(in_path)
    log.info("Loaded %d cards from %s", len(df), in_path)

    # Build canonical card documents
    documents = [
        build_card_document(row)
        for _, row in tqdm(df.iterrows(), total=len(df), desc="documents")
    ]
    df["card_text"] = documents

    # Encode
    model = SentenceTransformer(model_id, device=device, trust_remote_code=True)
    encode_kwargs: dict = {
        "batch_size": batch_size,
        "show_progress_bar": True,
        "convert_to_numpy": True,
        "normalize_embeddings": True,  # unit-norm → dot product = cosine similarity
    }
    if task_instruction:
        encode_kwargs["prompt"] = f"Instruct: {task_instruction}\nText: "

    embeddings = model.encode(documents, **encode_kwargs)
    log.info("Embeddings shape: %s", embeddings.shape)

    # Optional Matryoshka truncation
    if matryoshka_dim and matryoshka_dim < embeddings.shape[1]:
        log.info("Truncating to %dD (Matryoshka)", matryoshka_dim)
        embeddings = embeddings[:, :matryoshka_dim]
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    df["embedding"] = list(embeddings.astype(np.float32))

    # Write outputs:
    #   - {game}_cards_with_embeddings.parquet : full join (used by search.py
    #     and visualize.py for in-process work — keeps the existing internal
    #     contract intact)
    #   - {game}_embeddings.parquet : SLIM file for the embeddings HF repo —
    #     just the join key + the embedding-specific columns. Consumers join
    #     to the cards dataset by `id`.
    out_dir.mkdir(parents=True, exist_ok=True)

    full_path = out_dir / f"{game}_cards_with_embeddings.parquet"
    # The full file is for in-process use by search/visualize; image bytes
    # are big and unhelpful here, drop them if extract-art ran earlier.
    full_df = df.drop(columns=[c for c in ("image", "image_resolution") if c in df.columns])
    full_df.to_parquet(full_path, index=False)
    full_size_mb = full_path.stat().st_size / 1e6
    log.info("Wrote %d × %dD → %s (%.2f MB) [internal use]",
             len(full_df), embeddings.shape[1], full_path, full_size_mb)

    slim_cols = ["id", "game", "name", "card_text", "embedding"]
    slim_path = out_dir / f"{game}_embeddings.parquet"
    df[slim_cols].to_parquet(slim_path, index=False)
    slim_size_mb = slim_path.stat().st_size / 1e6
    log.info("Wrote slim embeddings → %s (%.2f MB) [for HF embeddings repo]",
             slim_path, slim_size_mb)

    out_path = full_path  # back-compat for the return value

    # Update provenance: load existing fetch info, add embed info
    fetch_prov_path = in_path.parent / f"{game}_provenance.json"
    if fetch_prov_path.exists():
        prov = DatasetProvenance.read(fetch_prov_path)
    else:
        log.warning("No fetch provenance found at %s — embedding will have"
                    " incomplete lineage", fetch_prov_path)
        from .provenance import FetchProvenance
        prov = DatasetProvenance(
            fetch=FetchProvenance(
                source="unknown",
                source_fetched_at="unknown",
                game=game,
                language="unknown",
                n_cards=len(df),
            ),
        )

    prov.embed = EmbedProvenance(
        model_id=model_id,
        embedding_dim=int(embeddings.shape[1]),
        task_instruction=task_instruction,
        embedded_at=now_iso(),
        matryoshka_dim=matryoshka_dim,
        sentence_transformers_version=get_st_version(),
    )
    prov.write(out_dir / f"{game}_provenance.json")

    return out_path
