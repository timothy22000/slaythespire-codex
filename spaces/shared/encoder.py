"""Encode user-supplied text into the same vector space as the published
slaythespire-codex embeddings.

The HF public Inference API does not serve `Qwen/Qwen3-Embedding-0.6B`
(verified 2026-05-07: 404). So this module loads the model locally via
`sentence_transformers`. The model + instruction prompt MUST match what
the indexed cards were encoded with, otherwise query vectors live in a
slightly different distribution and similarity scores degrade.

The published constants are:
    DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
    DEFAULT_TASK_INSTRUCTION = (
        "Represent this Slay the Spire card so that mechanically similar "
        "cards (same archetype, comparable damage/block patterns, related "
        "keywords) are close in embedding space."
    )

These are vendored here (not imported from sts_cards/) so the Space has
zero dependency on the parent package.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_TASK_INSTRUCTION = (
    "Represent this Slay the Spire card so that mechanically similar "
    "cards (same archetype, comparable damage/block patterns, related "
    "keywords) are close in embedding space."
)


@lru_cache(maxsize=1)
def _model():
    """Load the SentenceTransformer once per process. ~1.2 GB download
    on first call, cached locally afterward. Takes 30-60s cold."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(DEFAULT_MODEL, trust_remote_code=True)


def encode_query(text: str) -> np.ndarray:
    """Encode `text` into a unit-normalized 1024-D vector compatible with
    the published embeddings (i.e. dot-product equals cosine similarity
    against the indexed card matrix)."""
    if not text or not text.strip():
        raise ValueError("encode_query needs a non-empty string")

    vec = _model().encode(
        [text],
        prompt=f"Instruct: {DEFAULT_TASK_INSTRUCTION}\nText: ",
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0].astype(np.float32)
    return vec
