"""Joint text+image embeddings for cards via Qwen3-VL-Embedding-2B.

Distinct from `embed.py` (text-only via Qwen3-Embedding-0.6B). The two
encoders run on different cadences and have different deployment
profiles (CPU-friendly vs GPU-leaning), so they get separate modules
and ship to separate HuggingFace repos.

Cards without art still go through this same model — Qwen3-VL accepts
text-only inputs natively, which preserves the joint coordinate system
across rows.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import (
    DEFAULT_MULTIMODAL_DIM,
    DEFAULT_MULTIMODAL_MODEL,
    DEFAULT_MULTIMODAL_TASK_INSTRUCTION,
)
from .normalize import build_card_document
from .provenance import (
    DatasetProvenance,
    FetchProvenance,
    MultimodalEmbedProvenance,
    get_st_version,
    now_iso,
)

log = logging.getLogger(__name__)

# Image preprocessing recipe identifier — recorded in provenance verbatim
# so a re-run can reproduce vectors bit-for-bit.
IMAGE_PREPROCESSING = "rgb-resize-pad-512x512-grey"
PAD_SIZE = 512
PAD_COLOR = (128, 128, 128)  # neutral grey


def prepare_image(png_bytes: bytes | None):
    """Decode raw PNG bytes to a 512×512 RGB PIL image.

    Pads (rather than center-crops) on a neutral grey background — STS
    card portraits carry identifying detail at the edges (weapons,
    character silhouettes), and cropping would lose them. Returns None
    when the input is None so the caller can pass through to a
    text-only encode.
    """
    if png_bytes is None:
        return None
    from PIL import Image

    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    w, h = img.size
    scale = PAD_SIZE / max(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (PAD_SIZE, PAD_SIZE), PAD_COLOR)
    canvas.paste(resized, ((PAD_SIZE - new_w) // 2, (PAD_SIZE - new_h) // 2))
    return canvas


def _encode_batch(model, payloads: list[dict], batch_size: int, prompt: str | None):
    """Wrap `SentenceTransformer.encode` so the call site doesn't have
    to know whether the wrapper takes `{text, image}` dicts or kwargs."""
    encode_kwargs: dict = {
        "batch_size": batch_size,
        "show_progress_bar": True,
        "convert_to_numpy": True,
        "normalize_embeddings": True,
    }
    if prompt:
        encode_kwargs["prompt"] = prompt
    return model.encode(payloads, **encode_kwargs)


def embed_multimodal_game(
    in_path: Path,
    out_dir: Path,
    *,
    game: str,
    model_id: str = DEFAULT_MULTIMODAL_MODEL,
    task_instruction: str = DEFAULT_MULTIMODAL_TASK_INSTRUCTION,
    matryoshka_dim: int | None = DEFAULT_MULTIMODAL_DIM,
    batch_size: int = 4,
    device: str | None = None,
) -> Path:
    """Read `{game}_cards.parquet`, encode each row's text+image
    payload via Qwen3-VL-Embedding, write the slim
    `{game}_multimodal_embeddings.parquet` plus updated provenance.

    Cards without an `image` column or with `image=None` go through
    text-only via the same model — preserves the joint vector space.
    """
    from sentence_transformers import SentenceTransformer
    import torch

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    log.info("Multimodal embedding %s | model=%s | device=%s", game, model_id, device)

    df = pd.read_parquet(in_path)
    log.info("Loaded %d cards from %s", len(df), in_path)

    has_image_col = "image" in df.columns
    if not has_image_col:
        log.warning(
            "%s has no `image` column — every row will go through the "
            "text-only path. Run `sts-cards extract-art %s` first if you "
            "want portraits embedded too.",
            in_path.name, game,
        )

    documents: list[str] = []
    payloads: list[dict] = []
    has_image_flags: list[bool] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="payloads"):
        doc = build_card_document(row)
        img_bytes = row["image"] if has_image_col else None
        img = prepare_image(img_bytes)
        documents.append(doc)
        has_image_flags.append(img is not None)
        payloads.append({"text": doc, "image": img} if img is not None else {"text": doc})

    n_with_image = int(sum(has_image_flags))
    n_without_image = len(has_image_flags) - n_with_image
    log.info("Encoding %d payloads (%d with image, %d text-only)",
             len(payloads), n_with_image, n_without_image)

    model = SentenceTransformer(model_id, device=device, trust_remote_code=True)
    prompt = f"Instruct: {task_instruction}\nText: " if task_instruction else None
    embeddings = _encode_batch(model, payloads, batch_size=batch_size, prompt=prompt)
    log.info("Embeddings shape: %s", embeddings.shape)

    if matryoshka_dim and matryoshka_dim < embeddings.shape[1]:
        log.info("Truncating to %dD (Matryoshka)", matryoshka_dim)
        embeddings = embeddings[:, :matryoshka_dim]
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    slim = pd.DataFrame({
        "id": df["id"].values,
        "game": df["game"].values if "game" in df.columns else [game] * len(df),
        "name": df["name"].values,
        "card_text": documents,
        "has_image": has_image_flags,
        "multimodal_embedding": list(embeddings.astype(np.float32)),
    })
    out_path = out_dir / f"{game}_multimodal_embeddings.parquet"
    slim.to_parquet(out_path, index=False)
    out_size_mb = out_path.stat().st_size / 1e6
    log.info("Wrote multimodal embeddings → %s (%.2f MB)", out_path, out_size_mb)

    fetch_prov_path = in_path.parent / f"{game}_provenance.json"
    if fetch_prov_path.exists():
        prov = DatasetProvenance.read(fetch_prov_path)
    else:
        log.warning("No fetch provenance at %s — multimodal lineage will be incomplete",
                    fetch_prov_path)
        prov = DatasetProvenance(
            fetch=FetchProvenance(
                source="unknown", source_fetched_at="unknown",
                game=game, language="unknown", n_cards=len(df),
            ),
        )

    prov.multimodal_embed = MultimodalEmbedProvenance(
        model_id=model_id,
        embedding_dim=int(embeddings.shape[1]),
        task_instruction=task_instruction,
        embedded_at=now_iso(),
        image_preprocessing=IMAGE_PREPROCESSING,
        n_with_image=n_with_image,
        n_without_image=n_without_image,
        matryoshka_dim=matryoshka_dim,
        sentence_transformers_version=get_st_version(),
    )
    prov.write(out_dir / f"{game}_provenance.json")

    return out_path
