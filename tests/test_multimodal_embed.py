"""Tests for multimodal_embed.py.

The 2B model is never loaded — `SentenceTransformer` is patched to a
deterministic stub. We only verify the wiring: payload shape, image
preprocessing, output dim, unit-norm, has_image counting, and that
provenance picks up the right fields.
"""

from __future__ import annotations

import io
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from PIL import Image

# `sentence_transformers` is a heavy optional dep — stub the import so the
# module loads in environments that don't have it installed.
if "sentence_transformers" not in sys.modules:
    fake_st = types.ModuleType("sentence_transformers")
    class _StubST:
        def __init__(self, *a, **kw): pass
        def encode(self, *a, **kw): raise NotImplementedError
    fake_st.SentenceTransformer = _StubST  # type: ignore[attr-defined]
    sys.modules["sentence_transformers"] = fake_st


def _png_bytes(size=(500, 380)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGBA", size, (255, 0, 0, 255)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def fake_cards_with_art_parquet(tmp_path: Path) -> Path:
    """Mirror what extract-art produces: cards parquet with an image column."""
    out = tmp_path / "sts1_cards.parquet"
    pd.DataFrame([
        {"id": "STRIKE_R", "game": "sts1", "name": "Strike", "type": "Attack",
         "rarity": "Basic", "color": "Red", "cost": "1",
         "description": "Deal 6 damage.", "description_upgraded": "Deal 9 damage.",
         "keywords": "[]", "raw_json": "{}",
         "image": _png_bytes(), "image_resolution": "high"},
        {"id": "DEFEND_R", "game": "sts1", "name": "Defend", "type": "Skill",
         "rarity": "Basic", "color": "Red", "cost": "1",
         "description": "Gain 5 Block.", "description_upgraded": "Gain 8 Block.",
         "keywords": "[]", "raw_json": "{}",
         "image": _png_bytes(), "image_resolution": "high"},
        # Card without art — gets text-only encoding
        {"id": "IMPULSE", "game": "sts1", "name": "Impulse", "type": "Skill",
         "rarity": "Common", "color": "Blue", "cost": "1",
         "description": "If you have no other Skills, draw 2 cards.",
         "description_upgraded": "If you have no other Skills, draw 3 cards.",
         "keywords": "[]", "raw_json": "{}",
         "image": None, "image_resolution": None},
    ]).to_parquet(out, index=False)
    return out


@pytest.fixture
def fetch_provenance(fake_cards_with_art_parquet: Path) -> Path:
    """Drop a minimal fetch-only provenance.json next to the cards parquet
    so embed_multimodal_game has lineage to update."""
    from sts_cards.provenance import DatasetProvenance, FetchProvenance
    prov = DatasetProvenance(
        fetch=FetchProvenance(
            source="https://spire-archive.com/api/sts1/cards",
            source_fetched_at="2026-05-08T00:00:00+00:00",
            game="sts1", language="en", n_cards=3,
        ),
    )
    out = fake_cards_with_art_parquet.parent / "sts1_provenance.json"
    prov.write(out)
    return out


# --- prepare_image -----------------------------------------------------


def test_prepare_image_pads_to_512(tmp_path: Path):
    from sts_cards.multimodal_embed import PAD_SIZE, prepare_image
    img = prepare_image(_png_bytes((500, 380)))
    assert img is not None
    assert img.size == (PAD_SIZE, PAD_SIZE)
    assert img.mode == "RGB"  # alpha is dropped


def test_prepare_image_handles_square_input():
    from sts_cards.multimodal_embed import PAD_SIZE, prepare_image
    img = prepare_image(_png_bytes((1024, 1024)))
    assert img is not None
    assert img.size == (PAD_SIZE, PAD_SIZE)


def test_prepare_image_none_passes_through():
    from sts_cards.multimodal_embed import prepare_image
    assert prepare_image(None) is None


# --- embed_multimodal_game --------------------------------------------


class _FakeSentenceTransformer:
    """Returns deterministic float32 vectors — no model load."""
    def __init__(self, *args, **kwargs):
        self.dim = 2048

    def encode(self, payloads, **kwargs):
        # Each item is {"text": ...} or {"text": ..., "image": ...}
        rng = np.random.RandomState(0)
        out = rng.randn(len(payloads), self.dim).astype(np.float32)
        if kwargs.get("normalize_embeddings"):
            out = out / np.linalg.norm(out, axis=1, keepdims=True)
        return out


class _FakeTorch(types.SimpleNamespace):
    class cuda:  # noqa: N801 — mimic torch surface
        @staticmethod
        def is_available(): return False
    class backends:  # noqa: N801
        class mps:  # noqa: N801
            @staticmethod
            def is_available(): return False


@pytest.fixture
def patched_st_torch():
    """Patch the lazy imports inside embed_multimodal_game."""
    fake_st_mod = types.ModuleType("sentence_transformers")
    fake_st_mod.SentenceTransformer = _FakeSentenceTransformer  # type: ignore[attr-defined]
    fake_torch_mod = _FakeTorch()
    with patch.dict(sys.modules, {
        "sentence_transformers": fake_st_mod,
        "torch": fake_torch_mod,
    }):
        yield


def test_embed_multimodal_writes_parquet(
    fake_cards_with_art_parquet: Path,
    fetch_provenance: Path,
    patched_st_torch,
):
    from sts_cards.multimodal_embed import embed_multimodal_game

    out_path = embed_multimodal_game(
        fake_cards_with_art_parquet,
        fake_cards_with_art_parquet.parent,
        game="sts1",
        matryoshka_dim=1024,
        batch_size=2,
    )

    assert out_path.exists()
    df = pd.read_parquet(out_path)
    assert len(df) == 3
    assert {"id", "game", "name", "card_text", "has_image",
            "multimodal_embedding"}.issubset(df.columns)
    # Output dim is the requested Matryoshka dim
    vecs = np.vstack(df["multimodal_embedding"].values)
    assert vecs.shape == (3, 1024)
    # Unit-normalized after truncation
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    # has_image is True for cards with art, False for the one without
    assert df.loc[df["id"] == "STRIKE_R", "has_image"].iloc[0]
    assert df.loc[df["id"] == "DEFEND_R", "has_image"].iloc[0]
    assert not df.loc[df["id"] == "IMPULSE", "has_image"].iloc[0]


def test_embed_multimodal_updates_provenance(
    fake_cards_with_art_parquet: Path,
    fetch_provenance: Path,
    patched_st_torch,
):
    from sts_cards.multimodal_embed import embed_multimodal_game
    from sts_cards.provenance import DatasetProvenance

    embed_multimodal_game(
        fake_cards_with_art_parquet,
        fake_cards_with_art_parquet.parent,
        game="sts1",
        matryoshka_dim=1024,
    )

    prov = DatasetProvenance.read(fetch_provenance)
    assert prov.multimodal_embed is not None
    assert prov.multimodal_embed.embedding_dim == 1024
    assert prov.multimodal_embed.n_with_image == 2
    assert prov.multimodal_embed.n_without_image == 1
    assert prov.multimodal_embed.image_preprocessing == "rgb-resize-pad-512x512-grey"


def test_embed_multimodal_warns_when_no_image_column(
    tmp_path: Path,
    patched_st_torch,
    caplog,
):
    """A cards parquet without an image column still works — every row
    falls back to text-only encoding and a warning is logged."""
    from sts_cards.multimodal_embed import embed_multimodal_game
    from sts_cards.provenance import DatasetProvenance, FetchProvenance

    cards = tmp_path / "sts1_cards.parquet"
    pd.DataFrame([
        {"id": "STRIKE_R", "game": "sts1", "name": "Strike",
         "type": "Attack", "rarity": "Basic", "color": "Red", "cost": "1",
         "description": "Deal 6 damage.", "description_upgraded": "",
         "keywords": "[]", "raw_json": "{}"},
    ]).to_parquet(cards, index=False)
    DatasetProvenance(
        fetch=FetchProvenance(source="x", source_fetched_at="t",
                              game="sts1", language="en", n_cards=1),
    ).write(tmp_path / "sts1_provenance.json")

    import logging
    with caplog.at_level(logging.WARNING, logger="sts_cards.multimodal_embed"):
        embed_multimodal_game(cards, tmp_path, game="sts1", matryoshka_dim=1024)
    assert any("no `image` column" in rec.message for rec in caplog.records)

    prov = DatasetProvenance.read(tmp_path / "sts1_provenance.json")
    assert prov.multimodal_embed is not None
    assert prov.multimodal_embed.n_with_image == 0
    assert prov.multimodal_embed.n_without_image == 1
