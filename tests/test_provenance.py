"""Tests for the provenance module."""

import json
from pathlib import Path

import pytest

from sts_cards.provenance import (
    ArtProvenance,
    DatasetProvenance,
    EmbedProvenance,
    FetchProvenance,
    MultimodalEmbedProvenance,
    now_iso,
)


def test_now_iso_is_iso_format():
    s = now_iso()
    # Should parse without error
    from datetime import datetime
    datetime.fromisoformat(s)


def test_fetch_only_provenance(tmp_path):
    prov = DatasetProvenance(
        fetch=FetchProvenance(
            source="https://spire-archive.com/api/sts1/cards",
            source_fetched_at=now_iso(),
            game="sts1",
            language="en",
            n_cards=361,
        ),
    )
    out = tmp_path / "p.json"
    prov.write(out)

    loaded = DatasetProvenance.read(out)
    assert loaded.fetch.game == "sts1"
    assert loaded.fetch.n_cards == 361
    assert loaded.embed is None


def test_full_provenance_roundtrip(tmp_path):
    prov = DatasetProvenance(
        fetch=FetchProvenance(
            source="https://spire-archive.com/api/sts2/cards",
            source_fetched_at="2026-05-01T00:00:00+00:00",
            game="sts2",
            language="en",
            n_cards=577,
            sts_game_version="v0.103.0",
        ),
        embed=EmbedProvenance(
            model_id="Qwen/Qwen3-Embedding-0.6B",
            embedding_dim=1024,
            task_instruction="Represent this card.",
            embedded_at="2026-05-02T00:00:00+00:00",
            matryoshka_dim=512,
            sentence_transformers_version="3.0.1",
        ),
    )
    out = tmp_path / "p.json"
    prov.write(out)

    loaded = DatasetProvenance.read(out)
    assert loaded.fetch.sts_game_version == "v0.103.0"
    assert loaded.embed is not None
    assert loaded.embed.matryoshka_dim == 512
    assert loaded.embed.model_id == "Qwen/Qwen3-Embedding-0.6B"


def test_provenance_writes_pretty_json(tmp_path):
    """The file should be human-readable in a diff."""
    prov = DatasetProvenance(
        fetch=FetchProvenance(
            source="x", source_fetched_at="t", game="sts1",
            language="en", n_cards=1,
        ),
    )
    out = tmp_path / "p.json"
    prov.write(out)
    text = out.read_text()
    assert "\n" in text
    assert "  " in text  # indentation


def test_provenance_includes_environment_metadata():
    prov = DatasetProvenance(
        fetch=FetchProvenance(source="x", source_fetched_at="t",
                              game="sts1", language="en", n_cards=1),
    )
    d = prov.to_dict()
    # Environment metadata is auto-populated for reproducibility
    assert d["package_version"]
    assert d["python_version"]
    assert d["platform"]
    # Art is opt-in; absent until extract-art runs
    assert d["art"] is None


def test_art_provenance_roundtrip(tmp_path):
    prov = DatasetProvenance(
        fetch=FetchProvenance(source="x", source_fetched_at="t",
                              game="sts1", language="en", n_cards=10),
        art=ArtProvenance(
            extraction_source="jar",
            source_file_sha256="a" * 64,
            extracted_at="2026-05-08T00:00:00+00:00",
            n_art_files=9,
            n_cards_total=10,
            resolution="high",
            image_dimensions=(1024, 1024),
        ),
    )
    out = tmp_path / "p.json"
    prov.write(out)

    loaded = DatasetProvenance.read(out)
    assert loaded.art is not None
    assert loaded.art.extraction_source == "jar"
    assert loaded.art.resolution == "high"
    # Tuple is preserved across the JSON round-trip
    assert loaded.art.image_dimensions == (1024, 1024)
    assert loaded.art.gdre_tools_version is None


def test_multimodal_embed_provenance_roundtrip(tmp_path):
    prov = DatasetProvenance(
        fetch=FetchProvenance(source="x", source_fetched_at="t",
                              game="sts1", language="en", n_cards=360),
        embed=EmbedProvenance(
            model_id="Qwen/Qwen3-Embedding-0.6B",
            embedding_dim=1024,
            task_instruction="text instruction",
            embedded_at="2026-05-08T00:00:00+00:00",
        ),
        multimodal_embed=MultimodalEmbedProvenance(
            model_id="Qwen/Qwen3-VL-Embedding-2B",
            embedding_dim=1024,
            task_instruction="multimodal instruction",
            embedded_at="2026-05-08T01:00:00+00:00",
            image_preprocessing="rgb-resize-pad-512x512-grey",
            n_with_image=359,
            n_without_image=1,
            matryoshka_dim=1024,
            sentence_transformers_version="3.0.1",
        ),
    )
    out = tmp_path / "p.json"
    prov.write(out)

    loaded = DatasetProvenance.read(out)
    assert loaded.embed is not None
    assert loaded.multimodal_embed is not None
    # Both blocks coexist independently
    assert loaded.embed.model_id == "Qwen/Qwen3-Embedding-0.6B"
    assert loaded.multimodal_embed.model_id == "Qwen/Qwen3-VL-Embedding-2B"
    assert loaded.multimodal_embed.image_preprocessing == "rgb-resize-pad-512x512-grey"
    assert loaded.multimodal_embed.n_with_image == 359
    assert loaded.multimodal_embed.n_without_image == 1


def test_multimodal_embed_provenance_alone(tmp_path):
    """A run that only produced multimodal embeddings (no text embed step)
    serializes with `embed=None`."""
    prov = DatasetProvenance(
        fetch=FetchProvenance(source="x", source_fetched_at="t",
                              game="sts1", language="en", n_cards=10),
        multimodal_embed=MultimodalEmbedProvenance(
            model_id="Qwen/Qwen3-VL-Embedding-2B",
            embedding_dim=1024,
            task_instruction="t",
            embedded_at="2026-05-08T00:00:00+00:00",
            image_preprocessing="rgb-resize-pad-512x512-grey",
            n_with_image=10,
            n_without_image=0,
        ),
    )
    out = tmp_path / "p.json"
    prov.write(out)
    loaded = DatasetProvenance.read(out)
    assert loaded.embed is None
    assert loaded.multimodal_embed is not None


def test_art_provenance_with_gdre_version(tmp_path):
    prov = DatasetProvenance(
        fetch=FetchProvenance(source="x", source_fetched_at="t",
                              game="sts2", language="en", n_cards=20,
                              sts_game_version="v0.103.0"),
        art=ArtProvenance(
            extraction_source="pck",
            source_file_sha256="b" * 64,
            extracted_at="2026-05-08T00:00:00+00:00",
            n_art_files=18,
            n_cards_total=20,
            resolution="high",
            image_dimensions=None,
            gdre_tools_version="GDRE Tools 0.7.0",
        ),
    )
    out = tmp_path / "p.json"
    prov.write(out)

    loaded = DatasetProvenance.read(out)
    assert loaded.art is not None
    assert loaded.art.gdre_tools_version == "GDRE Tools 0.7.0"
    assert loaded.art.image_dimensions is None
