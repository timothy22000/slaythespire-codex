"""Tests for the provenance module."""

import json
from pathlib import Path

import pytest

from sts_cards.provenance import (
    DatasetProvenance,
    EmbedProvenance,
    FetchProvenance,
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
