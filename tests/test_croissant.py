"""Tests for Croissant JSON-LD generation.

Writes a small fixture Parquet, generates Croissant, and asserts that
the result is valid JSON-LD with the expected shape.
"""

import json

import pandas as pd
import pytest

from sts_cards.croissant import build_croissant, write_croissant


@pytest.fixture
def sample_cards_parquet(tmp_path):
    """Write a tiny STS1 cards Parquet to a temp dir."""
    df = pd.DataFrame([
        {
            "game": "sts1", "id": "Strike_R", "name": "Strike", "type": "Attack",
            "rarity": "Basic", "color": "Red", "cost": "1",
            "description": "Deal 6 damage.",
            "description_upgraded": "Deal 9 damage.",
            "keywords": "[]",
            "raw_json": "{}",
            "damage": 6,
            "damage_upgraded": 9,
            "block": None,
            "block_upgraded": None,
            "targets_all_enemies": False,
            "status_effects_applied": "[]",
            "mechanics": "[]",
            "orbs_channeled": "[]",
            "orbs_referenced": "[]",
            "forge_value": None,
            "souls_added": None,
        },
    ])
    out = tmp_path / "sts1_cards.parquet"
    df.to_parquet(out, index=False)
    return out


def test_build_croissant_returns_valid_jsonld(sample_cards_parquet):
    cr = build_croissant(
        sample_cards_parquet, game="sts1", kind="cards",
        repo_id="user/slay-the-spire-1-cards",
    )

    # Required top-level Croissant fields
    assert cr["@type"] == "sc:Dataset"
    assert "@context" in cr
    assert cr["name"]
    assert cr["description"]
    assert cr["license"]
    assert cr["url"] == "https://huggingface.co/datasets/user/slay-the-spire-1-cards"


def test_croissant_records_all_columns(sample_cards_parquet):
    cr = build_croissant(
        sample_cards_parquet, game="sts1", kind="cards",
        repo_id="user/slay-the-spire-1-cards",
    )
    fields = cr["recordSet"][0]["field"]
    field_names = {f["name"] for f in fields}

    # Every column in the Parquet should have a Croissant field
    df = pd.read_parquet(sample_cards_parquet)
    assert field_names == set(df.columns)


def test_croissant_field_types(sample_cards_parquet):
    cr = build_croissant(
        sample_cards_parquet, game="sts1", kind="cards",
        repo_id="user/slay-the-spire-1-cards",
    )
    fields = {f["name"]: f for f in cr["recordSet"][0]["field"]}

    # Spot-check type mappings
    assert fields["damage"]["dataType"] == "sc:Integer"
    assert fields["targets_all_enemies"]["dataType"] == "sc:Boolean"
    assert fields["name"]["dataType"] == "sc:Text"


def test_croissant_field_types_with_nulls(tmp_path):
    """Nullable int columns (with actual nulls) become float in pandas."""
    df = pd.DataFrame([
        {"game": "sts1", "name": "A", "damage": 6},
        {"game": "sts1", "name": "B", "damage": None},
    ])
    out = tmp_path / "sts1_cards.parquet"
    df.to_parquet(out, index=False)

    cr = build_croissant(out, game="sts1", kind="cards",
                        repo_id="user/slay-the-spire-1-cards")
    fields = {f["name"]: f for f in cr["recordSet"][0]["field"]}
    # When pandas has to accommodate null, the dtype becomes float64
    assert fields["damage"]["dataType"] == "sc:Float"


def test_croissant_marks_sts2_as_live_dataset(sample_cards_parquet, tmp_path):
    # STS2 should set isLiveDataset=true (Early Access drift)
    cr_sts2 = build_croissant(
        sample_cards_parquet, game="sts2", kind="cards",
        repo_id="user/slay-the-spire-2-cards",
    )
    assert cr_sts2["isLiveDataset"] is True

    cr_sts1 = build_croissant(
        sample_cards_parquet, game="sts1", kind="cards",
        repo_id="user/slay-the-spire-1-cards",
    )
    assert cr_sts1["isLiveDataset"] is False


def test_write_croissant_produces_valid_json(sample_cards_parquet, tmp_path):
    out = write_croissant(
        sample_cards_parquet, game="sts1", kind="cards",
        repo_id="user/slay-the-spire-1-cards",
    )
    # File exists and is valid JSON
    assert out.exists()
    parsed = json.loads(out.read_text())
    assert parsed["@type"] == "sc:Dataset"


def test_croissant_includes_code_repository(sample_cards_parquet):
    cr = build_croissant(
        sample_cards_parquet, game="sts1", kind="cards",
        repo_id="user/slay-the-spire-1-cards",
    )
    assert cr["isBasedOn"]["@type"] == "sc:SoftwareSourceCode"
    assert cr["isBasedOn"]["codeRepository"] == (
        "https://github.com/timothy22000/slaythespire-codex"
    )

    # Override is honored
    cr2 = build_croissant(
        sample_cards_parquet, game="sts1", kind="cards",
        repo_id="user/slay-the-spire-1-cards",
        code_repository="https://github.com/other/fork",
    )
    assert cr2["isBasedOn"]["codeRepository"] == "https://github.com/other/fork"


def test_croissant_includes_field_descriptions(sample_cards_parquet):
    cr = build_croissant(
        sample_cards_parquet, game="sts1", kind="cards",
        repo_id="user/slay-the-spire-1-cards",
    )
    for f in cr["recordSet"][0]["field"]:
        assert f.get("description"), f"field {f['name']} missing description"


# --- New tests for the embeddings kind ---


@pytest.fixture
def sample_embeddings_parquet(tmp_path):
    """Slim embeddings Parquet — just id, game, name, card_text, embedding."""
    import numpy as np
    df = pd.DataFrame([
        {
            "id": "Strike_R", "game": "sts1", "name": "Strike",
            "card_text": '{"name": "Strike"}',
            "embedding": np.array([0.1] * 1024, dtype=np.float32),
        },
    ])
    out = tmp_path / "sts1_embeddings.parquet"
    df.to_parquet(out, index=False)
    return out


def test_build_croissant_embeddings_kind(sample_embeddings_parquet):
    cr = build_croissant(
        sample_embeddings_parquet, game="sts1", kind="embeddings",
        repo_id="user/slay-the-spire-1-card-embeddings",
    )
    assert "Card Embeddings" in cr["name"]
    assert "embeddings" in cr["keywords"]
    assert cr["recordSet"][0]["@id"] == "embeddings"
    # Fields should include `embedding` and `id` (the join key)
    field_names = {f["name"] for f in cr["recordSet"][0]["field"]}
    assert "id" in field_names
    assert "embedding" in field_names


def test_build_croissant_rejects_bad_kind(sample_cards_parquet):
    with pytest.raises(ValueError, match="kind must be"):
        build_croissant(
            sample_cards_parquet, game="sts1", kind="invalid",
            repo_id="x/y",
        )


def test_write_croissant_filename_includes_kind(sample_cards_parquet):
    out_cards = write_croissant(
        sample_cards_parquet, game="sts1", kind="cards",
        repo_id="user/x",
    )
    # Default output path should have `cards` in the name so cards and
    # embeddings descriptors don't collide
    assert "cards" in out_cards.name
