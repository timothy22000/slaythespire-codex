"""Tests for upload file routing.

These don't make real HF calls — they just verify that `files_for(...)`
selects the right files for each (game, kind) pair.
"""

import pytest

from sts_cards.upload import files_for


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")


def test_cards_kind_picks_cards_files(tmp_path):
    _touch(tmp_path / "sts1_cards.parquet")
    _touch(tmp_path / "sts1_provenance.json")
    _touch(tmp_path / "sts1_cards_croissant.json")
    _touch(tmp_path / "sts1_embeddings.parquet")  # should NOT be picked
    _touch(tmp_path / "sts1_embeddings_croissant.json")  # should NOT be picked

    pairs = files_for("sts1", "cards", tmp_path)
    in_repo_names = {p[1] for p in pairs}

    assert in_repo_names == {"cards.parquet", "provenance.json", "croissant.json"}
    # Critically: the embeddings file is excluded
    src_names = {p[0].name for p in pairs}
    assert "sts1_embeddings.parquet" not in src_names


def test_embeddings_kind_picks_embeddings_files(tmp_path):
    _touch(tmp_path / "sts1_embeddings.parquet")
    _touch(tmp_path / "sts1_provenance.json")
    _touch(tmp_path / "sts1_embeddings_croissant.json")
    _touch(tmp_path / "sts1_cards.parquet")  # should NOT be picked
    _touch(tmp_path / "sts1_cards_croissant.json")  # should NOT be picked

    pairs = files_for("sts1", "embeddings", tmp_path)
    in_repo_names = {p[1] for p in pairs}

    assert in_repo_names == {"embeddings.parquet", "provenance.json", "croissant.json"}
    src_names = {p[0].name for p in pairs}
    assert "sts1_cards.parquet" not in src_names


def test_repo_filenames_strip_game_prefix(tmp_path):
    """Files inside each HF repo are named generically (cards.parquet, not
    sts1_cards.parquet) since the repo itself is game-specific."""
    _touch(tmp_path / "sts1_cards.parquet")
    _touch(tmp_path / "sts2_cards.parquet")

    pairs1 = files_for("sts1", "cards", tmp_path)
    pairs2 = files_for("sts2", "cards", tmp_path)

    # Both upload to "cards.parquet" inside their respective repos
    assert all(p[1] == "cards.parquet" for p in pairs1)
    assert all(p[1] == "cards.parquet" for p in pairs2)


def test_unknown_kind_raises(tmp_path):
    with pytest.raises(ValueError, match="unknown kind"):
        files_for("sts1", "garbage", tmp_path)


def test_multimodal_embeddings_kind_picks_right_files(tmp_path):
    _touch(tmp_path / "sts1_multimodal_embeddings.parquet")
    _touch(tmp_path / "sts1_provenance.json")
    _touch(tmp_path / "sts1_multimodal-embeddings_croissant.json")
    # Files that should NOT be picked up
    _touch(tmp_path / "sts1_cards.parquet")
    _touch(tmp_path / "sts1_embeddings.parquet")
    _touch(tmp_path / "sts1_embeddings_croissant.json")

    pairs = files_for("sts1", "multimodal-embeddings", tmp_path)
    in_repo_names = {p[1] for p in pairs}
    # Inside the repo, the parquet is named `embeddings.parquet` — same
    # surface as the text-embeddings repo so consumer code is portable.
    assert in_repo_names == {"embeddings.parquet", "provenance.json", "croissant.json"}

    src_names = {p[0].name for p in pairs}
    # The text-embeddings parquet must NOT be picked
    assert "sts1_embeddings.parquet" not in src_names
    assert "sts1_cards.parquet" not in src_names
    assert "sts1_multimodal_embeddings.parquet" in src_names
    assert "sts1_multimodal-embeddings_croissant.json" in src_names


def test_missing_files_silently_skipped(tmp_path):
    # Only provenance exists; cards parquet doesn't
    _touch(tmp_path / "sts1_provenance.json")

    pairs = files_for("sts1", "cards", tmp_path)
    # Should return just the one file that exists, no error
    assert len(pairs) == 1
    assert pairs[0][1] == "provenance.json"


def test_both_games_independent(tmp_path):
    """Files for sts1 should not include sts2 files even with both present."""
    _touch(tmp_path / "sts1_cards.parquet")
    _touch(tmp_path / "sts2_cards.parquet")

    pairs1 = files_for("sts1", "cards", tmp_path)
    src_names = {p[0].name for p in pairs1}
    assert "sts1_cards.parquet" in src_names
    assert "sts2_cards.parquet" not in src_names
