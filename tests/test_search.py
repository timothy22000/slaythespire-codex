"""Tests for the similarity search functions.

Uses small synthetic embeddings so we don't need to load any model.
"""

import numpy as np
import pytest

from sts_cards.search import search_by_card, topk_similar


def test_topk_returns_correct_count(small_dataframe, small_emb_matrix):
    query = small_emb_matrix[0]
    out = topk_similar(small_dataframe, small_emb_matrix, query, k=3)
    assert len(out) == 3


def test_topk_excludes_self(small_dataframe, small_emb_matrix):
    """When excluding the query index, it should not appear in results."""
    out = topk_similar(small_dataframe, small_emb_matrix,
                       small_emb_matrix[0], k=3, exclude_idx=0)
    assert "Strike" not in out["name"].values


def test_topk_orders_by_similarity_desc(small_dataframe, small_emb_matrix):
    """Cosine similarities should be monotonically non-increasing."""
    query = small_emb_matrix[0]  # the Strike vector
    out = topk_similar(small_dataframe, small_emb_matrix, query, k=4)
    sims = out["similarity"].values
    assert all(sims[i] >= sims[i + 1] for i in range(len(sims) - 1))


def test_topk_finds_the_obvious_neighbor(small_dataframe, small_emb_matrix):
    """Strike's closest non-self neighbor should be Bash (same axis-aligned
    family in the synthetic geometry — see conftest)."""
    out = topk_similar(small_dataframe, small_emb_matrix,
                       small_emb_matrix[0], k=3, exclude_idx=0)
    assert out.iloc[0]["name"] == "Bash"


def test_topk_handles_k_larger_than_corpus(small_dataframe, small_emb_matrix):
    """k > n should just return n rows, not crash."""
    out = topk_similar(small_dataframe, small_emb_matrix,
                       small_emb_matrix[0], k=999)
    assert len(out) == len(small_dataframe)


def test_topk_returns_expected_columns(small_dataframe, small_emb_matrix):
    out = topk_similar(small_dataframe, small_emb_matrix,
                       small_emb_matrix[0], k=2)
    assert "similarity" in out.columns
    assert "name" in out.columns
    # And similarity is the leading column
    assert list(out.columns)[0] == "similarity"


def test_search_by_card_finds_by_name(small_dataframe, small_emb_matrix):
    out = search_by_card(small_dataframe, small_emb_matrix, "Strike", k=2)
    assert "Strike" not in out["name"].values  # excluded as query


def test_search_by_card_case_insensitive(small_dataframe, small_emb_matrix):
    # Mixed case should still work
    out = search_by_card(small_dataframe, small_emb_matrix, "strike", k=2)
    assert len(out) == 2


def test_search_by_card_unknown_raises(small_dataframe, small_emb_matrix):
    with pytest.raises(ValueError, match="No card named"):
        search_by_card(small_dataframe, small_emb_matrix, "Nonexistent")


def test_topk_similarity_is_dot_product(small_dataframe, small_emb_matrix):
    """Embeddings are unit-normalized, so dot product == cosine similarity.
    Pin this contract with a test."""
    query = small_emb_matrix[1]  # Defend
    out = topk_similar(small_dataframe, small_emb_matrix, query, k=4)
    # Compute expected dot products manually
    expected_self_sim = float(np.dot(query, small_emb_matrix[1]))  # 1.0
    self_row = out[out["name"] == "Defend"].iloc[0]
    assert abs(self_row["similarity"] - expected_self_sim) < 1e-3
