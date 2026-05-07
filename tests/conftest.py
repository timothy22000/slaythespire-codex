"""Shared pytest fixtures."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_sts1_card_payload() -> dict:
    """Realistic spire-archive payload for an STS1 card."""
    return {
        "id": "Strike_R",
        "name": "Strike",
        "type": "Attack",
        "rarity": "Basic",
        "color": "Red",
        "cost": 1,
        "description": "Deal 6 damage.",
        "description_upgraded": "Deal 9 damage.",
        "keywords": [],
    }


@pytest.fixture
def sample_sts2_card_payload() -> dict:
    """Realistic spire-archive payload for an STS2 card with mechanic
    tags STS1 doesn't have (Channel, Orb)."""
    return {
        "id": "Zap",
        "name": "Zap",
        "type": "Skill",
        "rarity": "Basic",
        "color": "Defect",
        "cost": 1,
        "description": "Channel 1 Lightning.",
        "description_upgraded": "Channel 2 Lightning.",
        "keywords": ["Channel"],
    }


@pytest.fixture
def small_dataframe() -> pd.DataFrame:
    """4-card frame with 4D pre-normalized embeddings — enough for
    deterministic top-k tests without loading any model."""
    rows = [
        {"name": "Strike", "type": "Attack", "rarity": "Basic", "cost": "1",
         "description": "Deal 6 damage.",
         "embedding": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)},
        {"name": "Defend", "type": "Skill", "rarity": "Basic", "cost": "1",
         "description": "Gain 5 Block.",
         "embedding": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)},
        {"name": "Bash", "type": "Attack", "rarity": "Basic", "cost": "2",
         "description": "Deal 8 damage. Apply 2 Vulnerable.",
         "embedding": np.array([0.9, 0.0, 0.4359, 0.0], dtype=np.float32)},
        {"name": "Iron Wave", "type": "Attack", "rarity": "Common", "cost": "1",
         "description": "Gain 5 Block. Deal 5 damage.",
         "embedding": np.array([0.7071, 0.7071, 0.0, 0.0], dtype=np.float32)},
    ]
    df = pd.DataFrame(rows)
    # Verify embeddings are unit-normalized (within tolerance)
    norms = np.linalg.norm(np.vstack(df["embedding"]), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), f"fixture embeddings not unit-norm: {norms}"
    return df


@pytest.fixture
def small_emb_matrix(small_dataframe) -> np.ndarray:
    return np.vstack(small_dataframe["embedding"].values).astype(np.float32)
