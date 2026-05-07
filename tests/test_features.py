"""Tests for feature extraction.

Uses realistic snippets of card description text. No model loading,
no network — these are pure regex tests.
"""

import json

import pytest

from sts_cards.features import (
    ORB_TYPES,
    extract_block,
    extract_damage,
    extract_features,
    extract_forge_value,
    extract_mechanics,
    extract_orbs_channeled,
    extract_orbs_referenced,
    extract_souls_added,
    extract_status_effects_applied,
    targets_all_enemies,
)


class TestDamageBlock:
    def test_basic_damage(self):
        assert extract_damage("Deal 6 damage.") == 6
        assert extract_damage("Deal 12 damage.") == 12

    def test_damage_in_compound_text(self):
        # "Deal 8 damage. Apply 2 Vulnerable."
        assert extract_damage("Deal 8 damage. Apply 2 Vulnerable.") == 8

    def test_damage_case_insensitive(self):
        assert extract_damage("DEAL 5 DAMAGE.") == 5

    def test_no_damage(self):
        assert extract_damage("Gain 5 Block.") is None
        assert extract_damage(None) is None
        assert extract_damage("") is None

    def test_basic_block(self):
        assert extract_block("Gain 5 Block.") == 5

    def test_block_in_compound_text(self):
        assert extract_block("Gain 5 Block. Deal 5 damage.") == 5

    def test_no_block_when_only_damage(self):
        assert extract_block("Deal 6 damage.") is None


class TestAOE:
    def test_aoe_detected(self):
        assert targets_all_enemies("Deal 9 damage to ALL enemies.") is True

    def test_single_target_not_aoe(self):
        assert targets_all_enemies("Deal 6 damage.") is False

    def test_aoe_must_be_uppercase_all(self):
        # The game text consistently uses ALL in caps for AOE
        assert targets_all_enemies("Deal damage to all enemies.") is False


class TestStatusEffects:
    def test_apply_vulnerable(self):
        out = extract_status_effects_applied("Deal 8 damage. Apply 2 Vulnerable.")
        assert out == [{"effect": "Vulnerable", "count": 2}]

    def test_gain_strength(self):
        out = extract_status_effects_applied("Gain 2 Strength.")
        assert out == [{"effect": "Strength", "count": 2}]

    def test_multiple_effects(self):
        out = extract_status_effects_applied("Apply 2 Weak. Apply 2 Vulnerable.")
        assert {"effect": "Weak", "count": 2} in out
        assert {"effect": "Vulnerable", "count": 2} in out

    def test_no_effects(self):
        assert extract_status_effects_applied("Deal 6 damage.") == []


class TestMechanics:
    def test_declared_keywords_preserved(self):
        out = extract_mechanics("Some text", declared_keywords=["Exhaust"])
        assert "Exhaust" in out

    def test_inferred_from_text(self):
        out = extract_mechanics("Innate. Deal 11 damage. Exhaust.", [])
        assert "Innate" in out
        assert "Exhaust" in out

    def test_dedupes_declared_and_inferred(self):
        out = extract_mechanics("Exhaust.", declared_keywords=["Exhaust"])
        assert out.count("Exhaust") == 1

    def test_preserves_declared_order_first(self):
        # Declared keywords should appear before inferred ones
        out = extract_mechanics(
            "Innate. Exhaust.",
            declared_keywords=["Eternal"],
        )
        assert out[0] == "Eternal"


class TestSTS2OrbExtraction:
    def test_channel_lightning(self):
        out = extract_orbs_channeled("Channel 1 Lightning.")
        assert out == [{"type": "Lightning", "count": 1}]

    def test_channel_multiple_orbs(self):
        out = extract_orbs_channeled("Channel 2 Frost. Channel 1 Lightning.")
        assert {"type": "Frost", "count": 2} in out
        assert {"type": "Lightning", "count": 1} in out

    def test_orbs_referenced_without_channel(self):
        # 'if you have Frost' — references but doesn't channel
        out = extract_orbs_referenced("At the end of your turn, if you have Frost, deal 6 damage.")
        assert "Frost" in out

    def test_orbs_referenced_includes_channeled(self):
        out = extract_orbs_referenced("Channel 1 Lightning.")
        assert "Lightning" in out

    def test_unknown_orb_ignored(self):
        out = extract_orbs_channeled("Channel 1 Unicorn.")
        assert out == []

    def test_all_known_orb_types_match(self):
        for orb in ORB_TYPES:
            out = extract_orbs_channeled(f"Channel 1 {orb}.")
            assert out == [{"type": orb, "count": 1}], f"failed for {orb}"


class TestSTS2ForgeAndSouls:
    def test_forge(self):
        assert extract_forge_value("Deal 5 damage. Forge 5.") == 5

    def test_no_forge(self):
        assert extract_forge_value("Deal 5 damage.") is None

    def test_souls(self):
        assert extract_souls_added("Add 3 Souls into your Draw Pile.") == 3

    def test_souls_singular(self):
        assert extract_souls_added("Add 1 Soul into your Draw Pile.") == 1


class TestExtractFeaturesSTS1:
    """End-to-end feature extraction for an STS1 card."""

    def test_strike(self):
        card = {"description": "Deal 6 damage.", "description_upgraded": "Deal 9 damage."}
        feats = extract_features(card, game="sts1")
        assert feats["damage"] == 6
        assert feats["damage_upgraded"] == 9
        assert feats["block"] is None
        # STS2-only fields are present but empty
        assert feats["orbs_channeled"] == []
        assert feats["forge_value"] is None

    def test_bash(self):
        card = {
            "description": "Deal 8 damage. Apply 2 Vulnerable.",
            "description_upgraded": "Deal 10 damage. Apply 3 Vulnerable.",
            "keywords": [],
        }
        feats = extract_features(card, game="sts1")
        assert feats["damage"] == 8
        assert feats["status_effects_applied"] == [{"effect": "Vulnerable", "count": 2}]


class TestExtractFeaturesSTS2:
    """End-to-end feature extraction for an STS2 card."""

    def test_zap(self):
        card = {
            "description": "Channel 1 Lightning.",
            "description_upgraded": "Channel 2 Lightning.",
            "keywords": ["Channel"],
        }
        feats = extract_features(card, game="sts2")
        assert feats["orbs_channeled"] == [{"type": "Lightning", "count": 1}]
        assert "Lightning" in feats["orbs_referenced"]
        assert "Channel" in feats["mechanics"]

    def test_card_with_forge(self):
        card = {
            "description": "Deal 5 damage. Forge 5.",
            "description_upgraded": "",
            "keywords": [],
        }
        feats = extract_features(card, game="sts2")
        assert feats["damage"] == 5
        assert feats["forge_value"] == 5

    def test_aoe_dark_orb(self):
        card = {
            "description": "Channel 1 Dark. Trigger the passive ability of all Dark Orbs.",
            "description_upgraded": "",
            "keywords": [],
        }
        feats = extract_features(card, game="sts2")
        assert feats["orbs_channeled"] == [{"type": "Dark", "count": 1}]
        assert "Dark" in feats["orbs_referenced"]
