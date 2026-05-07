"""Tests for the normalization functions.

These are pure functions, no I/O, no model. Fast and deterministic.
"""

import json

import pytest

from sts_cards.normalize import (
    MECHANICS_FIELDS,
    build_card_document,
    normalize_card,
    normalize_card_name_in_text,
)


class TestNormalizeCardName:
    def test_replaces_self_reference(self):
        out = normalize_card_name_in_text("Strike", "Strike deals 6 damage.")
        assert out == "~ deals 6 damage."

    def test_case_insensitive(self):
        out = normalize_card_name_in_text("Strike", "Play STRIKE to deal damage.")
        assert "STRIKE" not in out
        assert "~" in out

    def test_multiple_occurrences(self):
        out = normalize_card_name_in_text("Wound", "Wound. Wound is unplayable.")
        assert out.count("~") == 2

    def test_handles_special_characters_in_name(self):
        # Card names sometimes contain regex metacharacters
        out = normalize_card_name_in_text("A+B", "A+B is a card.")
        assert out == "~ is a card."

    def test_empty_text_returns_empty(self):
        assert normalize_card_name_in_text("Strike", "") == ""
        assert normalize_card_name_in_text("Strike", None) == ""

    def test_empty_name_passthrough(self):
        assert normalize_card_name_in_text("", "Some text.") == "Some text."
        assert normalize_card_name_in_text(None, "Some text.") == "Some text."

    def test_no_match_unchanged(self):
        assert normalize_card_name_in_text("Strike", "Gain 5 block.") == "Gain 5 block."


class TestNormalizeCard:
    def test_sts1_basic(self, sample_sts1_card_payload):
        row = normalize_card(sample_sts1_card_payload, game="sts1")
        assert row["game"] == "sts1"
        assert row["id"] == "Strike_R"
        assert row["name"] == "Strike"
        assert row["cost"] == "1"  # always stringified
        assert row["description"] == "Deal 6 damage."
        # raw_json must round-trip cleanly
        assert json.loads(row["raw_json"])["id"] == "Strike_R"

    def test_sts2_includes_keywords(self, sample_sts2_card_payload):
        row = normalize_card(sample_sts2_card_payload, game="sts2")
        keywords = json.loads(row["keywords"])
        assert keywords == ["Channel"]

    def test_handles_missing_fields_gracefully(self):
        # spire-archive may use a different field name on some cards
        out = normalize_card({"id": "X", "title": "Mystery"}, game="sts1")
        assert out["name"] == "Mystery"  # falls back to "title"
        assert out["description"] == ""

    def test_alternate_field_names(self):
        # color vs character vs class — all should map to `color`
        out = normalize_card({"id": "X", "name": "Y", "character": "Defect"},
                             game="sts2")
        assert out["color"] == "Defect"

    def test_cost_x_or_unplayable(self):
        out = normalize_card({"id": "X", "name": "Y", "cost": "X"}, game="sts1")
        assert out["cost"] == "X"


class TestBuildCardDocument:
    def test_produces_valid_json(self, sample_sts1_card_payload):
        row = normalize_card(sample_sts1_card_payload, game="sts1")
        doc = build_card_document(row)
        # Must round-trip through JSON
        parsed = json.loads(doc)
        assert parsed["name"] == "Strike"
        assert parsed["description"] == "Deal 6 damage."

    def test_indentation_is_present(self, sample_sts1_card_payload):
        # The minimaxir recipe relies on indentation — verify it's there
        row = normalize_card(sample_sts1_card_payload, game="sts1")
        doc = build_card_document(row)
        assert "\n  " in doc, "document should be pretty-printed JSON"

    def test_skips_empty_fields(self):
        # Cards with no upgraded text shouldn't include empty strings
        row = normalize_card({"id": "X", "name": "Tiny", "cost": 1,
                              "description": "Something."}, game="sts1")
        doc = build_card_document(row)
        parsed = json.loads(doc)
        assert "description_upgraded" not in parsed

    def test_replaces_card_name_in_description(self):
        row = normalize_card(
            {"id": "Z", "name": "Zap", "type": "Skill", "cost": 1,
             "description": "Zap an enemy."},
            game="sts2",
        )
        doc = build_card_document(row)
        parsed = json.loads(doc)
        assert "Zap" not in parsed["description"]
        assert "~" in parsed["description"]

    def test_keywords_parsed_from_json_string(self):
        # keywords arrive as a JSON-string list (post-normalize_card)
        row = normalize_card(
            {"id": "Z", "name": "Zap", "cost": 1, "description": "x",
             "keywords": ["Channel", "Innate"]},
            game="sts2",
        )
        doc = build_card_document(row)
        parsed = json.loads(doc)
        assert parsed["keywords"] == ["Channel", "Innate"]

    def test_field_order_in_mechanics(self):
        # MECHANICS_FIELDS defines a stable order — important for embedding
        # consistency. Make sure it includes the core fields.
        for required in ("name", "type", "cost", "description", "keywords"):
            assert required in MECHANICS_FIELDS
