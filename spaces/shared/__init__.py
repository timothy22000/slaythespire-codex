"""Shared utilities used across the slaythespire-codex Spaces.

Vendored into each Space at deploy time (HF Space repos can't reach
sibling subdirs in the GitHub repo).
"""

from .data import REPOS, load_game, topk_similar
from .encoder import DEFAULT_MODEL, DEFAULT_TASK_INSTRUCTION, encode_query
from .normalize import MECHANICS_FIELDS, build_card_document, normalize_card_name_in_text
from .outliers import baselines, check_outliers

__all__ = [
    "REPOS",
    "load_game",
    "topk_similar",
    "DEFAULT_MODEL",
    "DEFAULT_TASK_INSTRUCTION",
    "encode_query",
    "MECHANICS_FIELDS",
    "build_card_document",
    "normalize_card_name_in_text",
    "baselines",
    "check_outliers",
]
