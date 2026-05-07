"""Generate Croissant JSON-LD metadata for a per-game dataset.

[Croissant](https://docs.mlcommons.org/croissant/) is the MLCommons
metadata format adopted by HuggingFace and Google Dataset Search.
HuggingFace auto-generates a Croissant descriptor from any dataset
repo, but shipping one explicitly:

  - documents the schema with type information ML tools understand
  - threads through provenance + license + citation in a structured way
  - improves discoverability via Google Dataset Search
  - costs ~1KB per dataset and one CLI command

The output is written to `{game}_croissant.json` next to the Parquets.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from . import __version__
from .provenance import DatasetProvenance

log = logging.getLogger(__name__)


# Map pandas/Arrow dtypes to Croissant data types.
# https://docs.mlcommons.org/croissant/docs/croissant-spec.html#data-types
_DTYPE_TO_CROISSANT = {
    "int64": "sc:Integer",
    "Int64": "sc:Integer",
    "int32": "sc:Integer",
    "float64": "sc:Float",
    "float32": "sc:Float",
    "bool": "sc:Boolean",
    "boolean": "sc:Boolean",
    "object": "sc:Text",
    "string": "sc:Text",
}


def _croissant_type(dtype_str: str) -> str:
    """Best-effort mapping from pandas dtype to a Croissant type."""
    return _DTYPE_TO_CROISSANT.get(dtype_str, "sc:Text")


def _build_field(name: str, dtype: str, description: str) -> dict[str, Any]:
    return {
        "@type": "cr:Field",
        "@id": f"field/{name}",
        "name": name,
        "description": description,
        "dataType": _croissant_type(dtype),
        "source": {
            "fileObject": {"@id": "cards-parquet"},
            "extract": {"column": name},
        },
    }


# Human-readable descriptions for the columns we ship. Anything not in
# this dict gets a generic description.
_FIELD_DESCRIPTIONS: dict[str, str] = {
    "game": "Source game: 'sts1' or 'sts2'.",
    "id": "Stable card identifier from the game files.",
    "name": "Display name of the card.",
    "type": "One of Attack, Skill, Power, Status, Curse.",
    "rarity": "Basic, Common, Uncommon, Rare, Special, Curse, Status.",
    "color": "Character/class identifier (Red/Green/Blue/Purple/Colorless for STS1; "
             "Ironclad/Silent/Defect/Necrobinder/Regent/Colorless for STS2).",
    "cost": "Energy cost as a string ('0', '1', 'X', '-' for unplayable).",
    "description": "Card text in its base (pre-upgrade) form.",
    "description_upgraded": "Card text after the upgrade (+) is applied.",
    "keywords": "JSON-encoded list of declared keywords (Exhaust, Innate, etc.).",
    "raw_json": "Full original API payload, preserved for fields not in the normalized schema.",
    "card_text": "Canonical prettified-JSON card document fed to the embedding model.",
    "embedding": "Unit-normalized text embedding vector.",
    "umap_x": "First UMAP-2D coordinate of the embedding.",
    "umap_y": "Second UMAP-2D coordinate of the embedding.",
    # Derived features
    "damage": "Primary damage value extracted from the description (e.g. 'Deal 6 damage' → 6).",
    "damage_upgraded": "Same as `damage` but for the upgraded form.",
    "block": "Primary block value extracted from the description.",
    "block_upgraded": "Same as `block` but for the upgraded form.",
    "targets_all_enemies": "True if the card targets ALL enemies (AOE).",
    "status_effects_applied": "JSON list of {effect, count} for Vulnerable/Weak/Frail/etc.",
    "mechanics": "JSON list of mechanic keywords detected (declared + inferred from text).",
    "orbs_channeled": "STS2: JSON list of {type, count} for Orbs the card channels.",
    "orbs_referenced": "STS2: JSON list of Orb types mentioned in the description.",
    "forge_value": "STS2: Forge value if the card has a 'Forge N' clause.",
    "souls_added": "STS2: number of Souls the card adds, if any.",
}


def build_croissant(
    parquet_path: Path,
    *,
    game: str,
    kind: str,
    repo_id: str,
    license_url: str = "https://creativecommons.org/licenses/by/4.0/",
) -> dict[str, Any]:
    """Build a Croissant JSON-LD descriptor for one (game, kind) dataset.

    `kind` is "cards" or "embeddings".
    """
    if kind not in ("cards", "embeddings"):
        raise ValueError(f"kind must be 'cards' or 'embeddings', got {kind!r}")

    df = pd.read_parquet(parquet_path)

    fields = [
        _build_field(
            col,
            str(df[col].dtype),
            _FIELD_DESCRIPTIONS.get(col, f"Column `{col}` from the {game} {kind} dataset."),
        )
        for col in df.columns
    ]

    # Pull provenance from the sidecar if present
    prov_path = parquet_path.parent / f"{game}_provenance.json"
    citation = (
        f"sts-cards v{__version__}, fetched from spire-archive.com. "
        f"Game IP © Mega Crit."
    )
    version = __version__
    if prov_path.exists():
        prov = DatasetProvenance.read(prov_path)
        version = (
            f"{__version__}+fetched_{prov.fetch.source_fetched_at}"
            + (f"+game_{prov.fetch.sts_game_version}"
               if prov.fetch.sts_game_version else "")
        )

    pretty_game = "Slay the Spire 1" if game == "sts1" else "Slay the Spire 2"
    if kind == "cards":
        pretty_name = f"{pretty_game} — Cards"
        description = (
            f"Normalized cards from {pretty_game} with derived feature columns "
            "(damage, block, status effects, mechanics, and STS2 Orb/Forge/Soul "
            "extraction). Joinable to the embeddings dataset by `id`."
        )
        record_set_name = "cards"
        record_set_description = f"One row per card in {pretty_game}."
        file_id = "cards-parquet"
    else:
        pretty_name = f"{pretty_game} — Card Embeddings"
        description = (
            f"1024-D text embeddings for every card in {pretty_game}, produced by "
            "Qwen/Qwen3-Embedding-0.6B with task-instruction prompting. "
            "Unit-normalized — dot product equals cosine similarity. Joinable "
            "to the cards dataset by `id`."
        )
        record_set_name = "embeddings"
        record_set_description = (
            f"One embedding vector per card in {pretty_game}."
        )
        file_id = "embeddings-parquet"

    return {
        "@context": {
            "@language": "en",
            "@vocab": "https://schema.org/",
            "sc": "https://schema.org/",
            "cr": "http://mlcommons.org/croissant/",
            "data": {"@id": "cr:data", "@type": "@json"},
            "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
            "extract": "cr:extract",
            "field": "cr:field",
            "fileObject": "cr:fileObject",
            "fileSet": "cr:fileSet",
            "format": "cr:format",
            "includes": "cr:includes",
            "isLiveDataset": "cr:isLiveDataset",
            "key": "cr:key",
            "md5": "cr:md5",
            "parentField": "cr:parentField",
            "path": "cr:path",
            "recordSet": "cr:recordSet",
            "references": "cr:references",
            "regex": "cr:regex",
            "repeated": "cr:repeated",
            "replace": "cr:replace",
            "separator": "cr:separator",
            "source": "cr:source",
            "subField": "cr:subField",
            "transform": "cr:transform",
        },
        "@type": "sc:Dataset",
        "name": pretty_name,
        "description": description,
        "license": license_url,
        "url": f"https://huggingface.co/datasets/{repo_id}",
        "version": version,
        "citeAs": citation,
        "isLiveDataset": game == "sts2",
        "keywords": [
            "slay-the-spire", "card-game",
            "deckbuilder", "roguelike",
        ] + (["embeddings", "sentence-transformers"] if kind == "embeddings" else []),
        "distribution": [
            {
                "@type": "sc:FileObject",
                "@id": file_id,
                "name": parquet_path.name,
                "description": f"{record_set_description}",
                "contentUrl": parquet_path.name,
                "encodingFormat": "application/vnd.apache.parquet",
                "sha256": "<computed at upload time>",
            }
        ],
        "recordSet": [
            {
                "@type": "cr:RecordSet",
                "@id": record_set_name,
                "name": record_set_name,
                "description": record_set_description,
                "field": [
                    # Update field source references to use the new file_id
                    {**f, "source": {"fileObject": {"@id": file_id},
                                     "extract": {"column": f["name"]}}}
                    for f in fields
                ],
            }
        ],
    }


def write_croissant(
    parquet_path: Path,
    *,
    game: str,
    kind: str,
    repo_id: str,
    out_path: Path | None = None,
) -> Path:
    """Write a Croissant descriptor for one (game, kind) dataset.

    `kind` is "cards" or "embeddings". The descriptor's name, description,
    and isLiveDataset flag adapt accordingly. Default output path is
    {parquet_dir}/{game}_{kind}_croissant.json so cards and embeddings
    descriptors don't collide.
    """
    if kind not in ("cards", "embeddings"):
        raise ValueError(f"kind must be 'cards' or 'embeddings', got {kind!r}")

    croissant = build_croissant(parquet_path, game=game, kind=kind, repo_id=repo_id)
    if out_path is None:
        out_path = parquet_path.parent / f"{game}_{kind}_croissant.json"
    out_path.write_text(json.dumps(croissant, indent=2, ensure_ascii=False))
    log.info("Wrote Croissant (%s) → %s (%d fields)",
             kind, out_path, len(croissant["recordSet"][0]["field"]))
    return out_path
