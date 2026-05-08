"""Provenance tracking for fetch + embed runs.

Every dataset snapshot carries a `provenance.json` next to its Parquet
documenting exactly how it was produced: source URL and fetch date, the
spire-archive snapshot reference, the embedding model and parameters,
plus host environment. Consumers can audit "is this the same data I
used last week?" without guesswork, and STS2 patch drift is traceable
because each rebuild bumps the timestamp and game version.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__


def now_iso() -> str:
    """Current UTC time as ISO 8601, second precision."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class FetchProvenance:
    source: str  # URL of the data source
    source_fetched_at: str  # ISO timestamp
    game: str  # "sts1" or "sts2"
    language: str
    n_cards: int
    spire_archive_snapshot: str | None = None  # commit/tag if known
    sts_game_version: str | None = None  # e.g. "v0.102.0" for STS2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EmbedProvenance:
    model_id: str
    embedding_dim: int
    task_instruction: str
    embedded_at: str
    matryoshka_dim: int | None = None
    model_revision: str | None = None
    sentence_transformers_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MultimodalEmbedProvenance:
    """Records how joint text+image embeddings were produced.

    Distinct from `EmbedProvenance` so the two encoders can re-version
    on independent cadences.
    """
    model_id: str
    embedding_dim: int
    task_instruction: str
    embedded_at: str
    image_preprocessing: str  # e.g. "rgb-resize-pad-512x512-grey"
    n_with_image: int
    n_without_image: int
    matryoshka_dim: int | None = None
    model_revision: str | None = None
    sentence_transformers_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ArtProvenance:
    """Records how card portrait art was extracted from local game files.

    STS1 art comes from desktop-1.0.jar (zipfile); STS2 art comes from
    sts2.pck (GDRE Tools). Extraction is local-only — game files are not
    redistributed.
    """
    extraction_source: str  # "jar" or "pck"
    source_file_sha256: str
    extracted_at: str  # ISO timestamp
    n_art_files: int
    n_cards_total: int
    resolution: str  # "high" or "low"
    image_dimensions: tuple[int, int] | None = None
    gdre_tools_version: str | None = None  # only meaningful for STS2

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # JSON has no native tuple — store as a 2-element list and read back as tuple
        if self.image_dimensions is not None:
            d["image_dimensions"] = list(self.image_dimensions)
        return d


@dataclass
class DatasetProvenance:
    """Top-level provenance attached to each dataset snapshot."""
    fetch: FetchProvenance
    embed: EmbedProvenance | None = None
    multimodal_embed: MultimodalEmbedProvenance | None = None
    art: ArtProvenance | None = None
    package_version: str = __version__
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    platform: str = field(default_factory=platform.platform)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fetch": self.fetch.to_dict(),
            "embed": self.embed.to_dict() if self.embed else None,
            "multimodal_embed": (self.multimodal_embed.to_dict()
                                 if self.multimodal_embed else None),
            "art": self.art.to_dict() if self.art else None,
            "package_version": self.package_version,
            "python_version": self.python_version,
            "platform": self.platform,
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))

    @classmethod
    def read(cls, path: Path) -> "DatasetProvenance":
        data = json.loads(path.read_text())
        fetch = FetchProvenance(**data["fetch"])
        embed = EmbedProvenance(**data["embed"]) if data.get("embed") else None
        multimodal_embed = (
            MultimodalEmbedProvenance(**data["multimodal_embed"])
            if data.get("multimodal_embed") else None
        )
        art = None
        if data.get("art"):
            art_data = dict(data["art"])
            dims = art_data.get("image_dimensions")
            if dims is not None:
                art_data["image_dimensions"] = tuple(dims)
            art = ArtProvenance(**art_data)
        return cls(
            fetch=fetch,
            embed=embed,
            multimodal_embed=multimodal_embed,
            art=art,
            package_version=data.get("package_version", "unknown"),
            python_version=data.get("python_version", "unknown"),
            platform=data.get("platform", "unknown"),
        )


def get_st_version() -> str | None:
    """Best-effort detection of the installed sentence-transformers version."""
    try:
        import sentence_transformers
        return sentence_transformers.__version__
    except Exception:
        return None
