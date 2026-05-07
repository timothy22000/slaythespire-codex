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
class DatasetProvenance:
    """Top-level provenance attached to each dataset snapshot."""
    fetch: FetchProvenance
    embed: EmbedProvenance | None = None
    package_version: str = __version__
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    platform: str = field(default_factory=platform.platform)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fetch": self.fetch.to_dict(),
            "embed": self.embed.to_dict() if self.embed else None,
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
        # Kwargs not exposed in __init__ signatures of frozen dataclasses are
        # filtered, but ours aren't frozen — pass through what we got.
        return cls(
            fetch=fetch,
            embed=embed,
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
