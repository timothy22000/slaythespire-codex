"""Upload one game's data to its HuggingFace dataset repo.

Each game maps to TWO HuggingFace dataset repos:

  - kind="cards"      → repo with metadata + derived feature columns
  - kind="embeddings" → repo with embedding vectors + UMAP coords

The split keeps each repo focused: people who want to train a card-text
classifier shouldn't have to download a 4MB embedding column they'll
discard, and the embedding repo can be re-versioned independently when
the model changes without churning the cards repo.

The two repos are joinable on `id`.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


def _files_for_cards(game: str, out_dir: Path) -> list[tuple[Path, str]]:
    """Files for the cards repo. Returns (local_path, path_in_repo)."""
    pairs: list[tuple[Path, str]] = []
    for src_name, in_repo_name in [
        (f"{game}_cards.parquet", "cards.parquet"),
        (f"{game}_provenance.json", "provenance.json"),
        (f"{game}_cards_croissant.json", "croissant.json"),
    ]:
        src = out_dir / src_name
        if src.exists():
            pairs.append((src, in_repo_name))
    return pairs


def _files_for_embeddings(game: str, out_dir: Path) -> list[tuple[Path, str]]:
    """Files for the embeddings repo."""
    pairs: list[tuple[Path, str]] = []
    for src_name, in_repo_name in [
        (f"{game}_embeddings.parquet", "embeddings.parquet"),
        (f"{game}_provenance.json", "provenance.json"),
        (f"{game}_embeddings_croissant.json", "croissant.json"),
    ]:
        src = out_dir / src_name
        if src.exists():
            pairs.append((src, in_repo_name))
    return pairs


def files_for(game: str, kind: str, out_dir: Path) -> list[tuple[Path, str]]:
    """Resolve which files belong in which repo kind."""
    if kind == "cards":
        return _files_for_cards(game, out_dir)
    if kind == "embeddings":
        return _files_for_embeddings(game, out_dir)
    raise ValueError(f"unknown kind: {kind!r} (expected 'cards' or 'embeddings')")


def upload(
    repo: str,
    game: str,
    kind: str,
    *,
    out_dir: Path = Path("output"),
    readme: Path | None = None,
    private: bool = False,
) -> None:
    """Push one (game, kind) combination to its HuggingFace dataset repo."""
    from huggingface_hub import HfApi, create_repo

    if kind not in ("cards", "embeddings"):
        raise ValueError(f"kind must be 'cards' or 'embeddings', got {kind!r}")

    api = HfApi()
    create_repo(repo_id=repo, repo_type="dataset", private=private, exist_ok=True)
    log.info("Repo ready: %s", repo)

    if readme and readme.exists():
        api.upload_file(
            path_or_fileobj=str(readme),
            path_in_repo="README.md",
            repo_id=repo, repo_type="dataset",
        )
        log.info("Uploaded README.md")

    pairs = files_for(game, kind, out_dir)
    if not pairs:
        log.warning("No files to upload for game=%s kind=%s in %s",
                    game, kind, out_dir)
        return

    for local_path, in_repo_name in pairs:
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=in_repo_name,
            repo_id=repo, repo_type="dataset",
        )
        log.info("Uploaded %s → %s", local_path.name, in_repo_name)

    log.info("Done: https://huggingface.co/datasets/%s", repo)
