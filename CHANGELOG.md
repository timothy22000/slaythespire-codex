# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-05-07

### Changed — split into 4 HF datasets

Each game now ships as TWO HuggingFace datasets instead of one, separating
card metadata from embeddings:

  - `{user}/slay-the-spire-1-cards`            (metadata + features)
  - `{user}/slay-the-spire-1-card-embeddings`  (vectors + UMAP)
  - `{user}/slay-the-spire-2-cards`
  - `{user}/slay-the-spire-2-card-embeddings`

Joinable on `id`. The split lets the embedding repos be re-versioned when
the model changes without touching the cards repos' commit history, and
lets card-text-only consumers avoid downloading an unnecessary 4MB
embedding column.

### Added
- `REPO_KINDS = ("cards", "embeddings")` constant in `__init__.py`
- `embed_game()` now writes a slim `{game}_embeddings.parquet` (just id,
  game, name, card_text, embedding) alongside the existing internal
  `{game}_cards_with_embeddings.parquet`
- `visualize.project_2d()` mirrors UMAP coords into the slim file too
- `upload.upload(repo, game, kind)` routes the right files to each repo
- `croissant.build_croissant(..., kind=...)` produces per-kind descriptors
  with appropriate names, descriptions, and keyword sets
- CLI flags: `--kind cards|embeddings` on both `upload` and `croissant`
- `tests/test_upload.py` — 6 new tests for file routing
- 3 new croissant tests covering the embeddings kind and validation

### Removed
- Old `upload.upload_game()` function (replaced by `upload.upload(..., kind=...)`)
- Old `--use-embeddings/--use-cards-only` flag on `croissant` (replaced by `--kind`)

## [0.2.0] - 2026-05-07

### Added — Tier 2 polish

- **HTTP caching for fetch** via `requests-cache`. Default 1h TTL, on-disk
  SQLite at `~/.cache/sts-cards/api.sqlite`. Bypass with `--no-cache`,
  wipe with `sts-cards cache-clear`. Tenacity retries layered on top.
- **STS2-specific feature columns**: `damage`, `block`, `targets_all_enemies`,
  `status_effects_applied`, `mechanics`, `orbs_channeled`, `orbs_referenced`,
  `forge_value`, `souls_added`. Generic columns also populated for STS1.
  All extracted from card description text via tested regex.
- **Croissant JSON-LD metadata**. New `sts-cards croissant {game}` command
  generates `{game}_croissant.json` describing the dataset for HuggingFace
  and Google Dataset Search. Field types inferred from the Parquet,
  human-readable descriptions for every column, and STS2 marked as a live
  dataset.
- **GitHub Actions**:
  - `ci.yml` — runs on push/PR with Python 3.10/3.11/3.12 matrix; lints
    with ruff and runs the full pytest suite, plus a CLI smoke check.
  - `refresh.yml` — scheduled (STS2 weekly, STS1 monthly) and manually
    triggerable. Fetches, embeds, projects with UMAP, generates Croissant,
    uploads to HuggingFace. Requires `HF_TOKEN` secret.
- **Test coverage** expanded from 33 to 80 tests across 6 modules. New
  files: `test_features.py`, `test_croissant.py`, `test_cache.py`.

### Changed
- `fetch_game()` now extracts derived features and merges them into the
  output Parquet schema.
- `upload.py` now also uploads the Croissant descriptor when present.

## [0.1.0] - 2026-05-07

### Added
- Initial release. Pipeline for fetching, normalizing, embedding, and uploading
  Slay the Spire 1 and Slay the Spire 2 cards as separate HuggingFace datasets.
- `sts-cards` Typer CLI with `fetch`, `embed`, `visualize`, `search`, `upload`
  commands operating on one game at a time.
- Default embedding model: `Qwen/Qwen3-Embedding-0.6B` with task-instruction
  prompting, unit-normalized 1024-D vectors, optional Matryoshka truncation.
- Provenance tracking via `{game}_provenance.json` recording source URL,
  fetch and embed timestamps, model id, embedding parameters, host environment,
  and STS2 game version.
- Pytest test suite covering normalization, provenance round-trip, and
  similarity search; runs in <1 second with no model or network dependencies.
- Dataset cards for both HF repos with full HF-standard sections.
- MIT license for code, CC BY 4.0 for dataset content; clear separation between
  pipeline code and game-IP attribution.

[Unreleased]: https://github.com/timothy22000/slaythespire-codex/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/timothy22000/slaythespire-codex/releases/tag/v0.1.0
