# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2026-05-08

### Added — multimodal embeddings via Qwen3-VL-Embedding-2B

Each game now publishes a third HuggingFace dataset:
`slay-the-spire-{1,2}-card-multimodal-embeddings`. Six repos total.
One unit-normalized 1024-D vector per card, joint text+image space,
joinable to the cards repos by `id`. Cross-game similarity is preserved
(same model, same instruction, same dim) so STS1×STS2 dot products work.

- `src/sts_cards/multimodal_embed.py` — new module. Loads
  `Qwen/Qwen3-VL-Embedding-2B` (Apache 2.0, lazy-imported), pads images
  to 512×512 RGB on neutral grey, encodes with task-instruction
  prompting, Matryoshka-truncates to 1024-D, unit-normalizes.
- `sts-cards embed-multimodal {game}` — new CLI command. Mirrors the
  existing `embed` flags (`--model`, `--task-instruction`,
  `--matryoshka-dim`, `--batch-size`, `--device`) with multimodal
  defaults (dim=1024, batch=4 — VL + images is heavier).
- `MultimodalEmbedProvenance` dataclass on `DatasetProvenance.multimodal_embed`
  records model id, dim, instruction, image preprocessing recipe,
  `n_with_image`, `n_without_image`.
- `croissant.py` accepts `kind="multimodal-embeddings"` with its own
  pretty name and keyword set; new `image` and `multimodal_embedding`
  field descriptions.
- `upload.py` adds `_files_for_multimodal_embeddings` and routes the
  new kind. Inside the repo the file is named `embeddings.parquet`
  (same as the text-embeddings repo) so consumer code is portable.
- Two new dataset cards in `dataset_cards/`.
- `.github/workflows/upload-multimodal.yml` — manual workflow_dispatch
  to publish a locally-encoded parquet. NOT wired into `refresh.yml`
  because the 2B model + image tensors won't fit on a free GH runner.
- 8 new tests bringing the total to 117.

### Changed

- `REPO_KINDS` extends to `("cards", "embeddings", "multimodal-embeddings")`.
- STS2 art extraction switched to `gdre_tools --recover` with targeted
  `--include` globs (recovery decodes `.ctex` blobs back to PNG; raw
  `--extract` returned the compressed textures unchanged). 100% match
  on STS2 (576/576), ~99.7% on STS1.
- `extract_art.candidate_keys` now also feeds the STS2 PCK path so
  parquet ids in SCREAMING_SNAKE_CASE join cleanly to lowercase
  recovered stems (case + word-split aware, same alias table as STS1
  plus `MAD_SCIENCE → mad_science_attack`).

## [0.4.0] - 2026-05-08

### Added — card art as a column on the cards Parquet

Each cards Parquet now carries an `image` column with the in-game card portrait
as raw PNG bytes. HuggingFace's `datasets.Image()` feature decodes those bytes
back to PIL on `load_dataset()`, and the dataset viewer renders thumbnails inline.

- `extract_art.py` module — STS1 portraits via `zipfile` over `desktop-1.0.jar`,
  STS2 portraits via a GDRE Tools subprocess over `sts2.pck` (cached on disk
  by `sha256(pck)`); both keep PNG bytes in memory and never write per-card
  files to disk.
- `sts-cards diagnose-art {game}` — lists candidate portrait paths and reports
  the join rate against `cards.parquet["id"]`. Run before `extract-art`.
- `sts-cards extract-art {game}` — extracts and attaches the `image` and
  `image_resolution` columns in place. Resolution flag (`--resolution
  high|low`) is honoured for STS1; STS2 ships one resolution.
- `ArtProvenance` dataclass on `DatasetProvenance.art` — records extraction
  source, source-file SHA-256, timestamp, match counts, resolution, and
  GDRE Tools version (STS2 only).
- `.github/workflows/upload-art.yml` — manual `workflow_dispatch` to
  re-upload the cards repo after a local extraction. CI does NOT run
  extraction (no game files on the runner).
- ~12 new tests in `tests/test_extract_art.py` and updates to
  `test_provenance.py` and `test_croissant.py` for the new column types.

### Changed

- `embed.py` drops `image` and `image_resolution` from
  `cards_with_embeddings.parquet` so the in-process file stays small;
  the slim `embeddings.parquet` already excluded them.
- `croissant.py` maps the `image` column to `sc:ImageObject` instead of
  defaulting to `sc:Text`.
- Dataset cards (`sts1_cards_README.md`, `sts2_cards_README.md`) declare
  the `image` feature in YAML frontmatter so the HF viewer renders
  thumbnails, and document the size jump (~200 KB → ~50–80 MB).
- `refresh.yml` logs whether the staged `cards.parquet` carries an
  `image` column instead of silently publishing without portraits.

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
