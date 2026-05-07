# slaythespire-codex

Slay the Spire 1 + 2 card dataset and embedding pipeline. **The four HuggingFace datasets this pipeline produces are collected for ML/DL training:** normalized card metadata, derived feature columns, and 2026-current text embeddings, shaped so consumers can `load_dataset(...)` and feed straight into a classifier, retriever, or model fine-tune without further preprocessing. One cards repo and one embeddings repo per game, joinable on `id`.

[![License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey.svg)](LICENSE-DATA)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

## Datasets

This pipeline produces **four** HuggingFace datasets, two per game, splitting card metadata from embeddings:

| Game | Kind | HuggingFace |
| --- | --- | --- |
| Slay the Spire 1 | Cards (metadata + features) | `t22000t/slay-the-spire-1-cards` |
| Slay the Spire 1 | Embeddings (vectors + UMAP) | `t22000t/slay-the-spire-1-card-embeddings` |
| Slay the Spire 2 | Cards (metadata + features) | `t22000t/slay-the-spire-2-cards` |
| Slay the Spire 2 | Embeddings (vectors + UMAP) | `t22000t/slay-the-spire-2-card-embeddings` |

Cards and embeddings are joinable on `id`. The split keeps each repo focused, people running text classifiers on cards don't pay for an embedding column they'll discard, and the embedding repos can be re-versioned independently when the model changes without churning the cards repos' commit history.

Both games' embeddings use the same model and instruction prompt, so vectors are directly comparable across games for cross-game similarity search.

## Quick start

The package is not on PyPI yet, install from source:

```bash
git clone https://github.com/timothy22000/slaythespire-codex
cd slaythespire-codex
pip install -e ".[all]"

# fetch + embed + visualize STS1
sts-cards fetch sts1
sts-cards embed sts1
sts-cards visualize sts1

# same for STS2, record the game version for provenance
sts-cards fetch sts2 --sts-game-version v0.103.0
sts-cards embed sts2
sts-cards visualize sts2

# search
sts-cards search sts1 --card "Strike" --k 10
sts-cards search sts2 --query "deal damage and apply vulnerable"

# upload (4 repos: 2 per game)
sts-cards upload sts1 --kind cards      --repo t22000t/slay-the-spire-1-cards
sts-cards upload sts1 --kind embeddings --repo t22000t/slay-the-spire-1-card-embeddings
sts-cards upload sts2 --kind cards      --repo t22000t/slay-the-spire-2-cards
sts-cards upload sts2 --kind embeddings --repo t22000t/slay-the-spire-2-card-embeddings
```

## What's in each dataset

**Cards repos** (`*-cards`):
- `cards.parquet`: normalized metadata + derived feature columns
- `provenance.json`: fetch source and date, package version, host platform
- `croissant.json`: JSON-LD schema descriptor

**Embeddings repos** (`*-card-embeddings`):
- `embeddings.parquet`: `id`, `name`, `card_text`, 1024-D embedding, UMAP coords
- `provenance.json`: same as above + embedding model, instruction, date
- `croissant.json`: JSON-LD schema descriptor

Cards and embeddings are joinable on the `id` column.

## Embedding recipe

**No training happens in this repo.** The `embed` command runs *inference* with a frozen pretrained model, no labels, no optimizer, no gradient updates. The "ML/DL training-ready" framing above is about what *consumers* can do with the published datasets, not about training that happens here.

Following the approach pioneered by [`minimaxir/mtg-embeddings`](https://huggingface.co/datasets/minimaxir/mtg-embeddings):

- Each card is encoded as a prettified-JSON string of its mechanics-relevant fields (the indentation measurably helps embedding quality)
- The card's own name inside its description is replaced with `~`
- An instruction prompt is prepended so the model knows what semantic axis matters
- Encoded with [`Qwen/Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), at release the 8B variant ranked #1 on the multilingual MTEB
- Unit-normalized so dot product == cosine similarity

Both datasets share the same model and instruction, so vectors are directly comparable across games for cross-game similarity search.

## Provenance

Every output records exactly how it was produced:

```json
{
  "fetch": {
    "source": "https://spire-archive.com/api/sts2/cards",
    "source_fetched_at": "2026-05-07T14:23:11+00:00",
    "game": "sts2",
    "language": "en",
    "n_cards": 576,
    "sts_game_version": "v0.103.0"
  },
  "embed": {
    "model_id": "Qwen/Qwen3-Embedding-0.6B",
    "embedding_dim": 1024,
    "task_instruction": "Represent this Slay the Spire card so that...",
    "embedded_at": "2026-05-07T14:25:00+00:00",
    "matryoshka_dim": null,
    "sentence_transformers_version": "5.4.1"
  },
  "package_version": "0.1.0",
  "python_version": "3.12.3",
  "platform": "Linux-..."
}
```

This makes STS2 patch drift traceable, each refresh bumps the timestamp and the game version, and consumers can compare snapshots without guessing.

## Development

```bash
git clone https://github.com/timothy22000/slaythespire-codex
cd slaythespire-codex
pip install -e ".[all]"

pytest                         # unit tests, no model/network needed
ruff check src tests           # lint
mypy src                       # types
```

The test suite runs without a GPU, without network, and without loading any embedding model. CI-friendly by design.

## Licensing

- **Code** is MIT (see [`LICENSE`](LICENSE))
- **Dataset content** is CC BY 4.0 (see [`LICENSE-DATA`](LICENSE-DATA))
- **Game IP** belongs to Mega Crit. This project ships factual reference data about a published game; no card art or proprietary creative assets are included.

## Credits

- Card data parsed from game files by [`nkhoit/spire-archive`](https://github.com/nkhoit/spire-archive), please credit them if you build on this
- Embedding recipe adapted from [`minimaxir/mtg-embeddings`](https://huggingface.co/datasets/minimaxir/mtg-embeddings)
- Slay the Spire and Slay the Spire 2 are © [Mega Crit](https://www.megacrit.com/)
