---
license: cc-by-4.0
language:
- en
pretty_name: "Slay the Spire 1: Card Embeddings"
size_categories:
- n<1K
task_categories:
- feature-extraction
- sentence-similarity
tags:
- games
- card-games
- slay-the-spire
- embeddings
- sentence-transformers
- qwen3
configs:
- config_name: default
  data_files: embeddings.parquet
---

# Slay the Spire 1: Card Embeddings

1024-D unit-normalized text embeddings for every card in **Slay the Spire**, produced by *inference* with the pretrained [`Qwen/Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) (frozen, no fine-tuning was done to generate this dataset). **Collected for ML/DL training:** drop directly into a retriever, similarity index, or downstream model that consumes pre-encoded vectors.

This is the **embeddings** dataset. For card metadata (name, cost, description, derived features), see the companion dataset:
**[`t22000t/slay-the-spire-1-cards`](https://huggingface.co/datasets/t22000t/slay-the-spire-1-cards)**, joinable to this dataset by `id`.

For Slay the Spire 2 embeddings, see **[`t22000t/slay-the-spire-2-card-embeddings`](https://huggingface.co/datasets/t22000t/slay-the-spire-2-card-embeddings)**. Vectors from both games share the same model and instruction prompt, so they're directly comparable for cross-game similarity search.

## Dataset Description

- **Repository:** [`timothy22000/slaythespire-codex`](https://github.com/timothy22000/slaythespire-codex)
- **Card count:** 360
- **Embedding model:** [`Qwen/Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- **Embedding dimension:** 1024
- **Normalization:** unit-norm (so dot product equals cosine similarity)

## Data Fields

| Field | Type | Description |
| --- | --- | --- |
| `id` | string | Card identifier, **the join key to the cards dataset** |
| `game` | string | Always `"sts1"` |
| `name` | string | Display name (kept for convenience) |
| `card_text` | string | The prettified-JSON card document fed to the embedder |
| `embedding` | list[float32] (1024) | Unit-normalized text embedding |
| `umap_x` | float | First UMAP-2D coordinate |
| `umap_y` | float | Second UMAP-2D coordinate |

## Embedding Recipe

Following the approach pioneered by [`minimaxir/mtg-embeddings`](https://huggingface.co/datasets/minimaxir/mtg-embeddings):

- Each card is encoded as a **prettified JSON string** of its mechanics-relevant fields. Indentation is intentional, measurably improves embedding quality.
- Card-name self-references inside descriptions are replaced with `~` so the embedding isn't dominated by name surface form.
- A **task instruction** is prepended at encode time: *"Represent this Slay the Spire card so that mechanically similar cards (same archetype, comparable damage/block patterns, related keywords) are close in embedding space."*
- Encoded with [`Qwen/Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), the 8B variant of this family ranked #1 on the multilingual MTEB leaderboard at release.
- Embeddings are **unit-normalized**, so cosine similarity is just a matrix dot product.

The exact model id, instruction, and embedding date are recorded in `provenance.json` for full reproducibility.

## Loading

```python
from datasets import load_dataset
import numpy as np

ds = load_dataset("t22000t/slay-the-spire-1-card-embeddings", split="train")
emb = np.array(ds["embedding"], dtype=np.float32)

# Find the 10 cards most similar to "Strike"
i = ds["name"].index("Strike")
sims = emb @ emb[i]
top = np.argsort(-sims)[1:11]
for j in top:
    print(f"{sims[j]:.3f}  {ds[j]['name']}")
```

For 360 cards, the entire similarity matrix fits in a few MB of RAM and a query is sub-millisecond.

## Joining with card metadata

The embedding dataset is intentionally minimal. To get card details (cost, description, type, etc.), join with the cards dataset on `id`:

```python
import pandas as pd
from datasets import load_dataset

embs  = load_dataset("t22000t/slay-the-spire-1-card-embeddings", split="train").to_pandas()
cards = load_dataset("t22000t/slay-the-spire-1-cards", split="train").to_pandas()

df = embs.merge(cards, on="id", suffixes=("", "_card"))
```

## Cross-game similarity

Both STS1 and STS2 embeddings use the same model and instruction prompt, so vectors are directly comparable:

```python
import numpy as np
from datasets import load_dataset

sts1 = load_dataset("t22000t/slay-the-spire-1-card-embeddings", split="train")
sts2 = load_dataset("t22000t/slay-the-spire-2-card-embeddings", split="train")

e1 = np.array(sts1["embedding"], dtype=np.float32)
e2 = np.array(sts2["embedding"], dtype=np.float32)

# STS2 cards most similar to "Bash" (an STS1 card)
i = sts1["name"].index("Bash")
sims = e2 @ e1[i]
print(np.array(sts2["name"])[np.argsort(-sims)[:10]])
```

## Considerations for Using the Data

### Discussion of Biases

The embeddings inherit the biases of `Qwen/Qwen3-Embedding-0.6B`, which was trained on multilingual web text. Mechanics described with vocabulary common in the training distribution (damage, block, draw) will likely be better separated than mechanics described with rarer or more idiosyncratic phrasing.

### Other Known Limitations

- **English only.** Other locales would require re-encoding with the multilingual capacity of Qwen3-Embedding (which it has, just not exercised here).
- **Quality not yet formally evaluated.** A held-out card-pair similarity benchmark is on the project roadmap. Until it lands, treat the rankings as reasonable but not metric-validated.

## Provenance

A `provenance.json` ships with this dataset recording the embedding model id, the task instruction, the embedding date, and the upstream fetch source. Critical for downstream search code, which must encode queries with the same model + instruction.

## Citation

```bibtex
@dataset{sts1_card_embeddings_dataset,
  title = {Slay the Spire 1: Card Embeddings},
  author = {timothy22000},
  year = {2026},
  url = {https://huggingface.co/datasets/t22000t/slay-the-spire-1-card-embeddings},
  note = {Embeddings from Qwen/Qwen3-Embedding-0.6B; card data via nkhoit/spire-archive; game IP © Mega Crit}
}
```

## Licensing

- **Dataset:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Pipeline code:** MIT, see [github.com/timothy22000/slaythespire-codex](https://github.com/timothy22000/slaythespire-codex)
- **Game IP:** Slay the Spire is © [Mega Crit](https://www.megacrit.com/).
