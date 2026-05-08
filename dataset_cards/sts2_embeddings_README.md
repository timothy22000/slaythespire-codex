---
license: cc-by-4.0
language:
- en
pretty_name: "Slay the Spire 2: Card Embeddings"
size_categories:
- n<1K
task_categories:
- feature-extraction
- sentence-similarity
tags:
- games
- card-games
- slay-the-spire
- slay-the-spire-2
- embeddings
- sentence-transformers
- qwen3
- early-access
configs:
- config_name: default
  data_files: embeddings.parquet
---

# Slay the Spire 2: Card Embeddings

1024-D unit-normalized text embeddings for every card in **Slay the Spire 2** (Early Access), produced by *inference* with the pretrained [`Qwen/Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) (frozen, no fine-tuning was done to generate this dataset). **Collected for ML/DL training:** drop directly into a retriever, similarity index, or downstream model that consumes pre-encoded vectors.

This is the **text-embeddings** dataset. Companion datasets, all joinable on `id`:

- **[`t22000t/slay-the-spire-2-cards`](https://huggingface.co/datasets/t22000t/slay-the-spire-2-cards)** - card metadata + derived features + inline portrait art
- **[`t22000t/slay-the-spire-2-card-multimodal-embeddings`](https://huggingface.co/datasets/t22000t/slay-the-spire-2-card-multimodal-embeddings)** - joint text+image embeddings via `Qwen/Qwen3-VL-Embedding-2B` (use when portrait similarity matters too)

For Slay the Spire 1 text embeddings, see **[`t22000t/slay-the-spire-1-card-embeddings`](https://huggingface.co/datasets/t22000t/slay-the-spire-1-card-embeddings)**. Vectors from both games share the same model and instruction prompt, so they're directly comparable for cross-game similarity search.

The full bundle (6 datasets across both games + 3 Gradio demos) is in the [**slaythespire-codex collection**](https://huggingface.co/collections/t22000t/slaythespire-codex).

> ⚠️ **Early Access, content is unstable.** STS2 cards change with patches. When the cards dataset is refreshed, this embedding dataset is re-built and re-uploaded. Always check `provenance.json` for the snapshot version and embedding date.

## Dataset Description

- **Repository:** [`timothy22000/slaythespire-codex`](https://github.com/timothy22000/slaythespire-codex)
- **Card count:** ~576 (varies by patch)
- **Embedding model:** [`Qwen/Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- **Embedding dimension:** 1024
- **Normalization:** unit-norm (so dot product equals cosine similarity)

## Data Fields

| Field | Type | Description |
| --- | --- | --- |
| `id` | string | Card identifier, **the join key to the cards dataset** |
| `game` | string | Always `"sts2"` |
| `name` | string | Display name (kept for convenience) |
| `card_text` | string | The prettified-JSON card document fed to the embedder |
| `embedding` | list[float32] (1024) | Unit-normalized text embedding |
| `umap_x` | float | First UMAP-2D coordinate |
| `umap_y` | float | Second UMAP-2D coordinate |

## Embedding Recipe

Identical to the STS1 embeddings dataset:

- Each card encoded as a prettified-JSON string of mechanics-relevant fields (indentation intentional)
- Card-name self-references replaced with `~`
- Task instruction prepended at encode time
- Encoded with [`Qwen/Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), unit-normalized

Because the recipe matches STS1, embeddings from both games live in a shared coordinate system and are directly comparable.

## Loading

```python
from datasets import load_dataset
import numpy as np

ds = load_dataset("t22000t/slay-the-spire-2-card-embeddings", split="train")
emb = np.array(ds["embedding"], dtype=np.float32)

# Find the 10 cards most similar to "Zap"
i = ds["name"].index("Zap")
sims = emb @ emb[i]
top = np.argsort(-sims)[1:11]
for j in top:
    print(f"{sims[j]:.3f}  {ds[j]['name']}")
```

## Joining with card metadata

```python
from datasets import load_dataset

embs  = load_dataset("t22000t/slay-the-spire-2-card-embeddings", split="train").to_pandas()
cards = load_dataset("t22000t/slay-the-spire-2-cards", split="train").to_pandas()

df = embs.merge(cards, on="id", suffixes=("", "_card"))
```

## Considerations for Using the Data

### Patch drift

This dataset reflects the STS2 cards in the snapshot the embedding model was run against. After a patch, both the cards repo and this embeddings repo get rebuilt. **Use the `provenance.json` to align snapshots**, never assume a specific embedding still corresponds to its current card text.

### Discussion of Biases

The embeddings inherit the biases of `Qwen/Qwen3-Embedding-0.6B`. STS2-specific mechanics (Orbs, Forge, Souls, Enchantments) use vocabulary the model has likely never seen during training, so semantic separation along those axes depends on contextual cues from the rest of the card text. Cross-game similarity is more reliable for shared mechanics (damage, block, vulnerable) than for STS2-only ones.

### Other Known Limitations

- **English only.**
- **Quality not yet formally evaluated** on a card-pair benchmark, on the project roadmap.

## Provenance

A `provenance.json` ships with this dataset recording the embedding model id, the task instruction, the embedding date, and the upstream STS2 game version.

## Citation

```bibtex
@dataset{sts2_card_embeddings_dataset,
  title = {Slay the Spire 2: Card Embeddings},
  author = {timothy22000},
  year = {2026},
  url = {https://huggingface.co/datasets/t22000t/slay-the-spire-2-card-embeddings},
  note = {Early Access snapshot; embeddings from Qwen/Qwen3-Embedding-0.6B; card data via nkhoit/spire-archive; game IP © Mega Crit}
}
```

## Licensing

- **Dataset:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Pipeline code:** MIT, see [github.com/timothy22000/slaythespire-codex](https://github.com/timothy22000/slaythespire-codex)
- **Game IP:** Slay the Spire 2 is © [Mega Crit](https://www.megacrit.com/).
