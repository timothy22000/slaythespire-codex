---
license: cc-by-4.0
language:
- en
pretty_name: "Slay the Spire 2: Multimodal Card Embeddings"
size_categories:
- n<1K
task_categories:
- feature-extraction
- sentence-similarity
- image-feature-extraction
tags:
- games
- card-games
- deckbuilder
- slay-the-spire
- slay-the-spire-2
- roguelike
- early-access
- embeddings
- multimodal
- vision-language
configs:
- config_name: default
  data_files: embeddings.parquet
---

# Slay the Spire 2: Multimodal Card Embeddings

Joint text+image embeddings for every card in **Slay the Spire 2** (Early Access), produced by [`Qwen/Qwen3-VL-Embedding-2B`](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B). One unit-normalized 1024-D vector per card. Mechanically AND visually similar cards land near each other; cards across STS1 and STS2 share the coordinate system.

This is the **multimodal-embeddings** dataset. For text-only embeddings or the underlying card metadata + portraits, see:

- **[`t22000t/slay-the-spire-2-cards`](https://huggingface.co/datasets/t22000t/slay-the-spire-2-cards)** — metadata + features + inline portrait art
- **[`t22000t/slay-the-spire-2-card-embeddings`](https://huggingface.co/datasets/t22000t/slay-the-spire-2-card-embeddings)** — text-only embeddings via Qwen3-Embedding-0.6B

All three are joinable on `id`. For the STS1 multimodal counterpart, see [`t22000t/slay-the-spire-1-card-multimodal-embeddings`](https://huggingface.co/datasets/t22000t/slay-the-spire-1-card-multimodal-embeddings).

> ⚠️ **Early Access, content is unstable.** Slay the Spire 2 entered Early Access on March 5, 2026. Cards are added, removed, and rebalanced regularly. Always check `provenance.json` for the snapshot version.

## Dataset Description

- **Repository:** [`timothy22000/slaythespire-codex`](https://github.com/timothy22000/slaythespire-codex)
- **Embedding model:** [`Qwen/Qwen3-VL-Embedding-2B`](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) (Apache 2.0)
- **Card count:** ~576 (varies by patch). All cards have portraits at the time of writing.
- **Vector dim:** 1024 (Matryoshka-truncated from 2048; matches the text-embedding repos for consistency)
- **License:** CC BY 4.0; game IP belongs to Mega Crit

## Data Fields

| Field | Type | Description |
| --- | --- | --- |
| `id` | string | Stable card identifier — **the join key** |
| `game` | string | Always `"sts2"` |
| `name` | string | Display name |
| `card_text` | string | Prettified-JSON document fed to the encoder |
| `has_image` | bool | True when the card had a portrait at embed time |
| `multimodal_embedding` | list[float32] (1024) | Unit-normalized joint text+image vector |

## Embedding recipe

Identical to the STS1 multimodal repo so the two are coordinate-compatible:

- Model `Qwen/Qwen3-VL-Embedding-2B`, frozen.
- Image preprocessing: decode PNG → RGB → resize-with-pad to 512×512 on neutral grey.
- Same task instruction (mechanics primary, art secondary).
- Matryoshka truncation to 1024-D + re-normalize.

See the [STS1 multimodal repo](https://huggingface.co/datasets/t22000t/slay-the-spire-1-card-multimodal-embeddings) for the full instruction wording and rationale.

## Loading

```python
from datasets import load_dataset
import numpy as np

ds = load_dataset("t22000t/slay-the-spire-2-card-multimodal-embeddings", split="train")
emb = np.array(ds["multimodal_embedding"], dtype=np.float32)
print(emb.shape)          # (~576, 1024)
```

## Considerations for Using the Data

- **Patch drift.** STS2 cards turn over in Early Access; vectors only describe the snapshot in `provenance.json`. Don't compare numerical analyses across snapshots without aligning on game version.
- **Lookalike bias.** See the STS1 multimodal repo for discussion. Mitigated by the instruction wording but worth evaluating if you observe over-clustering on art alone.
- **English only** in this snapshot.
- **Game IP.** Slay the Spire 2 is © [Mega Crit](https://www.megacrit.com/). This dataset ships factual reference data + numerical embedding vectors only.

## Citation

```bibtex
@dataset{sts2_multimodal_card_embeddings,
  title  = {Slay the Spire 2: Multimodal Card Embeddings},
  author = {timothy22000},
  year   = {2026},
  url    = {https://huggingface.co/datasets/t22000t/slay-the-spire-2-card-multimodal-embeddings},
  note   = {Early Access snapshot; embedded with Qwen3-VL-Embedding-2B; card data via nkhoit/spire-archive; game IP (c) Mega Crit}
}
```

## Licensing

- **Dataset:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Pipeline code:** MIT
- **Embedding model:** Apache 2.0
- **Game IP:** Slay the Spire 2 is © [Mega Crit](https://www.megacrit.com/)
