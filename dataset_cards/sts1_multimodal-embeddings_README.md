---
license: cc-by-4.0
language:
- en
pretty_name: "Slay the Spire 1: Multimodal Card Embeddings"
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
- roguelike
- embeddings
- multimodal
- vision-language
configs:
- config_name: default
  data_files: embeddings.parquet
---

# Slay the Spire 1: Multimodal Card Embeddings

Joint text+image embeddings for every card in **Slay the Spire** (1.0 release), produced by [`Qwen/Qwen3-VL-Embedding-2B`](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B). One unit-normalized 1024-D vector per card. Mechanically AND visually similar cards land near each other; cards across STS1 and STS2 share the coordinate system.

This is the **multimodal-embeddings** dataset. For text-only embeddings or the underlying card metadata + portraits, see:

- **[`t22000t/slay-the-spire-1-cards`](https://huggingface.co/datasets/t22000t/slay-the-spire-1-cards)** - metadata + features + inline portrait art
- **[`t22000t/slay-the-spire-1-card-embeddings`](https://huggingface.co/datasets/t22000t/slay-the-spire-1-card-embeddings)** - text-only embeddings via Qwen3-Embedding-0.6B

All three are joinable on `id`. For the STS2 multimodal counterpart, see [`t22000t/slay-the-spire-2-card-multimodal-embeddings`](https://huggingface.co/datasets/t22000t/slay-the-spire-2-card-multimodal-embeddings).

The full bundle (6 datasets across both games + 3 Gradio demos) is in the [**slaythespire-codex collection**](https://huggingface.co/collections/t22000t/slaythespire-codex).

## Dataset Description

- **Repository:** [`timothy22000/slaythespire-codex`](https://github.com/timothy22000/slaythespire-codex)
- **Embedding model:** [`Qwen/Qwen3-VL-Embedding-2B`](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) (Apache 2.0)
- **Card count:** 360 (1 text-only - `IMPULSE` has no portrait in the JAR)
- **Vector dim:** 1024 (Matryoshka-truncated from 2048; matches the text-embedding repos for consistency)
- **License:** CC BY 4.0; game IP belongs to Mega Crit

## Data Fields

| Field | Type | Description |
| --- | --- | --- |
| `id` | string | Stable card identifier - **the join key** |
| `game` | string | Always `"sts1"` |
| `name` | string | Display name |
| `card_text` | string | Prettified-JSON document fed to the encoder |
| `has_image` | bool | True when the card had a portrait at embed time |
| `multimodal_embedding` | list[float32] (1024) | Unit-normalized joint text+image vector |

## Embedding recipe

- **Model:** `Qwen/Qwen3-VL-Embedding-2B`, frozen, no fine-tuning. The 0.6B text-only encoder used for `card-embeddings` is a different family member; the two repos exist on independent re-version cadences.
- **Image preprocessing:** decode PNG → RGB (alpha dropped) → resize-with-pad to 512×512 on neutral grey. Padding rather than center-cropping preserves character iconography at the edges of STS portraits.
- **Cards without art:** fed text-only through the same model. Preserves the joint coordinate system; provenance records `n_with_image` and `n_without_image`.
- **Truncation:** Matryoshka to 1024-D, then re-normalized.
- **Task instruction prepended at encode time:**

  > Represent this Slay the Spire card so that mechanically similar cards (same archetype, comparable damage/block patterns, related keywords) are close in embedding space, using the card's text mechanics as the primary signal and the portrait art as a secondary cue for character/class and visual archetype.

  The "primary/secondary" framing explicitly demotes art so visually similar but mechanically distant cards don't dominate clusters.

## Loading

```python
from datasets import load_dataset
import numpy as np

ds = load_dataset("t22000t/slay-the-spire-1-card-multimodal-embeddings", split="train")
emb = np.array(ds["multimodal_embedding"], dtype=np.float32)
print(emb.shape)          # (360, 1024)
print(emb @ emb[0])       # cosine similarity to card 0
```

## Cross-game similarity

Same model, same instruction, same dim → STS1×STS2 similarity is a single dot product:

```python
from datasets import load_dataset
import numpy as np

sts1 = load_dataset("t22000t/slay-the-spire-1-card-multimodal-embeddings", split="train")
sts2 = load_dataset("t22000t/slay-the-spire-2-card-multimodal-embeddings", split="train")

emb1 = np.array(sts1["multimodal_embedding"], dtype=np.float32)
emb2 = np.array(sts2["multimodal_embedding"], dtype=np.float32)

i = sts1["name"].index("Bash")
sims = emb2 @ emb1[i]
top = np.argsort(-sims)[:10]
for j in top:
    print(f"{sims[j]:.3f}  {sts2[j]['name']}")
```

## Considerations for Using the Data

- **Lookalike-bias risk.** Multimodal embeddings can over-index on visual similarity. The instruction subordinates art to mechanics; if you observe undesirable lookalike clustering, the follow-up is a weighted concat of the separate text and image embeddings rather than a joint encode.
- **Cards without art.** 1 card (`IMPULSE`) has no portrait in the source JAR - its vector is text-only. `has_image=False` lets you filter or weight differently.
- **English only** in this snapshot.
- **Game IP.** Slay the Spire is © [Mega Crit](https://www.megacrit.com/). The dataset ships factual reference data + numerical embedding vectors only - no card art bytes are redistributed in this repo (the upstream `cards` repo is where art lives).

## Provenance

A `provenance.json` ships alongside the parquet with `multimodal_embed` block recording the model id, embedding dim, task instruction, image preprocessing recipe, `n_with_image`, `n_without_image`, and timestamp.

## Citation

```bibtex
@dataset{sts1_multimodal_card_embeddings,
  title  = {Slay the Spire 1: Multimodal Card Embeddings},
  author = {timothy22000},
  year   = {2026},
  url    = {https://huggingface.co/datasets/t22000t/slay-the-spire-1-card-multimodal-embeddings},
  note   = {Embedded with Qwen3-VL-Embedding-2B; card data via nkhoit/spire-archive; game IP (c) Mega Crit}
}
```

## Licensing

- **Dataset:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Pipeline code:** MIT, see [github.com/timothy22000/slaythespire-codex](https://github.com/timothy22000/slaythespire-codex)
- **Embedding model:** Apache 2.0 ([Qwen/Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B))
- **Game IP:** Slay the Spire is © [Mega Crit](https://www.megacrit.com/)
