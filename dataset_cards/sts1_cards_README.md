---
license: cc-by-4.0
language:
- en
pretty_name: "Slay the Spire 1: Cards"
size_categories:
- n<1K
task_categories:
- text-classification
- feature-extraction
tags:
- games
- card-games
- deckbuilder
- slay-the-spire
- roguelike
configs:
- config_name: default
  data_files: cards.parquet
features:
- name: image
  dtype: image
- name: image_resolution
  dtype: string
---

# Slay the Spire 1: Cards

A normalized dataset of every card in **Slay the Spire** (the original 2019 release), with derived feature columns for damage, block, status effects, and mechanics. **Collected for ML/DL training:** load with `datasets.load_dataset(...)` and feed straight into a card-text classifier, deckbuilder simulator, or design-analysis model.

This is the **cards** dataset. Companion datasets, all joinable on `id`:

- **[`t22000t/slay-the-spire-1-card-embeddings`](https://huggingface.co/datasets/t22000t/slay-the-spire-1-card-embeddings)** - 1024-D text embeddings via `Qwen/Qwen3-Embedding-0.6B`
- **[`t22000t/slay-the-spire-1-card-multimodal-embeddings`](https://huggingface.co/datasets/t22000t/slay-the-spire-1-card-multimodal-embeddings)** - joint text+image embeddings via `Qwen/Qwen3-VL-Embedding-2B`

The full bundle (4 datasets across both games + 3 Gradio demos) is in the [**slaythespire-codex collection**](https://huggingface.co/collections/t22000t/slaythespire-codex).

For Slay the Spire 2, see **[`t22000t/slay-the-spire-2-cards`](https://huggingface.co/datasets/t22000t/slay-the-spire-2-cards)**.

## Dataset Description

- **Repository:** [`timothy22000/slaythespire-codex`](https://github.com/timothy22000/slaythespire-codex)
- **Source data:** Parsed from game files by [`nkhoit/spire-archive`](https://github.com/nkhoit/spire-archive)
- **Card count:** 360
- **Languages:** English
- **License:** CC BY 4.0 (this dataset's structure and curation); game IP belongs to Mega Crit

### Why split from embeddings?

People doing card-text classification, deckbuilder simulators, or design analysis don't need the embedding column, and shipping it would mean a 4MB file instead of 200KB. The split also lets the embedding repo re-version independently when the embedding model changes, without touching the cards repo's commit history.

## Data Fields

Columns appear below in the order they're stored on disk. `id` and the portrait columns lead the row so the HF dataset viewer surfaces a thumbnail before metadata text.

### Identifier + portrait

| Field | Type | Description |
| --- | --- | --- |
| `id` | string | Stable card identifier, **the join key to the embeddings, multimodal-embeddings, and derived datasets** |
| `image` | HF `Image` feature (`struct<bytes, path>`) | Card portrait art. Decoded to a PIL Image automatically by `datasets.load_dataset()`; via `pd.read_parquet` you'll get a dict (`{"bytes": ..., "path": null}`). Cards without art get `null` (1 of 360: `IMPULSE`). |
| `image_resolution` | string | `"high"` (~1024×1024) or `"low"` (~256×256). |

### Card metadata

| Field | Type | Description |
| --- | --- | --- |
| `game` | string | Always `"sts1"` |
| `name` | string | Display name |
| `type` | string | `Attack`, `Skill`, `Power`, `Status`, or `Curse` |
| `rarity` | string | One of `Basic`, `Common`, `Uncommon`, `Rare`, `Special`, `Curse` (Title Case) |
| `color` | string | Character class, one of `ironclad`, `silent`, `defect`, `watcher`, plus `colorless` and `curse` (lowercase, as emitted by the upstream API) |
| `cost` | string | Energy cost (`"0"`, `"1"`, `"X"`, `"-"` for unplayable) |
| `description` | string | Card text, base form |
| `description_upgraded` | string | Card text after upgrade (`+`) |
| `keywords` | string (JSON list) | Declared keywords |
| `raw_json` | string (JSON object) | Full original payload |

### Derived feature columns

Numeric columns surface as `double` on the wire (pandas casts ints with nulls to float); semantically they're integer counts.

| Field | Type | Description |
| --- | --- | --- |
| `damage` | float \| null | Primary damage value (e.g. "Deal 6 damage" → 6) |
| `damage_upgraded` | float \| null | Same for the upgraded form |
| `block` | float \| null | Primary block value |
| `block_upgraded` | float \| null | Same for the upgraded form |
| `targets_all_enemies` | bool | True if the card hits ALL enemies (AOE) |
| `status_effects_applied` | string (JSON list) | `[{effect, count}]` for Vulnerable/Weak/Frail/Strength/Dexterity/Poison |
| `mechanics` | string (JSON list) | Mechanic keywords (declared + inferred from text) |

The schema also carries STS2-specific columns (`orbs_channeled`, `orbs_referenced`, `forge_value`, `souls_added`) for cross-game compatibility; they're all-null on STS1 rows.

## Card art

Each row carries the in-game card portrait as a PIL-decodable image. The HuggingFace dataset viewer renders thumbnails inline.

- **Source.** Card portraits are extracted directly from a local Steam install of Slay the Spire (`desktop-1.0.jar`). Game files are not redistributed by this pipeline; extraction happens on the maintainer's machine.
- **Resolution.** This snapshot ships the high-resolution variant (~1024×1024 from `images/1024Portraits/`). Downscale at use time with `image.resize(...)` if you need smaller.
- **Why this is large.** The cards Parquet jumps from ~200 KB to ~50 MB once portraits are inlined. If you don't need the bytes, use `streaming=True` or drop the `image` column after loading.
- **Attribution.** Slay the Spire art is © [Mega Crit](https://www.megacrit.com/). Mega Crit has publicly blessed redistribution of fan-extracted card art for tooling and mods (e.g. [Slay the Relics](https://github.com/Skrelpoid/SlayTheRelics) and broader STS1 modding ecosystem). This dataset includes a takedown clause regardless: if Mega Crit objects, the `image` column will be removed.

## Loading

```python
from datasets import load_dataset

cards = load_dataset("t22000t/slay-the-spire-1-cards", split="train")
print(cards[0]["name"], "→", cards[0]["image"])
# Strike → <PIL.PngImagePlugin.PngImageFile image mode=RGBA size=1024x1024>

cards[0]["image"].size      # (1024, 1024)
# cards[0]["image"].show()  # opens the portrait in your default viewer
```

To skip the image bytes (small download, drop the column):

```python
cards = load_dataset("t22000t/slay-the-spire-1-cards",
                     split="train", streaming=True)
```

## Joining with the embeddings dataset

```python
from datasets import load_dataset
import pandas as pd, numpy as np

cards = load_dataset("t22000t/slay-the-spire-1-cards", split="train").to_pandas()
embs  = load_dataset("t22000t/slay-the-spire-1-card-embeddings", split="train").to_pandas()

# Join on id
df = cards.merge(embs[["id", "embedding"]], on="id", how="inner")
print(len(df), "cards with embeddings")
```

## Source Data

Card definitions are parsed directly from the Slay the Spire game files by the [`nkhoit/spire-archive`](https://github.com/nkhoit/spire-archive) project. The pipeline in [`sts-cards`](https://github.com/timothy22000/slaythespire-codex) calls `spire-archive`'s public JSON API, normalizes the heterogeneous fields into a stable schema, and extracts the derived feature columns via regex over the description text.

## Considerations for Using the Data

### Discussion of Biases

The derived feature columns are extracted via best-effort regex. Cards with non-standard description phrasing may have null values where humans would extract a number. The full original text is always available in the `description` column for downstream re-extraction.

### Other Known Limitations

- **English only** in this snapshot. The upstream pipeline supports 13 languages but multilingual data is not currently shipped.
- **Text + portrait art only**, no audio, animations, or other media.
- STS1 has been stable since its 1.0 release, so drift is minimal, but check `provenance.json` for the fetch date if you need certainty.

## Provenance

A `provenance.json` ships alongside the data file documenting the exact fetch source and timestamp used to produce this snapshot.

## Citation

```bibtex
@dataset{sts1_cards_dataset,
  title = {Slay the Spire 1: Cards},
  author = {timothy22000},
  year = {2026},
  url = {https://huggingface.co/datasets/t22000t/slay-the-spire-1-cards},
  note = {Card data via nkhoit/spire-archive; game IP © Mega Crit}
}
```

## Licensing

- **Dataset (this repository):** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Pipeline code:** MIT, see [github.com/timothy22000/slaythespire-codex](https://github.com/timothy22000/slaythespire-codex)
- **Game IP:** Slay the Spire is © [Mega Crit](https://www.megacrit.com/). This dataset ships factual reference data plus extracted card portraits for tooling and modding use; Mega Crit has publicly blessed redistribution of fan-extracted card art. If Mega Crit objects, the `image` column will be removed.
