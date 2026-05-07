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
---

# Slay the Spire 1: Cards

A normalized dataset of every card in **Slay the Spire** (the original 2019 release), with derived feature columns for damage, block, status effects, and mechanics. **Collected for ML/DL training:** load with `datasets.load_dataset(...)` and feed straight into a card-text classifier, deckbuilder simulator, or design-analysis model.

This is the **cards** dataset. For text embeddings of these cards, see the companion dataset:
**[`t22000t/slay-the-spire-1-card-embeddings`](https://huggingface.co/datasets/t22000t/slay-the-spire-1-card-embeddings)**, joinable to this dataset by `id`.

For Slay the Spire 2, see **[`t22000t/slay-the-spire-2-cards`](https://huggingface.co/datasets/t22000t/slay-the-spire-2-cards)**.

## Dataset Description

- **Repository:** https://github.com/timothy22000/slaythespire-codex
- **Source data:** Parsed from game files by [`nkhoit/spire-archive`](https://github.com/nkhoit/spire-archive)
- **Card count:** 360
- **Languages:** English
- **License:** CC BY 4.0 (this dataset's structure and curation); game IP belongs to Mega Crit

### Why split from embeddings?

People doing card-text classification, deckbuilder simulators, or design analysis don't need the embedding column, and shipping it would mean a 4MB file instead of 200KB. The split also lets the embedding repo re-version independently when the embedding model changes, without touching the cards repo's commit history.

## Data Fields

### Core columns

| Field | Type | Description |
| --- | --- | --- |
| `game` | string | Always `"sts1"` |
| `id` | string | Stable card identifier, **the join key to the embeddings dataset** |
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

| Field | Type | Description |
| --- | --- | --- |
| `damage` | int \| null | Primary damage value (e.g. "Deal 6 damage" → 6) |
| `damage_upgraded` | int \| null | Same for the upgraded form |
| `block` | int \| null | Primary block value |
| `block_upgraded` | int \| null | Same for the upgraded form |
| `targets_all_enemies` | bool | True if the card hits ALL enemies (AOE) |
| `status_effects_applied` | string (JSON list) | `[{effect, count}]` for Vulnerable/Weak/Frail/Strength/Dexterity/Poison |
| `mechanics` | string (JSON list) | Mechanic keywords (declared + inferred from text) |

The schema also includes STS2-specific columns (`orbs_channeled`, `orbs_referenced`, `forge_value`, `souls_added`) for cross-game compatibility, but they're empty/null for all STS1 cards.

## Loading

```python
from datasets import load_dataset

cards = load_dataset("t22000t/slay-the-spire-1-cards", split="train")
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
- **Card text only**, no card art, audio, or other media.
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
- **Game IP:** Slay the Spire is © [Mega Crit](https://www.megacrit.com/). This dataset contains factual reference data and includes no card art or proprietary creative assets. If Mega Crit objects to redistribution, the dataset will be taken down.
