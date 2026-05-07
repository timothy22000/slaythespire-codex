---
license: cc-by-4.0
language:
- en
pretty_name: "Slay the Spire 2: Cards"
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
- slay-the-spire-2
- roguelike
- early-access
configs:
- config_name: default
  data_files: cards.parquet
---

# Slay the Spire 2: Cards

A normalized dataset of every card in **Slay the Spire 2** (Early Access), with derived feature columns including STS2-specific Orb, Forge, and Soul mechanics. **Collected for ML/DL training:** load with `datasets.load_dataset(...)` and feed straight into a card-text classifier, deckbuilder simulator, or design-analysis model.

This is the **cards** dataset. For text embeddings of these cards, see the companion dataset:
**[`t22000t/slay-the-spire-2-card-embeddings`](https://huggingface.co/datasets/t22000t/slay-the-spire-2-card-embeddings)**, joinable to this dataset by `id`.

For Slay the Spire 1, see **[`t22000t/slay-the-spire-1-cards`](https://huggingface.co/datasets/t22000t/slay-the-spire-1-cards)**.

> ⚠️ **Early Access, content is unstable.** Slay the Spire 2 entered Early Access on March 5, 2026. Cards are added, removed, and rebalanced regularly. **Always check the `sts_game_version` field in `provenance.json`** before drawing comparisons across snapshots.

## Dataset Description

- **Repository:** https://github.com/timothy22000/slaythespire-codex
- **Source data:** Parsed from game files by [`nkhoit/spire-archive`](https://github.com/nkhoit/spire-archive)
- **Card count:** ~576 (varies by patch)
- **Languages:** English
- **License:** CC BY 4.0; game IP belongs to Mega Crit

### Why split from embeddings?

Card-text classification, design analysis, modding, and deckbuilder simulators don't need the embedding column. Splitting also lets the embedding repo be re-versioned independently when the embedding model changes, without touching the cards repo's commit history. Both repos are joinable on `id`.

## Data Fields

### Core columns

| Field | Type | Description |
| --- | --- | --- |
| `game` | string | Always `"sts2"` |
| `id` | string | Stable card identifier, **the join key to the embeddings dataset** |
| `name` | string | Display name |
| `type` | string | One of `Attack`, `Skill`, `Power`, `Status`, `Curse`, `Quest` |
| `rarity` | string | One of `Basic`, `Common`, `Uncommon`, `Rare`, `Special`, `Curse`, `Status`, `Ancient`, `Event`, `Quest`, `Token` (Title Case) |
| `color` | string | Character class or pseudo-class (lowercase, as emitted by the upstream API): `ironclad`, `silent`, `defect`, `necrobinder`, `regent`, `colorless`, `curse`, `event`, `quest`, `status`, `token`. New character classes may appear with future patches |
| `cost` | string | Energy cost (`"0"`, `"1"`, `"X"`, `"-"` for unplayable) |
| `description` | string | Card text, base form |
| `description_upgraded` | string | Card text after upgrade (`+`) |
| `keywords` | string (JSON list) | Declared keywords |
| `raw_json` | string (JSON object) | Full original payload (preserves Orbs/Forge/Souls/Enchantments structure) |

### Generic feature columns

| Field | Type | Description |
| --- | --- | --- |
| `damage` | int \| null | Primary damage value (e.g. "Deal 6 damage" → 6) |
| `damage_upgraded` | int \| null | Same for the upgraded form |
| `block` | int \| null | Primary block value |
| `block_upgraded` | int \| null | Same for the upgraded form |
| `targets_all_enemies` | bool | True if the card hits ALL enemies (AOE) |
| `status_effects_applied` | string (JSON list) | `[{effect, count}]` for Vulnerable/Weak/Frail/etc. |
| `mechanics` | string (JSON list) | Mechanic keywords (declared + inferred from text) |

### STS2-specific feature columns

| Field | Type | Description |
| --- | --- | --- |
| `orbs_channeled` | string (JSON list) | `[{type, count}]` for Orbs the card channels |
| `orbs_referenced` | string (JSON list) | Orb types mentioned in the description, channeled or not |
| `forge_value` | int \| null | Forge value if the card has a "Forge N" clause |
| `souls_added` | int \| null | Number of Souls added (Necrobinder mechanic) |

## Loading

```python
from datasets import load_dataset

cards = load_dataset("t22000t/slay-the-spire-2-cards", split="train")
```

## Filtering on STS2 mechanics

```python
import json
import pandas as pd

df = cards.to_pandas()

# All cards that channel a Lightning orb
def channels_lightning(row):
    return any(o["type"] == "Lightning" for o in json.loads(row))

mask = df["orbs_channeled"].apply(channels_lightning)
print(df.loc[mask, ["name", "description"]])
```

## Joining with the embeddings dataset

```python
from datasets import load_dataset

cards = load_dataset("t22000t/slay-the-spire-2-cards", split="train").to_pandas()
embs  = load_dataset("t22000t/slay-the-spire-2-card-embeddings", split="train").to_pandas()

df = cards.merge(embs[["id", "embedding"]], on="id", how="inner")
```

## Considerations for Using the Data

### Early Access Drift, the most important caveat

STS2 cards change frequently. A snapshot from one week may contain cards that no longer exist, missing cards added since, or rebalanced versions of existing cards. **Always read `provenance.json` to know which patch this snapshot reflects.** Don't compare numerical analyses across snapshots without aligning on game version.

### Discussion of Biases

The derived feature columns are extracted via best-effort regex tuned for English STS2 phrasing. STS2-specific mechanics (Orbs, Forge, Souls) are extracted via name lookups against an explicit allowlist; new mechanics introduced in future patches won't appear in derived columns until the regex is updated. The full original text is always available in `description` and `raw_json` for re-extraction.

### Other Known Limitations

- **English only** in this snapshot.
- **Card text only**, no card art, audio, or other media.
- **Patch drift**, see above.

## Provenance

The `provenance.json` records:
- `sts_game_version`: the STS2 Steam version at fetch time
- `source_fetched_at`: UTC timestamp
- model and pipeline metadata

## Citation

```bibtex
@dataset{sts2_cards_dataset,
  title = {Slay the Spire 2: Cards},
  author = {timothy22000},
  year = {2026},
  url = {https://huggingface.co/datasets/t22000t/slay-the-spire-2-cards},
  note = {Early Access snapshot; card data via nkhoit/spire-archive; game IP © Mega Crit}
}
```

## Licensing

- **Dataset:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Pipeline code:** MIT, see [github.com/timothy22000/slaythespire-codex](https://github.com/timothy22000/slaythespire-codex)
- **Game IP:** Slay the Spire 2 is © [Mega Crit](https://www.megacrit.com/).
