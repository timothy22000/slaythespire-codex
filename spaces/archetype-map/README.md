---
title: Slay the Spire Archetype Map
emoji: 🗺
colorFrom: red
colorTo: indigo
sdk: gradio
sdk_version: "6.14.0"
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Interactive UMAP map of every card in Slay the Spire 1 + 2.
---

# Slay the Spire Archetype Map

Interactive UMAP scatter of every card in **Slay the Spire 1 + 2**, projected from 1024-D Qwen3 text embeddings. Cards close together in the plot play similarly. Pick a card from the dropdown to see its 5 nearest neighbors (computed in the full embedding space, not the 2D projection).

Each game has its own coordinate system, so STS1 and STS2 are shown on separate tabs.

## Data sources

- [t22000t/slay-the-spire-1-cards](https://huggingface.co/datasets/t22000t/slay-the-spire-1-cards)
- [t22000t/slay-the-spire-1-card-embeddings](https://huggingface.co/datasets/t22000t/slay-the-spire-1-card-embeddings)
- [t22000t/slay-the-spire-2-cards](https://huggingface.co/datasets/t22000t/slay-the-spire-2-cards)
- [t22000t/slay-the-spire-2-card-embeddings](https://huggingface.co/datasets/t22000t/slay-the-spire-2-card-embeddings)

## Pipeline

Code lives at [github.com/timothy22000/slaythespire-codex](https://github.com/timothy22000/slaythespire-codex).
