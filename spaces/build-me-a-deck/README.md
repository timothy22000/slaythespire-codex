---
title: Slay the Spire Build Me a Deck
emoji: 🃏
colorFrom: pink
colorTo: indigo
sdk: gradio
sdk_version: "6.14.0"
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Describe a Slay the Spire playstyle, get a deck.
---

# Slay the Spire: Build Me a Deck

Describe a Slay the Spire playstyle in plain English and get back a 20-card deck whose cards are scored against the prompt. The algorithm:

1. Encodes your prompt with the same `Qwen/Qwen3-Embedding-0.6B` model used for the indexed cards (so similarity scores live in the same vector space).
2. Filters the candidate pool to the chosen character's drafted cards plus colorless.
3. Optionally locks the character's starter deck into the first slots.
4. Greedily picks remaining cards by descending cosine similarity, with feasibility checks against attack/skill/power and mana-curve targets so the algorithm doesn't dead-end.
5. Returns a card grid with similarity scores, a fit-summary banner with type ratio + curve histogram + top theme keywords, and an honesty layer that flags weak prompts with diagnostic copy.

Greedy similarity-based selection, not an optimization solver. Decks are aspirational, actual STS runs build decks card-by-card from card-reward draws.

## Data sources

- [t22000t/slay-the-spire-1-cards](https://huggingface.co/datasets/t22000t/slay-the-spire-1-cards)
- [t22000t/slay-the-spire-1-card-embeddings](https://huggingface.co/datasets/t22000t/slay-the-spire-1-card-embeddings)
- [t22000t/slay-the-spire-2-cards](https://huggingface.co/datasets/t22000t/slay-the-spire-2-cards)
- [t22000t/slay-the-spire-2-card-embeddings](https://huggingface.co/datasets/t22000t/slay-the-spire-2-card-embeddings)

## Pipeline

Code lives at [github.com/timothy22000/slaythespire-codex](https://github.com/timothy22000/slaythespire-codex).
