---
title: Slay the Spire Synergy Inspector
emoji: 🔍
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: "6.14.0"
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Custom Slay the Spire card design assistant.
---

# Slay the Spire Synergy Inspector

Designing a custom card for **Slay the Spire 1 or 2**? Drop your card spec into this Space and see:

- **Tiered similarity verdict** against every card in the chosen game (identical-ish / very similar / similar / novel) with cosine scores
- **Top-10 closest existing cards** by embedding similarity, with a join back to full card metadata
- **Statistical outlier check** on damage and block: per-(type, cost) percentile baselines from the indexed corpus, with a fallback to type-only buckets when the (type, cost) bucket is too sparse

Encodes user-submitted cards through `Qwen/Qwen3-Embedding-0.6B` (the same model used to produce the indexed embeddings), so query vectors live in the same space as the corpus. The model is pre-warmed at startup; first analyze takes ~30-60s on a cold Space, subsequent ones are ~2s.

## Data sources

- [t22000t/slay-the-spire-1-cards](https://huggingface.co/datasets/t22000t/slay-the-spire-1-cards)
- [t22000t/slay-the-spire-1-card-embeddings](https://huggingface.co/datasets/t22000t/slay-the-spire-1-card-embeddings)
- [t22000t/slay-the-spire-2-cards](https://huggingface.co/datasets/t22000t/slay-the-spire-2-cards)
- [t22000t/slay-the-spire-2-card-embeddings](https://huggingface.co/datasets/t22000t/slay-the-spire-2-card-embeddings)

## Pipeline

Code lives at [github.com/timothy22000/slaythespire-codex](https://github.com/timothy22000/slaythespire-codex).
