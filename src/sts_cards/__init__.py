"""sts-cards: Slay the Spire 1 + 2 card dataset and embedding pipeline."""

__version__ = "0.4.0"

GAMES = ("sts1", "sts2")

# Each game ships as TWO HuggingFace datasets:
#   - "cards": metadata + derived features (no embeddings)
#   - "embeddings": embedding vectors + UMAP coords, joinable to cards by `id`
REPO_KINDS = ("cards", "embeddings")

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_TASK_INSTRUCTION = (
    "Represent this Slay the Spire card so that mechanically similar cards "
    "(same archetype, comparable damage/block patterns, related keywords) "
    "are close in embedding space."
)
