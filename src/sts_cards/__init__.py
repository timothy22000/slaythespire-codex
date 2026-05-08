"""sts-cards: Slay the Spire 1 + 2 card dataset and embedding pipeline."""

__version__ = "0.5.0"

GAMES = ("sts1", "sts2")

# Each game ships as up to THREE HuggingFace datasets:
#   - "cards": metadata + derived features + (optionally) inline card art
#   - "embeddings": text embedding vectors + UMAP coords
#   - "multimodal-embeddings": joint text+image embeddings (Qwen3-VL)
# All joinable to the cards repo by `id`.
REPO_KINDS = ("cards", "embeddings", "multimodal-embeddings")

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_TASK_INSTRUCTION = (
    "Represent this Slay the Spire card so that mechanically similar cards "
    "(same archetype, comparable damage/block patterns, related keywords) "
    "are close in embedding space."
)

DEFAULT_MULTIMODAL_MODEL = "Qwen/Qwen3-VL-Embedding-2B"
DEFAULT_MULTIMODAL_DIM = 1024
DEFAULT_MULTIMODAL_TASK_INSTRUCTION = (
    "Represent this Slay the Spire card so that mechanically similar cards "
    "(same archetype, comparable damage/block patterns, related keywords) "
    "are close in embedding space, using the card's text mechanics as the "
    "primary signal and the portrait art as a secondary cue for "
    "character/class and visual archetype."
)
