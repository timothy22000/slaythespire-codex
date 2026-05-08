# Quickstart — running locally

A condensed setup guide. For full project docs see [`DOCUMENTATION.md`](DOCUMENTATION.md).

> Just want to play with the data without installing anything? The full bundle (4 datasets + 3 Spaces) lives at [**huggingface.co/collections/t22000t/slaythespire-codex**](https://huggingface.co/collections/t22000t/slaythespire-codex).

## 1. Set up a virtual environment

```bash
cd sts-cards
python -m venv .venv
source .venv/bin/activate     # on Windows: .venv\Scripts\activate
pip install --upgrade pip
```

## 2. Install the package

For just the test suite (fastest, no model deps):

```bash
pip install -e ".[dev]"
```

For everything (~2 GB on first run because of torch + the embedding model):

```bash
pip install -e ".[all]"
```

If you want a partial install — say, only fetch and embed without UMAP/upload:

```bash
pip install -e ".[embed]"
```

## 3. Verify the install

```bash
pytest                        # ~117 tests should pass in ~2s
sts-cards --help              # CLI help should render with 11 commands
sts-cards version             # → 0.5.0
```

## 4. Run the pipeline (no HF account needed)

```bash
sts-cards fetch sts1
sts-cards embed sts1                          # downloads Qwen3-Embedding-0.6B (~1.2GB) on first run
sts-cards visualize sts1                      # writes output/sts1_umap_2d.html
sts-cards search sts1 --card "Strike" --k 10
```

Outputs land in `./output/`. Open `output/sts1_umap_2d.html` in a browser to see the UMAP scatter.

## 5. Same for STS2

```bash
sts-cards fetch sts2 --sts-game-version v0.103.0
sts-cards embed sts2
sts-cards visualize sts2
```

> The `--sts-game-version` flag is recorded in `provenance.json`. Important for STS2 because it changes weekly with patches.

## 5b. Adding card art (optional)

The published cards Parquets can also carry the in-game card portraits
as a column. Extraction reads from a local Steam install — game files are
never re-uploaded by the pipeline. Run from the project root:

```bash
# STS1: needs only Python stdlib (zipfile reads desktop-1.0.jar)
sts-cards diagnose-art sts1            # report join rate first
sts-cards extract-art  sts1            # adds `image` column to sts1_cards.parquet

# STS2: needs GDRE Tools — https://github.com/bruvzg/gdsdecomp
brew install gdre_tools                # or use the release binary directly
sts-cards diagnose-art sts2 --gdre-tools-path /path/to/gdre_tools
sts-cards extract-art  sts2 --gdre-tools-path /path/to/gdre_tools
```

`extract-art` also updates `provenance.json` with an `art` block recording
the source-file SHA-256 and extraction timestamp. Run it AFTER `fetch` and
BEFORE `embed`. Embedding does not change — image data isn't part of
`card_text`.

## 5c. Multimodal embeddings (optional)

Joint text+image embeddings via Qwen3-VL-Embedding-2B. Local-only — the
2B model + image tensors don't fit on a free GitHub runner. Run AFTER
`extract-art` so the cards Parquet has portraits to encode. Cards
without art still get a vector via text-only encoding through the same
model, preserving the joint coordinate system.

```bash
sts-cards extract-art      sts1
sts-cards embed-multimodal sts1                   # ~5 min on Apple Silicon GPU
sts-cards croissant sts1 --kind multimodal-embeddings \
  --repo t22000t/slay-the-spire-1-card-multimodal-embeddings
sts-cards upload    sts1 --kind multimodal-embeddings \
  --repo t22000t/slay-the-spire-1-card-multimodal-embeddings
```

The output is `output/{game}_multimodal_embeddings.parquet` with columns
`id, game, name, card_text, has_image, multimodal_embedding (1024D)`.
Provenance is updated with a `multimodal_embed` block.

## 6. Cross-game similarity

After running both games:

```python
import pandas as pd, numpy as np

sts1 = pd.read_parquet("output/sts1_cards_with_embeddings.parquet")
sts2 = pd.read_parquet("output/sts2_cards_with_embeddings.parquet")

emb1 = np.vstack(sts1["embedding"])
emb2 = np.vstack(sts2["embedding"])

# Find the STS2 cards most similar to "Bash" from STS1
i = sts1.index[sts1["name"] == "Bash"][0]
sims = emb2 @ emb1[i]
top = np.argsort(-sims)[:10]
print(sts2.iloc[top][["name", "description"]])
```

## 7. Upload to your HuggingFace account (optional)

You'll need:
- A HuggingFace account
- A write-scoped token from https://huggingface.co/settings/tokens
- Run `huggingface-cli login` and paste the token

Then for each of the four datasets:

```bash
# Generate Croissant descriptors first
sts-cards croissant sts1 --kind cards --repo t22000t/slay-the-spire-1-cards
sts-cards croissant sts1 --kind embeddings --repo t22000t/slay-the-spire-1-card-embeddings
sts-cards croissant sts2 --kind cards --repo t22000t/slay-the-spire-2-cards
sts-cards croissant sts2 --kind embeddings --repo t22000t/slay-the-spire-2-card-embeddings

# Then upload
sts-cards upload sts1 --kind cards --repo t22000t/slay-the-spire-1-cards
sts-cards upload sts1 --kind embeddings --repo t22000t/slay-the-spire-1-card-embeddings
sts-cards upload sts2 --kind cards --repo t22000t/slay-the-spire-2-cards
sts-cards upload sts2 --kind embeddings --repo t22000t/slay-the-spire-2-card-embeddings
```

## 8. Run a Gradio Space locally (optional)

Three Gradio Spaces live in [`spaces/`](spaces/) and are deployed at:

- [t22000t/slaythespire-archetype-map](https://huggingface.co/spaces/t22000t/slaythespire-archetype-map) — interactive UMAP scatter with nearest-neighbor lookup
- [t22000t/slaythespire-synergy-inspector](https://huggingface.co/spaces/t22000t/slaythespire-synergy-inspector) — custom-card design assistant (form / JSON / CSV / text / screenshot input)
- [t22000t/slaythespire-build-me-a-deck](https://huggingface.co/spaces/t22000t/slaythespire-build-me-a-deck) — chat-first deck builder driven by playstyle prompts

To run one locally:

```bash
cd spaces/archetype-map      # or synergy-inspector / build-me-a-deck
pip install -r requirements.txt
python app.py                # serves on http://127.0.0.1:7860
```

Each Space pulls its data straight from the published HF datasets — no local fetch/embed run required. The `synergy-inspector` and `build-me-a-deck` Spaces additionally load `Qwen/Qwen3-Embedding-0.6B` (~1.2 GB) on first launch so user-submitted prompts/cards are encoded into the same vector space as the corpus.

To redeploy your fork to HuggingFace, run `./deploy.sh` from inside the Space directory. It uses a separate working tree under `.hf-deploy/` (gitignored) so the parent repo stays free of HF git artifacts.

## 9. Set up GitHub Actions (optional)

Push the repo to GitHub. Add `HF_TOKEN` as a repository secret in `Settings → Secrets and variables → Actions`. The workflows in `.github/workflows/`:

- **`ci.yml`** runs on every push/PR — lint and test on Python 3.10/3.11/3.12.
- **`refresh.yml`** is scheduled (STS2 weekly, STS1 monthly) and manually triggerable. Edit the `env` block at the top to set your repo names.

## Common issues

**`pip install -e .` fails with "License file does not exist"** — make sure `LICENSE` (no extension) is in the repo root, not `LICENSE.md` or `LICENSE.txt`.

**`pytest` reports "no module named sts_cards"** — make sure you ran `pip install -e .` (the editable install), not just `pip install -r requirements.txt`.

**First `sts-cards embed` is very slow** — that's the model download. Subsequent runs use the local cache in `~/.cache/huggingface/`.

**`sts-cards fetch sts2` returns 0 cards or fails** — the upstream `spire-archive.com` may be down or have changed its API shape. Check `https://spire-archive.com/api/sts2/cards?limit=5` in a browser. The fetch logic logs which URL it's hitting.

**HTTP cache returning stale data** — `sts-cards cache-clear` wipes the local cache, or pass `--no-cache` to a single fetch.

## Working with Cursor / Claude Code

A few things that make the package pleasant to extend:

- `src/sts_cards/normalize.py` and `src/sts_cards/features.py` are pure functions with no I/O. Trivial to add new feature extractors and unit-test them.
- `tests/conftest.py` provides reusable fixtures — `sample_sts1_card_payload`, `small_dataframe` (with pre-normalized embeddings) — that let new tests skip model loading entirely.
- The CLI is one Typer app in `src/sts_cards/cli.py`. Adding a new command is ~15 lines.
- The provenance contract (`src/sts_cards/provenance.py`) is the project's narrowest waist — every artifact must update it. Read this file first if you're modifying outputs.

For Claude Code specifically: pointing it at `DOCUMENTATION.md` first gives it the full mental model in one read. After that it can navigate the package without much hand-holding.
