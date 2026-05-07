#!/usr/bin/env bash
# Deploy the Archetype Map Space to HuggingFace.
#
# Source-of-truth files live in this directory + ../shared/. Deploy uses a
# separate working tree at $ROOT/.hf-deploy/archetype-map/ (gitignored) so
# the source dir stays free of HF git artifacts and the parent repo can
# track everything cleanly.
#
# Idempotent: re-running re-syncs and re-pushes.

set -euo pipefail

SPACE_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SPACE_DIR/../.." && pwd)"
SPACE_NAME="t22000t/slaythespire-archetype-map"
HF_REMOTE_URL="https://huggingface.co/spaces/$SPACE_NAME"

DEPLOY_DIR="$ROOT_DIR/.hf-deploy/archetype-map"
mkdir -p "$ROOT_DIR/.hf-deploy"

if [ ! -d "$DEPLOY_DIR/.git" ]; then
    echo "==> Cloning HF Space into $DEPLOY_DIR"
    git clone "$HF_REMOTE_URL" "$DEPLOY_DIR"
else
    echo "==> Pulling latest HF Space main into $DEPLOY_DIR"
    git -C "$DEPLOY_DIR" pull --ff-only origin main
fi

echo "==> Syncing source files to deploy tree"
# Copy source files (skip __pycache__, deploy.sh)
rsync -a --delete \
    --exclude='.git/' \
    --exclude='.gradio/' \
    --exclude='__pycache__/' \
    --exclude='deploy.sh' \
    --exclude='shared/' \
    "$SPACE_DIR/" "$DEPLOY_DIR/"

# Vendor shared/ from spaces/shared/
rm -rf "$DEPLOY_DIR/shared"
cp -R "$ROOT_DIR/spaces/shared" "$DEPLOY_DIR/shared"
# Strip __pycache__ that might have been copied
find "$DEPLOY_DIR/shared" -type d -name __pycache__ -exec rm -rf {} +

cd "$DEPLOY_DIR"
git add -A
if git diff --cached --quiet; then
    echo "No changes to deploy."
    exit 0
fi
git commit -m "Update archetype-map Space"
git push origin main
echo "==> Deployed: https://huggingface.co/spaces/$SPACE_NAME"
