#!/bin/bash
# ==============================================================================
# SCRIPT: Set up the ViT training example (Perlmutter)
# DESCRIPTION: Clones the sc25-dl-tutorial training code at the pinned commit
#              these examples are tested with, and creates the log directory.
#              Safe to re-run.
# ==============================================================================

set -euo pipefail

# Device-portable branch of NERSC/sc25-dl-tutorial (upstream PR #4,
# branch sunspot-xpu-support). Same commit on every system.
REPO_URL=https://github.com/saforem2/sc25-dl-tutorial.git
COMMIT=7e91316d7af4af0f51bbfc7e9e7194418ae454ee

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

if [[ ! -d "$SC25_DIR/.git" ]]; then
    git clone --quiet --filter=blob:none "$REPO_URL" "$SC25_DIR"
fi

if [[ -n "$(git -C "$SC25_DIR" status --porcelain --untracked-files=no)" ]]; then
    echo "error: $SC25_DIR has local changes; not switching commits" >&2
    exit 1
fi

git -C "$SC25_DIR" fetch --quiet origin "$COMMIT"
git -C "$SC25_DIR" -c advice.detachedHead=false checkout --quiet "$COMMIT"

mkdir -p "$LOGDIR"

echo "sc25-dl-tutorial at $(git -C "$SC25_DIR" rev-parse --short HEAD) in $SC25_DIR"
echo "logs will be written to $LOGDIR"
