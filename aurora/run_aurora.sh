#!/bin/bash
# ==============================================================================
# SCRIPT: Launch one ViT training config on Aurora with ezpz
# USAGE:  bash run_aurora.sh <config> [extra train_mp.py args]
#   Runs train_mp.py (tensor/context/data parallel; tp, cp from the config or
#   --tensor_parallel / --context_parallel). With tp = cp = 1 it is plain data
#   parallel, so the dp_* configs work too. TRAIN_SCRIPT=train.py for the
#   original data-parallel-only script.
#   Run on a compute node with the frameworks module and venv active
#   (submit_aurora.sh does this; see README.md for interactive use).
# ==============================================================================

set -euo pipefail

CONFIG="${1:-smoke}"
EXTRA_ARGS="${*:2}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train_mp.py}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

# HDF5 over Lustre needs file locking disabled
export HDF5_USE_FILE_LOCKING=FALSE
export CCL_BACKEND=native
export CCL_KVS_MODE=mpi
# PBS sets a long per-job TMPDIR; DataLoader workers bind AF_UNIX sockets there
# and fail with "AF_UNIX path too long" (108-char limit)
export TMPDIR=/tmp

LOG_TS="$(date +%Y%m%d-%H%M%S)"
RUNDIR="$LOGDIR/$CONFIG-$LOG_TS"
mkdir -p "$RUNDIR"
LOGFILE="$RUNDIR/run.log"

# The training script takes paths only from the YAML, so run from a copy with $DATADIR and
# $LOGDIR filled in (also keeps the exact config next to run.log)
YAML_CONFIG="$RUNDIR/ViT_aurora.yaml"
python3 -c 'import os, sys; sys.stdout.write(os.path.expandvars(open(sys.argv[1]).read()))' \
    "$SCRIPT_DIR/ViT_aurora.yaml" > "$YAML_CONFIG"

cd "$SC25_DIR"
if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    echo "error: $SC25_DIR/$TRAIN_SCRIPT not found" >&2
    exit 1
fi
echo "[$(date)] config=$CONFIG script=$TRAIN_SCRIPT nodes=$(sort -u "${PBS_NODEFILE:-/dev/null}" | wc -l)" \
    "sc25-dl-tutorial=$(git rev-parse --short HEAD) log=$LOGFILE" | tee -a "$LOGFILE"

ezpz launch -- python3 "$TRAIN_SCRIPT" \
    --yaml_config="$YAML_CONFIG" \
    --config="$CONFIG" \
    --run_num="$LOG_TS" \
    ${EXTRA_ARGS} 2>&1 | tee -a "$LOGFILE"
