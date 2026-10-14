#!/bin/bash
#SBATCH -J vit-era5
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 4
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-node 4
#SBATCH -t 00:30:00
#SBATCH --image=nersc/pytorch:25.06.01
#SBATCH --module=gpu,nccl-plugin
#SBATCH -o %x-%j.out

# ==============================================================================
# SCRIPT: ViT training on ERA5 (Perlmutter)
# DESCRIPTION: Runs one config from config.yaml with Shifter. Submit from this
#              directory after running setup.sh.
# USAGE:       sbatch [sbatch options] submit.sh <dp|mp> [extra train args]
#   sbatch submit.sh dp
#   sbatch submit.sh mp
#   sbatch submit.sh mp --tensor_parallel=4 --context_parallel=1
# ==============================================================================

set -euo pipefail

CONFIG=${1:?usage: sbatch submit.sh <dp|mp> [extra train args]}
shift

EXAMPLE_DIR=$SLURM_SUBMIT_DIR
source "$EXAMPLE_DIR/env.sh"

# Model-parallel configs use train_mp.py; default to 4-way context parallelism.
# Later args win, so layout flags passed on the command line override these.
case "$CONFIG" in
    mp*)
        SCRIPT=train_mp.py
        ARGS=(--tensor_parallel=1 --context_parallel=4 --parallel_order=cp-tp-dp "$@")
        ;;
    *)
        SCRIPT=train.py
        ARGS=("$@")
        ;;
esac

mkdir -p "$LOGDIR"
echo "config=$CONFIG script=$SCRIPT sc25-dl-tutorial=$(git -C "$SC25_DIR" rev-parse --short HEAD)"

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export HDF5_USE_FILE_LOCKING=FALSE
export FI_MR_CACHE_MONITOR=userfaultfd

# Reversing order of GPUs to match default CPU affinities from Slurm
export CUDA_VISIBLE_DEVICES=3,2,1,0

set -x
srun -u shifter -V "$DATADIR:/data" -V "$LOGDIR:/logs" \
    bash -c 'export RANK=$SLURM_PROCID LOCAL_RANK=$SLURM_LOCALID WORLD_SIZE=$SLURM_NTASKS
             exec python "$@"' _ \
    "$SC25_DIR/$SCRIPT" \
    --yaml_config="$EXAMPLE_DIR/config.yaml" \
    --config="$CONFIG" \
    "${ARGS[@]}"
