#!/bin/bash
#SBATCH -J interact
#SBATCH -t 0:30:00
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/test-%x-%j.out

source /lustre/orion/stf218/world-shared/atsaris/ENV/source_env.sh

CONFIG=${1:-dp}
shift

case "$CONFIG" in
    *)
        SCRIPT=train.py
        ARGS=("$@")
        ;;
esac

CMD="time python sc25-dl-tutorial/$SCRIPT \
          --yaml_config config.yaml \
          --data_loader_config pytorch \
          --config $CONFIG \
          ${ARGS[*]}
"

mkdir -p logs

echo $CMD

HOME=/tmp srun \
          --nodes=${SLURM_NNODES} \
          --ntasks=$((SLURM_NNODES*8)) \
          --ntasks-per-node=8 \
          -c7 \
	  bash -c "source env.sh; $CMD"

