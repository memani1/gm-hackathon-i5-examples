# Site settings shared by setup.sh and submit.sh. Override any of these by
# exporting them before running either script.

# Checkout of the training code (created by setup.sh)
export SC25_DIR=${SC25_DIR:-$SCRATCH/amsc-atscale-examples/sc25-dl-tutorial}

# ERA5 tutorial data; mounted at /data inside the container
export DATADIR=${DATADIR:-/pscratch/sd/s/shas1693/data/sc25_tutorial_data}

# Training outputs (TensorBoard logs, checkpoints); mounted at /logs inside the container
export LOGDIR=${LOGDIR:-$SCRATCH/amsc-atscale-examples/vit-logs}
