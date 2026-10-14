# ViT training on Perlmutter

Trains a Vision Transformer on ERA5 weather data using the code from the
[SC25 deep learning at scale tutorial](https://github.com/NERSC/sc25-dl-tutorial),
in data-parallel (`dp`) or model-parallel (`mp`) mode.

| File | Purpose |
|---|---|
| `setup.sh` | Clones the training code at the pinned commit. Safe to re-run. |
| `env.sh` | Paths for the code checkout, data, and logs. Export a variable to override it. |
| `config.yaml` | The `dp` and `mp` training configs. |
| `submit.sh` | Slurm batch script; runs one config with Shifter. |

## Quick start

```bash
./setup.sh
sbatch -A <account> submit.sh dp
sbatch -A <account> submit.sh mp
```

Submit from this directory. Output goes to `vit-era5-<jobid>.out` here, and
TensorBoard logs go to `$LOGDIR/<config>/<N>GPU/00/`.

## Configs

Both default to 4 nodes (16 A100s) with 16 samples per data-parallel rank, and train
on the full 37,960-sample training set.

| Config | Script | Model | Global batch | Parallelism | Epochs (iterations) | Throughput | Job time |
|---|---|---|---|---|---|---|---|
| `dp` | `train.py` | ViT, embed dim 384, 12 layers | 256 | 16-way data parallel | 4 (592) | ~760 samples/s | ~7 min |
| `mp` | `train_mp.py` | ViT, embed dim 1024, 12 layers | 64 | 4-way context parallel × 4-way data parallel | 2 (1186) | ~167 samples/s | ~11 min |

Throughput and job times were measured on 2026-10-09; job time includes about
2 minutes of startup.

Training runs whole epochs only: `num_iters` is rounded down to a multiple of the
epoch length, so a value below one epoch trains nothing. The first epoch includes
compilation and warmup; use epochs 2 onward for throughput.

Both use the DALI data loader, fp16 mixed precision, and fused Adam.
They match the upstream `bs256_opt` and `mp` configs except for `num_iters`
(upstream uses 30000, which in practice runs until the job time limit).

## Variations

Extra arguments after the config name are passed to the training script:

```bash
# Quick check: one epoch
sbatch -q debug submit.sh dp --num_iters=148

# Tensor parallelism instead of context parallelism
sbatch submit.sh mp --tensor_parallel=4 --context_parallel=1

# Fewer nodes with the same per-GPU batch size
sbatch -N 1 submit.sh dp --local_batch_size=16
```

`--local_batch_size` keeps 16 samples per GPU at any node count, but the learning
rate stays at the 16-GPU value.

## Training code version

`setup.sh` checks out commit `7e91316` of the `sunspot-xpu-support` branch of
[saforem2/sc25-dl-tutorial](https://github.com/saforem2/sc25-dl-tutorial)
(upstream PR #4). That branch makes `train.py` run on NVIDIA, AMD, and Intel GPUs,
so every system in these examples uses the same commit. Each job prints the commit
it ran in its output file.
