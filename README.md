# Genesis hackathon I5: ViT training examples

Examples for the Genesis hackathon session **I5. Getting started with pretraining
foundation models on DOE supercomputers**.

They train a Vision Transformer (ViT) on ERA5 weather data on three DOE
supercomputers, using the code from the
[SC25 deep learning at scale tutorial](https://github.com/NERSC/sc25-dl-tutorial).

Each system has its own folder with setup, configs, a batch script, and a README.
Start with the README for the system you're using.

| System | Folder | Configs | Notes |
|---|---|---|---|
| Perlmutter (NERSC) | [`perlmutter`](perlmutter) | `dp` (data parallel), `mp` (model parallel) | Slurm + Shifter |
| Aurora (ALCF) | [`aurora`](aurora) | `smoke`, `dp_*`, `mp_*` | PBS + [ezpz](https://github.com/saforem2/ezpz); `mp_*` not yet benchmarked |
| Frontier (OLCF) | [`frontier`](frontier) | `dp` | Slurm, 1 node |

## Hackathon reservations

**Perlmutter:** use account `m5429_g` and reservation `gm_hackathon26_session5`
(2026-10-15, 13:30–15:00 Pacific). From `perlmutter`:

```bash
./setup.sh
sbatch -A m5429_g --reservation=gm_hackathon26_session5 submit.sh dp
sbatch -A m5429_g --reservation=gm_hackathon26_session5 submit.sh mp
```

**Aurora:** see the Quick start in [`aurora/README.md`](aurora/README.md).

## Training code

All three systems use commit `7e91316` of the `sunspot-xpu-support` branch of
[saforem2/sc25-dl-tutorial](https://github.com/saforem2/sc25-dl-tutorial), which makes
the tutorial's `train.py` run on NVIDIA, AMD, and Intel GPUs. The setup steps clone it
for you. On Aurora, `setup.sh` also applies two patches on top of it: one keeps
DataLoader workers alive between epochs, and one ports the model-parallel
`train_mp.py` to Intel GPUs.

## Sources and credits

The folders are copied from the `vit-training/` folders of the per-system AmSC example repositories.
The only edits are two path references in `aurora/` (a comment in `ViT_aurora.yaml` and an
error message in `submit_aurora.sh`), updated for this repo's layout.

| Folder | Source | Commit | Authors |
|---|---|---|---|
| `perlmutter` | `amsc2/users/ai/perlmutter` (`main`) | `1e24b3b` | Steve Farrell |
| `aurora` | `amsc2/users/ai/aurora` (`vit-training-xpu-mp`) | `b04e123` | Kyle Felker, Murali Emani |
| `frontier` | `amsc2/users/ai/frontier` (`main`) | `3330b41` | Aristeidis Tsaris |

The training code is from the SC25 tutorial by NERSC, with device portability by
Sam Foreman.
