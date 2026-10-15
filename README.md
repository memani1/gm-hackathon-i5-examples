# Genesis hackathon I5: ViT training examples

Examples for the Genesis hackathon session **I5: Getting started with pretraining
foundation models on DOE supercomputers**.

Here we train a Vision Transformer (ViT) on ERA5 weather data on three DOE
supercomputers, using the code from the
[SC25 deep learning at scale tutorial](https://github.com/NERSC/sc25-dl-tutorial).

Each system has its own folder with setup, configs, a batch script, and a README.
Start with the README for the system you're using.

| System | Folder  | Notes |
|---|---|---|
| Perlmutter (NERSC) | [`perlmutter`](perlmutter)  | Slurm + Shifter |
| Aurora (ALCF) | [`aurora`](aurora)  | PBS + [ezpz](https://github.com/saforem2/ezpz);|
| Frontier (OLCF) | [`frontier`](frontier) | Slurm |

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

## Credits

- Perlmutter example: Steve Farrell
- Aurora example: Kyle Felker, Murali Emani
- Frontier example: Aristeidis Tsaris
- Training code: the [SC25 deep learning at scale tutorial](https://github.com/NERSC/sc25-dl-tutorial)
  by NERSC, with device portability by Sam Foreman
