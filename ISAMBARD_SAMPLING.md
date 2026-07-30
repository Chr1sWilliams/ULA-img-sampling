# Isambard day-095 sampling

`slurm/run_eht_095.sh` submits a two-task Slurm array:

- task `0`: `SR1_M87_2017_095_hi_hops_netcal_StokesI.uvfits`
- task `1`: `SR1_M87_2017_095_lo_hops_netcal_StokesI.uvfits`

Each task uses one GPU, retains every final sample from every schedule round,
and records the schedule, energy, and path-length progression locally and in
Weights & Biases.

## One-time setup

Use the conda environment described for this repository and add W&B:

```bash
conda activate "$PROJECTDIR/$USER/conda-envs/ula-prior"
python -m pip install wandb
wandb login
```

The launcher defaults to that environment. If yours is elsewhere, export its
path as `CONDA_ENV`. If Miniforge is installed elsewhere, also set `CONDA_SH`
to its `etc/profile.d/conda.sh`.

## Submit both datasets

Run this from the repository root:

```bash
mkdir -p slurm/logs
sbatch slurm/run_eht_095.sh
```

If your allocation requires an account, add it at submission time:

```bash
sbatch --account=YOUR_ACCOUNT slurm/run_eht_095.sh
```

The default W&B mode is `online`. Authentication is taken from `wandb login`
or `WANDB_API_KEY`. A missing login, missing CUDA device, missing model, or
missing input file causes the job to fail early.

## Common overrides

All main run settings can be adjusted without editing the launcher:

```bash
NUM_ROUNDS=6 \
NUM_SAMPLES=64 \
LOG_INTERVAL=50 \
WANDB_PROJECT=bh_sampling \
WANDB_ENTITY=YOUR_ENTITY \
sbatch slurm/run_eht_095.sh
```

Available settings include `NUM_SCHEDULE_POINTS`, `NUM_ROUNDS`,
`CORRECTOR_STEPS`, `NUM_SAMPLES`, `INITIAL_SIGMA`,
`STEP_TAIL_PROBABILITY`, `LOG_INTERVAL`, `SEED`, `OUTPUT_ROOT`,
`WANDB_MODE`, `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_GROUP`, and
`WANDB_TAGS`.

To use an estimated consistency cutoff:

```bash
CONSISTENCY_RESULT=/path/to/consistency/output \
sbatch slurm/run_eht_095.sh
```

Alternatively, set `GUIDANCE_THRESHOLD` directly. Do not set both.

## Recorded results

By default, results are kept under:

```text
$PROJECTDIR/$USER/ula-img-sampling-runs/eht-095/<run-name>/
```

Each round contains:

- the complete `samples.pt` tensor and sample preview;
- pixelwise sample mean and standard deviation;
- the schedule before and after optimization;
- energy and length increments;
- cumulative length, diffusion time, guidance, and Langevin-speed arrays;
- a compressed diagnostics archive and a JSON summary.

The run root contains the full schedule history, a `round_progress.csv`,
`run_history.npz`, the latest learned schedule, and plots of schedule,
energy, length, and schedule-change progression.

W&B receives live likelihood, score, step-size, sample-range, energy, and GPU
memory diagnostics. It also receives per-round samples and plots, aggregate
progression plots, and—after a successful run—a `sampling-run` artifact
containing the retained local results.
