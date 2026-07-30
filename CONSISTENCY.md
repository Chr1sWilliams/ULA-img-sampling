# Estimating the likelihood-guidance consistency threshold

`estimate_consistency.py` records the relative likelihood-score error

```text
eta(s)^2 =
    sum_r ||grad log L(y | X_0^r) - grad log L(y | Xhat_0^r)||^2
    ----------------------------------------------------------------
                    sum_r ||grad log L(y | X_0^r)||^2
```

along the unconditional forward-diffusion path. For each schedule value it:

1. draws representative clean images `X_0` from `DATA_DIR`;
2. draws `X_s = (1-s) X_0 + sqrt(1-(1-s)^2) epsilon`;
3. evaluates the unconditional prior score at `X_s`;
4. forms the Tweedie estimate
   `Xhat_0 = (X_s + (1-(1-s)^2) score(X_s)) / (1-s)`;
5. compares the likelihood gradients at `X_0` and `Xhat_0`.

The stored schedule is ordered from low noise to high noise. Threshold selection
scans it in the reverse direction and chooses the first level satisfying
`eta(s) <= eta_tolerance`. This is the largest consistent `s`, which is the
cutoff needed by the existing gamma schedule.

## Run the diagnostic

Provide a directory of clean images representative of the prior's training
distribution:

```bash
DATA_DIR=/path/to/clean/images \
ETA_TOLERANCE=0.1 \
./launch_consistency_example.sh
```

The default likelihood is the built-in EHT interferometric likelihood. A custom
differentiable likelihood can be selected with:

```bash
DATA_DIR=/path/to/clean/images \
LIKELIHOOD_SPEC=my_module:log_likelihood \
./launch_consistency_example.sh
```

The likelihood must return one scalar per image. The constant `zero` likelihood
cannot be used because the denominator of the relative error is zero.

Useful controls:

- `NUM_GRID_POINTS`: number of recorded diffusion levels, default `50`
- `NUM_SAMPLES`: clean/noisy pairs per level, default `40`
- `BATCH_SIZE`: likelihood evaluation batch size, default `1`
- `ETA_TOLERANCE`: target relative error, default `0.1`
- `OUTPUT_DIR`: output directory, default `outputs/consistency`

## Recorded artifacts

The output directory contains:

- `consistency_path.csv`: human-readable eta path and supporting diagnostics
- `consistency_path.npz`: all recorded arrays
- `consistency_path.png`: eta versus `s`, tolerance, and selected cutoff
- `summary.json`: selected threshold and run metadata
- `guidance_threshold.npy`: scalar cutoff, when a crossing exists
- `config.json`: resolved command configuration

The estimator also records the Tweedie reconstruction MSE and likelihood-score
RMS values to help distinguish reconstruction failure from a near-zero
likelihood gradient.

## Use the estimated threshold

Pass the result directory directly into conditional sampling:

```bash
CONSISTENCY_RESULT=outputs/consistency ./launch_example.sh
```

Equivalently:

```bash
python sample.py --consistency_result outputs/consistency [other options]
```

The existing linear gamma ramp is then non-zero only below the selected
schedule cutoff. If no consistency result is supplied, sampling retains the
original preprint cutoff `1-exp(-0.5)`.

A threshold can also be supplied manually:

```bash
GUIDANCE_THRESHOLD=0.25 ./launch_example.sh
```
