# Training a diffusion prior

`image_train.py` trains the grayscale or RGB diffusion model used by
`conditional_diffusion.py`. Its default architecture matches the sampling code:
128 x 128 images, 64 base channels, and two residual blocks. Training uses a
cosine noise schedule and adaptive schedule learning by default.

## Dataset

Place PNG, JPG, JPEG, or GIF images in one directory. Subdirectories are scanned
recursively. Images are converted to the requested channel mode, resized,
center-cropped, and scaled to `[-1, 1]`.

Choose the image mode with `IMAGE_CHANNELS` in the launcher or
`--image_channels` on the Python command:

```bash
# Grayscale
IMAGE_CHANNELS=1 DATA_DIR=/path/to/images ./launch_training_example.sh

# RGB
IMAGE_CHANNELS=3 DATA_DIR=/path/to/images ./launch_training_example.sh
```

## Start training

The launcher requires `DATA_DIR` and writes checkpoints and schedule arrays to
`outputs/diffusion_prior` by default:

```bash
DATA_DIR=/path/to/training/images ./launch_training_example.sh
```

Useful overrides include:

```bash
DATA_DIR=/path/to/training/images \
OUTPUT_DIR=/path/to/checkpoints \
TRAIN_STEPS=300000 \
BATCH_SIZE=16 \
USE_FP16=true \
./launch_training_example.sh
```

The default is a cosine schedule with adaptive learning enabled. Both settings
are command-line options and launcher environment variables:

```bash
DATA_DIR=/path/to/training/images \
NOISE_SCHEDULE=cosine \
SCHEDULE_TUNE=true \
SCHEDULE_LR=0.01 \
./launch_training_example.sh
```

Disable schedule learning with `SCHEDULE_TUNE=false` or
`--schedule_tune false`. To initialize from saved schedule arrays instead of
cosine, provide both paths:

```bash
DATA_DIR=/path/to/training/images \
BETA_VALUES_FILE=/path/to/ddpm_betas.npy \
OMEGA_VALUES_FILE=/path/to/ddpm_omegas.npy \
./launch_training_example.sh
```

Arguments after the launcher command are passed directly to `image_train.py`.

## MNIST smoke training

The MNIST launcher downloads and exports a small training subset, then runs a
20-step grayscale training job with a compact 32 x 32 model:

```bash
PYTHON_BIN=/path/to/python ./launch_mnist_test.sh
```

Useful overrides include `MNIST_LIMIT`, `TRAIN_STEPS`, `BATCH_SIZE`,
`OUTPUT_DIR`, and `MNIST_CACHE_DIR`.

## Use the trained prior

Each checkpoint step produces three files:

- `modelNNNNNN.pt`
- `ddpm_betasNNNNNN.npy`
- `ddpm_omegasNNNNNN.npy`

Point the sampler at files from the same step:

```bash
MODEL_PATH=/path/to/checkpoints/model300000.pt \
BETA_PATH=/path/to/checkpoints/ddpm_betas300000.npy \
OMEGA_PATH=/path/to/checkpoints/ddpm_omegas300000.npy \
./launch_example.sh
```

For an RGB prior, also set `IMAGE_CHANNELS=3` when sampling. The
interferometric likelihood is intended for single-channel images; RGB priors
are suitable for unconditional or custom-likelihood sampling.
