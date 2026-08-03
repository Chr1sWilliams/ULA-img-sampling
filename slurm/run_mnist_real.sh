#!/bin/bash
#SBATCH --job-name=mnist_real
#SBATCH --output=slurm/logs/mnist_real_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:30:00

source ~/miniforge3/bin/activate
conda activate ula
cd ~/ULA-img-sampling
source ~/.wandb_key
PYTHON_BIN=$(which python) ./launch_mnist_real.sh
