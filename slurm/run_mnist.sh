#!/bin/bash
#SBATCH --job-name=mnist_test
#SBATCH --output=slurm/logs/mnist_test_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00

source ~/miniforge3/bin/activate
conda activate ula
source ~/.wandb_key
cd ~/ULA-img-sampling
PYTHON_BIN=$(which python) ./launch_mnist_test.sh
