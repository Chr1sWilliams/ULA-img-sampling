#!/bin/bash
#SBATCH --job-name=mnist_test
#SBATCH --output=slurm/logs/mnist_test_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:20:00

source ~/miniforge3/bin/activate
conda activate ula
cd ~/ULA-img-sampling
PYTHON_BIN=$(which python) ./launch_mnist_test.sh
