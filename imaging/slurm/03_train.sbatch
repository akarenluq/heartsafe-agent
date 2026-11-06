#!/bin/bash
#SBATCH --job-name=hs_train
#SBATCH --account=def-CHANGEME
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=%x_%j.out
# One fold, 250 epochs, 2d. If it times out, resubmit with --c added to continue.
set -e
source $HOME/scratch/heartsafe/HeartSafe_WAI-Hackathon/imaging/slurm/env.sh
nnUNetv2_train 201 2d 0 -tr nnUNetTrainer_250epochs ${RESUME:+--c}
