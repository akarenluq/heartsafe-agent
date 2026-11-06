#!/bin/bash
#SBATCH --job-name=hs_train
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=%x_%j.out
# NEEDS_GPU
# One fold, 250 epochs, 2d. If it times out, resubmit with --c added to continue.
set -e
cd "${SLURM_SUBMIT_DIR:?submit from the imaging/ folder}"; source slurm/env.sh
nnUNetv2_train 201 2d 0 -tr nnUNetTrainer_250epochs ${RESUME:+--c}
