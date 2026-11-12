#!/bin/bash
#SBATCH --job-name=hs_ecg_rhythm
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --gpus-per-node=1
#SBATCH --output=%x_%j.out
# Rhythm model (normal / AF / other abnormal). Run after 02_train_evaluate (it adds to those outputs).
# From the ecg/ folder: sbatch --account=<ACCOUNT> --partition=<GPU> slurm/03_rhythm.sbatch
# Trained members are kept, so resubmitting after a timeout continues where it stopped.
set -e
cd "${SLURM_SUBMIT_DIR:?submit from the ecg/ folder}"; source slurm/env.sh
cd scripts
python 05_rhythm.py --data $ECG_WORK --models $ECG_WORK/models --results $ECG_RESULTS --config ../config.yaml \
    --ptbxl $ECG_DATA/ptbxl
echo "DONE -> $ECG_RESULTS/rhythm_evaluation.json"
