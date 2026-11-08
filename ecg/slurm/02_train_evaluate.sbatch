#!/bin/bash
#SBATCH --job-name=hs_ecg_train
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --gpus-per-node=1
#SBATCH --output=%x_%j.out
# From the ecg/ folder: sbatch --account=<ACCOUNT> --partition=<GPU> slurm/02_train_evaluate.sbatch
# Trained members are kept, so resubmitting after a timeout continues where it stopped.
set -e
cd "${SLURM_SUBMIT_DIR:?submit from the ecg/ folder}"; source slurm/env.sh
cd scripts
python 02_train.py --data $ECG_WORK --out $ECG_WORK/models --config ../config.yaml
rm -f $ECG_RESULTS/thresholds.json   # recalibrate for this ensemble
python 03_predict_reliability.py --data $ECG_WORK --models $ECG_WORK/models --out $ECG_RESULTS --config ../config.yaml
python 04_audit_eval.py --results $ECG_RESULTS --out $ECG_RESULTS
echo "DONE -> $ECG_RESULTS"
