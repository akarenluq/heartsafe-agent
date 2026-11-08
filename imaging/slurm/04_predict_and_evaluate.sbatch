#!/bin/bash
#SBATCH --job-name=hs_predict
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
# NEEDS_GPU
# Clean + faulty predictions, reliability outputs, equity audit, evaluation.
set -e
cd "${SLURM_SUBMIT_DIR:?submit from the imaging/ folder}"; source slurm/env.sh
TR=nnUNetTrainer_250epochs
OUT=$HS_ROOT/results; mkdir -p $OUT
cd $HS_CODE/scripts

# Clean held-out exams (val + test). Thresholds are calibrated on this run.
rm -f $RAW/reliability_thresholds.json
nnUNetv2_predict -i $RAW/imagesTs -o $HS_ROOT/preds/clean -d 201 -c 2d -f 0 -tr $TR --save_probabilities
python 04_postprocess_reliability.py --raw $RAW --pred $HS_ROOT/preds/clean --out $OUT/clean

# Faulty copies with known faults, same thresholds.
python 05_inject_faults.py --raw $RAW --out $HS_ROOT/faulty --fraction 0.5
nnUNetv2_predict -i $HS_ROOT/faulty/imagesTs -o $HS_ROOT/preds/faulty -d 201 -c 2d -f 0 -tr $TR --save_probabilities
python 04_postprocess_reliability.py --raw $RAW --pred $HS_ROOT/preds/faulty --out $OUT/faulty \
    --images $HS_ROOT/faulty/imagesTs --labels $HS_ROOT/faulty/labelsTs --faults $HS_ROOT/faulty/faults.csv

python 06_subgroup_audit.py --results $OUT/clean/exam_results.csv --metadata $RAW/metadata.csv --out $OUT/audit
python 07_eval_reliability.py --clean $OUT/clean/exam_results.csv --faulty $OUT/faulty/exam_results.csv --out $OUT/reliability.json
echo "DONE -> $OUT"
