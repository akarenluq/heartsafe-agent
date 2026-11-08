#!/bin/bash
#SBATCH --job-name=hs_reliability
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
# CPU only. Re-runs reliability, audit and evaluation on the existing predictions
# (after a config.yaml change, e.g. an approved safeguard). Same saved thresholds.
set -e
cd "${SLURM_SUBMIT_DIR:?submit from the imaging/ folder}"; source slurm/env.sh
OUT=$HS_ROOT/results
cd $HS_CODE/scripts
python 04_postprocess_reliability.py --raw $RAW --pred $HS_ROOT/preds/clean --out $OUT/clean
python 04_postprocess_reliability.py --raw $RAW --pred $HS_ROOT/preds/faulty --out $OUT/faulty \
    --images $HS_ROOT/faulty/imagesTs --labels $HS_ROOT/faulty/labelsTs --faults $HS_ROOT/faulty/faults.csv
python 06_subgroup_audit.py --results $OUT/clean/exam_results.csv --metadata $RAW/metadata.csv --out $OUT/audit
python 07_eval_reliability.py --clean $OUT/clean/exam_results.csv --faulty $OUT/faulty/exam_results.csv --out $OUT/reliability.json
echo "DONE -> $OUT"
