#!/bin/bash
#SBATCH --job-name=hs_smoke
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=%x_%j.out
# NEEDS_GPU
# 5-epoch run + prediction on 4 cases + postprocessing. Must pass before 03_train.
set -e
cd "${SLURM_SUBMIT_DIR:?submit from the imaging/ folder}"; source slurm/env.sh
python -c "import torch; assert torch.cuda.is_available(), 'no GPU'; print('GPU:', torch.cuda.get_device_name(0))"

nnUNetv2_train 201 2d 0 -tr nnUNetTrainer_5epochs

SMOKE=$HS_ROOT/smoke; mkdir -p $SMOKE/in
ls $RAW/imagesTs/*_0000.nii.gz | head -4 | xargs -I{} cp {} $SMOKE/in/
nnUNetv2_predict -i $SMOKE/in -o $SMOKE/pred -d 201 -c 2d -f 0 -tr nnUNetTrainer_5epochs --save_probabilities
ls $SMOKE/pred
cd $HS_CODE/scripts
python 04_postprocess_reliability.py --raw $RAW --pred $SMOKE/pred --images $SMOKE/in --out $SMOKE/results
rm -f $RAW/reliability_thresholds.json   # smoke thresholds are not real; recalibrate later
echo "SMOKE TEST PASSED"
