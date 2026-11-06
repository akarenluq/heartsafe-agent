#!/bin/bash
#SBATCH --job-name=hs_prepare
#SBATCH --account=def-CHANGEME
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
# CPU only: unzip check, conversion and nnU-Net planning/preprocessing.
set -e
source $HOME/scratch/heartsafe/HeartSafe_WAI-Hackathon/imaging/slurm/env.sh
cd $HS_CODE/scripts

python 00_inspect_dataset.py $HS_ROOT/mnms
python 01_convert_mnms_to_nnunet.py --mnms $HS_ROOT/mnms --out $nnUNet_raw --config ../config.yaml
nnUNetv2_plan_and_preprocess -d 201 -c 2d --verify_dataset_integrity -np 8
