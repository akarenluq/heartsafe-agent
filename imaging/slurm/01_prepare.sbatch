#!/bin/bash
#SBATCH --job-name=hs_prepare
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
# CPU only: unzip, check, conversion and nnU-Net planning/preprocessing.
set -e
cd "${SLURM_SUBMIT_DIR:?submit from the imaging/ folder}"; source slurm/env.sh
cd $HS_CODE/scripts

# Unzip once with Python (some sites have no unzip command)
if [ ! -d $HS_ROOT/mnms ]; then
  python -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" $HS_ROOT/archive.zip $HS_ROOT/mnms.tmp
  mv $HS_ROOT/mnms.tmp $HS_ROOT/mnms
fi

python 00_inspect_dataset.py $HS_ROOT/mnms
python 01_convert_mnms_to_nnunet.py --mnms $HS_ROOT/mnms --out $nnUNet_raw --config ../config.yaml
nnUNetv2_plan_and_preprocess -d 201 -c 2d --verify_dataset_integrity -np 8
