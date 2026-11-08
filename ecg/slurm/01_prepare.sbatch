#!/bin/bash
#SBATCH --job-name=hs_ecg_prepare
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=%x_%j.out
# CPU only. From the ecg/ folder: sbatch --account=<ACCOUNT> --partition=<CPU> slurm/01_prepare.sbatch
set -e
cd "${SLURM_SUBMIT_DIR:?submit from the ecg/ folder}"; source slurm/env.sh

# Unzip once with Python (some sites have no unzip command)
for pair in "ptbxl.zip ptbxl" "training2017.zip cinc2017"; do
  set -- $pair
  if [ ! -d $ECG_DATA/$2 ]; then
    python -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" $ECG_DATA/$1 $ECG_DATA/$2.tmp
    mv $ECG_DATA/$2.tmp $ECG_DATA/$2
  fi
done

python scripts/01_prepare.py --ptbxl $ECG_DATA/ptbxl --cinc $ECG_DATA/cinc2017 --out $ECG_WORK \
    --config config.yaml --workers ${SLURM_CPUS_PER_TASK:-8}
