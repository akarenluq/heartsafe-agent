#!/bin/bash
# One-time site setup. Values come from the SDRE Technical Wiki you get at onboarding.
#
#   bash slurm/configure_site.sh --root /path/to/project/heartsafe \
#        --account YOUR_ACCOUNT [--gpu-partition GPU_PARTITION] [--cpu-partition CPU_PARTITION] \
#        [--gpus 1 | --gpus h200:1] [--modules "StdEnv/2023 python/3.11"]
#
# Writes slurm/site.env and puts account/partition/GPU lines into every slurm/*.sbatch.
set -e
cd "$(dirname "$0")"
GPUS="1"; MODULES="StdEnv/2023 python/3.11"; CPU_PART=""
while [ $# -gt 0 ]; do
  case $1 in
    --root) ROOT=$2; shift 2;;
    --account) ACCOUNT=$2; shift 2;;
    --gpu-partition) GPU_PART=$2; shift 2;;
    --cpu-partition) CPU_PART=$2; shift 2;;
    --gpus) GPUS=$2; shift 2;;
    --modules) MODULES=$2; shift 2;;
    *) echo "unknown option $1"; exit 1;;
  esac
done
: "${ROOT:?--root is required (use Project storage, not Home: Home is only 25 GB)}"
mkdir -p "$ROOT"

cat > site.env <<EOF
export HS_ROOT=$ROOT
export HS_MODULES="$MODULES"
EOF

for f in *.sbatch; do
  sed -i '/^#SBATCH --account=/d; /^#SBATCH --partition=/d; /^#SBATCH --gres=/d; /^#SBATCH --gpus-per-node=/d' "$f"
  lines=""
  [ -n "$ACCOUNT" ] && lines="#SBATCH --account=$ACCOUNT"
  if grep -q "^# NEEDS_GPU" "$f"; then
    [ -n "$GPU_PART" ] && lines="$lines\n#SBATCH --partition=$GPU_PART"
    lines="$lines\n#SBATCH --gpus-per-node=$GPUS"
  elif [ -n "$CPU_PART" ]; then
    lines="$lines\n#SBATCH --partition=$CPU_PART"
  fi
  sed -i "0,/^#SBATCH --job-name=.*/s//&\n${lines#\\n}/" "$f"
done
echo "site.env:"; cat site.env
echo; grep -H "^#SBATCH --\(account\|partition\|gres\)" *.sbatch
