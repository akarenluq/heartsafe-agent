# Sourced by every job. Works on SDRE and on Alliance clusters.
# Site values (storage root, modules) live in slurm/site.env, written by configure_site.sh.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/site.env" ] && source "$HERE/site.env"

: "${HS_ROOT:?HS_ROOT not set. Run: bash slurm/configure_site.sh first}"
export HS_CODE="$(cd "$HERE/.." && pwd)"
export HS_VENV="${HS_VENV:-$HS_ROOT/envs/heartsafe}"

module load ${HS_MODULES:-StdEnv/2023 python/3.11} 2>/dev/null || echo "WARNING: module load failed; check HS_MODULES in slurm/site.env"
source "$HS_VENV/bin/activate"

export nnUNet_raw=$HS_ROOT/nnUNet_raw
export nnUNet_preprocessed=$HS_ROOT/nnUNet_preprocessed
export nnUNet_results=$HS_ROOT/nnUNet_results
export nnUNet_compile=f          # avoids torch.compile issues on cluster nodes
export nnUNet_n_proc_DA=${SLURM_CPUS_PER_TASK:-8}
export RAW=$nnUNet_raw/Dataset201_HeartSafeMNMs
mkdir -p $nnUNet_raw $nnUNet_preprocessed $nnUNet_results
