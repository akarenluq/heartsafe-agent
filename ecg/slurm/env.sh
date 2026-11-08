# Sourced by every ECG job. Reuses the imaging site settings and Python environment
# (run imaging/slurm/configure_site.sh and setup_env.sh first, then: pip install wfdb).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../../imaging/slurm/site.env"
: "${HS_ROOT:?HS_ROOT not set. Run imaging/slurm/configure_site.sh first}"
module load ${HS_MODULES:-StdEnv/2023 python/3.11} 2>/dev/null || echo "WARNING: module load failed"
source "${HS_VENV:-$HS_ROOT/envs/heartsafe}/bin/activate"
export ECG_DATA=$HS_ROOT/ecg_data       # ptbxl.zip and training2017.zip go here
export ECG_WORK=$HS_ROOT/ecg_work       # prepared arrays and models
export ECG_RESULTS=$HS_ROOT/ecg_results
mkdir -p $ECG_DATA $ECG_WORK $ECG_RESULTS
