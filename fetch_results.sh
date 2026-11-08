#!/bin/bash
# Copy imaging and ECG results from SDRE to this PC (run in Git Bash, VPN on).
# Results only: summary tables, per-exam JSONs, plots, training logs. No raw data.
#
# Needs an open MFA connection first (lasts 12 h):
#   ssh -M -S ~/.ssh/sdre.sock -o ControlPersist=12h safwatab@login.sdre.ualberta.ca
# Then:  bash fetch_results.sh            (both parts)
#        bash fetch_results.sh imaging    (one part)
set -e
HOST=${SDRE_HOST:-safwatab@login.sdre.ualberta.ca}
ROOT=${SDRE_ROOT:-/uhome/safwatab/projects/p70566/heartsafe}
HERE="$(cd "$(dirname "$0")" && pwd)"
sdre() { ssh -S ~/.ssh/sdre.sock -O proxy -o BatchMode=yes "$HOST" "$@"; }
sdre true || { echo "No open SDRE connection. Run the ssh -M command at the top of this file first."; exit 1; }

fetch() {  # fetch <remote dir under ROOT> <local dir> [tar excludes...]
  local src=$1 dst=$2; shift 2
  if ! sdre "test -d $ROOT/$src"; then echo "skip $src (not on SDRE yet)"; return; fi
  mkdir -p "$dst"
  sdre "cd $ROOT/$src && tar cf - $* ." | tar xf - -C "$dst"
  echo "copied $src -> $dst ($(find "$dst" -type f | wc -l) files)"
}

TR=nnUNet_results/Dataset201_HeartSafeMNMs/nnUNetTrainer_250epochs__nnUNetPlans__2d/fold_0
if [ -z "$1" ] || [ "$1" = imaging ]; then
  fetch results "$HERE/imaging/results"
  fetch $TR "$HERE/imaging/results/training" --exclude='*.pth' --exclude=validation
fi
if [ -z "$1" ] || [ "$1" = ecg ]; then
  fetch ecg_results "$HERE/ecg/results"
  fetch ecg_work/models "$HERE/ecg/results/training" --exclude='*.pt'
fi
sdre "cd $ROOT/HeartSafe_WAI-Hackathon && tar cf - imaging/*.out ecg/*.out 2>/dev/null" | tar xf - -C "$HERE/" 2>/dev/null \
  && echo "copied job logs (*.out) into imaging/ and ecg/"
