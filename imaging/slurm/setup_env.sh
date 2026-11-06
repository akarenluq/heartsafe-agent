#!/bin/bash
# Run ONCE on a login node after configure_site.sh. pip + virtualenv, no conda.
# Install order: site wheelhouse (CVMFS) -> our offline wheelhouse/ -> PyPI (only if internet).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/site.env"
WH="$HERE/../wheelhouse"
module load $HS_MODULES
python --version

virtualenv --no-download "$HS_ROOT/envs/heartsafe" 2>/dev/null || python -m venv "$HS_ROOT/envs/heartsafe"
source "$HS_ROOT/envs/heartsafe/bin/activate"

# Torch (with CUDA) and the standard stack from the site's prebuilt wheels.
pip install --no-index --upgrade pip || true
pip install --no-index torch torchvision numpy scipy pandas scikit-learn scikit-image matplotlib seaborn \
    nibabel pyyaml SimpleITK tqdm tifffile requests \
  || { echo "Site wheels missing; trying PyPI (needs internet)"; pip install torch torchvision numpy scipy pandas scikit-learn scikit-image matplotlib seaborn nibabel pyyaml SimpleITK tqdm tifffile requests; }

# nnU-Net and its niche dependencies: offline bundle first.
if [ -d "$WH" ] && ls "$WH"/nnunetv2-*.whl >/dev/null 2>&1; then
  pip install --no-index --find-links "$WH" nnunetv2
else
  pip install --no-index nnunetv2 || pip install nnunetv2
fi

python - <<'EOF'
import torch, nnunetv2, nibabel, SimpleITK, sklearn, yaml
print("torch", torch.__version__, "| CUDA build:", torch.version.cuda)
print("nnunetv2 OK at", nnunetv2.__file__)
EOF
echo "Environment ready. The smoke test checks the GPU on a compute node."
