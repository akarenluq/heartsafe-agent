#!/bin/bash
# Run ONCE on a login node (needs internet). pip + virtualenv, no conda.
set -e
module load StdEnv/2023 python/3.11
virtualenv --no-download $HOME/envs/heartsafe
source $HOME/envs/heartsafe/bin/activate
pip install --no-index --upgrade pip
# Prebuilt cluster wheels first (torch with CUDA, scientific stack).
pip install --no-index torch torchvision numpy scipy pandas scikit-learn matplotlib nibabel pyyaml SimpleITK
# nnU-Net: cluster wheel if available, otherwise PyPI.
pip install --no-index nnunetv2 || pip install nnunetv2
python - <<'EOF'
import torch, nnunetv2, nibabel, SimpleITK, sklearn, yaml
print("torch", torch.__version__, "| CUDA build:", torch.version.cuda)
print("nnunetv2 OK at", nnunetv2.__file__)
EOF
echo "Environment ready. GPU is only visible on compute nodes; the smoke test checks it."
