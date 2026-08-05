#!/usr/bin/env bash
# One-off script: test whether CuRobo's older, pre-Warp-tile-matmul release (v0.7.8)
# avoids the cuBLASDx/Tensor-Core requirement that broke the current main ("curobov2")
# branch on this machine's GTX 1060 (Pascal, sm_61, no Tensor Cores).
# Reuses the curobo_test conda env (already python 3.10 + torch 2.4.1+cu121, both
# already proven to work on this GPU).
set -euo pipefail

REPO_ROOT="/home/mohamed/repos/Parallel_Robotics_Lab"
CUROBO_LEGACY_DIR="$REPO_ROOT/third_party/curobo_legacy"

echo "==> Cloning curobo v0.7.8 into $CUROBO_LEGACY_DIR"
if [ -d "$CUROBO_LEGACY_DIR" ]; then
    echo "    $CUROBO_LEGACY_DIR already exists, skipping clone"
else
    git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git "$CUROBO_LEGACY_DIR"
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate curobo_test

echo "==> Removing the previously installed curobo v2 (main branch) package"
pip uninstall -y nvidia-curobo curobo || true

echo "==> Building curobo v0.7.8, restricted to compute capability 6.1 (this GPU's real arch)"
cd "$CUROBO_LEGACY_DIR"
export TORCH_CUDA_ARCH_LIST="6.1+PTX"
pip install -e . --no-build-isolation

echo "==> Running the classic basic IK example (ur10e.yml, no args needed)"
python examples/ik_example.py

echo "==> Done. If the loop above printed 10 'Success, Solve Time(s), hz' lines with no crash,"
echo "    v0.7.8 works on this GPU even though the v2/main branch does not."
