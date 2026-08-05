#!/usr/bin/env bash
# One-off script: try installing CuRobo v0.7.8 directly into rob_env (Python 3.12),
# to see if we can avoid a two-environment split with curobo_test (Python 3.10, already
# proven working - see temp2.sh). CuRobo's docs list Python 3.8-3.10 as supported,
# 3.11+ as "untested" - this is a genuine experiment, not a known-good recipe.
#
# Clones into a SEPARATE directory from curobo_legacy (which curobo_test already uses),
# since editable installs (`pip install -e .`) place compiled .so files inside the
# source tree itself - rebuilding from the same clone for a different Python/torch
# would overwrite curobo_test's working build.
set -euo pipefail

REPO_ROOT="/home/mohamed/repos/Parallel_Robotics_Lab"
CUROBO_DIR="$REPO_ROOT/third_party/curobo_legacy_rob_env"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rob_env

echo "==> Fixing rob_env's torch (currently 2.11.0+cu128, which dropped Pascal/sm_61 support)"
pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --force-reinstall

echo "==> Verifying GPU is usable (expect: True, GTX 1060 name, NO compute-capability warning)"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

echo "==> Installing ninja (for parallel CUDA extension build - without it, expect 30-90min)"
pip install ninja

echo "==> Cloning curobo v0.7.8 into $CUROBO_DIR"
if [ -d "$CUROBO_DIR" ]; then
    echo "    $CUROBO_DIR already exists, skipping clone"
else
    git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git "$CUROBO_DIR"
fi

echo "==> Building curobo, restricted to compute capability 6.1 (this GPU's real arch)"
cd "$CUROBO_DIR"
export TORCH_CUDA_ARCH_LIST="6.1+PTX"
pip install -v -e . --no-build-isolation

echo "==> Build succeeded. Sanity check with our real robot + camera-link IK test:"
echo "    cd $REPO_ROOT && python curobo_ur5_ik_test.py"
echo "    (same script that already passed in curobo_test - if it passes here too,"
echo "     rob_env alone is enough and curobo_test/the split-env fallback isn't needed)"
