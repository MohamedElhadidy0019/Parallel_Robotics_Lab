#!/usr/bin/env bash
# One-off script: prove whether CuRobo can build on this machine's GTX 1060 (Pascal, sm_61).
# Isolated from rob_env on purpose - uses a throwaway env (curobo_test, python 3.10)
# matching CuRobo's actual documented supported range, and restricts the CUDA build
# to exactly this GPU's arch so a real incompatibility fails fast instead of after
# a slow multi-arch compile.
set -euo pipefail

REPO_ROOT="/home/mohamed/repos/Parallel_Robotics_Lab"
CUROBO_DIR="$REPO_ROOT/third_party/curobo"

echo "==> Setting up curobo_test conda env (python 3.10)"
if conda env list | grep -q '^curobo_test '; then
    echo "    curobo_test already exists, reusing it"
else
    conda create -n curobo_test python=3.10 -y
fi

# Needed so 'conda activate' works inside a non-interactive script
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate curobo_test

echo "==> Installing torch 2.4.1+cu121 (last known-good build for Pascal/sm_61)"
pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1

echo "==> Checking git-lfs (curobo needs it for asset files)"
if ! command -v git-lfs >/dev/null 2>&1; then
    echo "    git-lfs not found, installing (will prompt for sudo password)"
    sudo apt install -y git-lfs
fi

echo "==> Cloning curobo into $CUROBO_DIR"
if [ -d "$CUROBO_DIR" ]; then
    echo "    $CUROBO_DIR already exists, skipping clone"
else
    git clone https://github.com/NVlabs/curobo.git "$CUROBO_DIR"
fi

echo "==> Building curobo, restricted to compute capability 6.1 (this GPU's real arch)"
cd "$CUROBO_DIR"
export TORCH_CUDA_ARCH_LIST="6.1+PTX"
pip install -e . --no-build-isolation

echo "==> Done. If you saw no 'Unsupported gpu architecture' error above, the build succeeded."
echo "    Sanity check: python -c \"import curobo; print(curobo.__file__)\""
