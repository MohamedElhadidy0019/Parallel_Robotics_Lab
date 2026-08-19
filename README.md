# Next-Best-View (NBV) 3D Scan Pipeline

Autonomous Next-Best-View 3D scanning pipeline on a simulated UR5 robot arm with a Robotiq-85 gripper and wrist camera in PyBullet.

The system plans collision-free camera trajectories around a target object using CuRobo, evaluates candidate views using GPU ray-casting against a known CAD model, captures RGB-D depth frames, and reconstructs the visible surface until reaching ~92% coverage (excluding the occluded 15mm table-contact base).

---

## Hardware and GPU Target

This setup is tailored and verified for **NVIDIA Pascal GPUs (e.g. GTX 10-series, Compute Capability 6.1)**:
- **CuRobo v0.7.8**: Newer CuRobo releases require Volta or newer GPUs with Tensor Cores. Version 0.7.8 maintains full Pascal (`sm_61`) support.
- **PyTorch 2.4.1 (CUDA 12.1)**: Pinned because PyTorch dropped default Pascal support in later prebuilt wheels.
- **Custom Trajopt Configs**: CuRobo CUDA LBFGS trajectory kernels are disabled in config (`use_cuda_*_kernel: False`) to prevent Pascal driver argument faults, falling back to CuRobo's PyTorch JIT execution path.

For newer GPUs (RTX 20/30/40 series), update `TORCH_CUDA_ARCH_LIST` during the CuRobo build step accordingly (e.g. `7.5`, `8.6`, or `8.9`).

---

## Quick Setup

### 1. Create Conda Environment

```bash
conda create -n rob_env python=3.12 -y
conda activate rob_env
conda install -c conda-forge "cgal<6" ninja -y
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```

### 2. Install Dependencies

```bash
# Clone and install shelf_gym (robot meshes and base environment)
git clone --recurse-submodules -j8 https://github.com/NilsDengler/manipulation_enhanced_map_prediction third_party/shelf_gym_repo
git apply patches/shelf_gym.patch --directory=third_party/shelf_gym_repo
cd third_party/shelf_gym_repo
pip install "pybind11[global]==2.11.1"
bash install.sh
pip install -e . --no-build-isolation
cd ../..

# Clone and compile CuRobo (Pascal GTX 10-series build)
git clone --branch v0.7.8 https://github.com/nvlabs/curobo.git third_party/curobo_legacy_rob_env
cd third_party/curobo_legacy_rob_env
TORCH_CUDA_ARCH_LIST=6.1+PTX pip install -v -e . --no-build-isolation
cd ../..

# Install local package in editable mode
pip install -e .
```

---

## Architecture and Roadmap

```
Parallel_Robotics_Lab/
├── nbv_core/
│   ├── camera.py          # Stage 1: Intrinsics, 2D depth -> 3D point cloud backprojection
│   ├── coverage.py        # Stage 2: Known-CAD mesh sampling, KD-Tree tracking, visualization
│   ├── ray_scoring.py     # Stage 3: PyTorch GPU Möller-Trumbore ray casting & utility scoring
│   └── motion_planner.py  # Stage 4: CuRobo reachability check & collision-aware trajectory
├── sim_env.py             # PyBullet UR5 + Table + YCB Object environment
└── main.py                # Stage 5: Closed-loop NBV scan execution loop
```
