# Next-Best-View (NBV) 3D Scan Pipeline

Autonomous Next-Best-View (NBV) 3D scanning pipeline on a simulated UR5 arm with a Robotiq-85 gripper and wrist camera in PyBullet.

Target: Plan collision-free camera trajectories around a target object, evaluate candidate viewpoints via GPU ray-casting against a CAD model, capture RGB-D depth frames, and iteratively reconstruct the surface.

---

## Environment Setup

Tested on NVIDIA GTX 10-series (Pascal sm_61) using CuRobo v0.7.8 and PyTorch 2.4.1 (CUDA 12.1).

### 1. Conda Environment

```bash
conda create -n rob_env python=3.12 -y
conda activate rob_env
conda install -c conda-forge "cgal<6" ninja -y
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```

### 2. Submodules and Dependencies

```bash
# shelf_gym (robot models and simulation base)
git clone --recurse-submodules -j8 https://github.com/NilsDengler/manipulation_enhanced_map_prediction third_party/shelf_gym_repo
git apply patches/shelf_gym.patch --directory=third_party/shelf_gym_repo
cd third_party/shelf_gym_repo
pip install "pybind11[global]==2.11.1"
bash install.sh
pip install -e . --no-build-isolation
cd ../..

# CuRobo v0.7.8 (kinematics and collision-aware motion planning)
git clone --branch v0.7.8 https://github.com/nvlabs/curobo.git third_party/curobo_legacy_rob_env
cd third_party/curobo_legacy_rob_env
TORCH_CUDA_ARCH_LIST=6.1+PTX pip install -v -e . --no-build-isolation
cd ../..
```

---

## Development Roadmap

The pipeline is being built modularly across the following stages:

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

---

## Observations and Known Issues (from `main` branch)

- Merged raw point clouds exhibit minor multi-view registration ribbing due to arm settling precision residuals.
- Table-contact base (~15mm) is physically unobservable from above-table camera orbits and is excluded from coverage evaluation.
