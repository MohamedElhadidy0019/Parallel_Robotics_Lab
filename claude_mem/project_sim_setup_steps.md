---
name: project-sim-setup-steps
description: Actual working setup steps using conda/miniforge (uv abandoned — has no pip)
metadata:
  type: project
---

Full setup from scratch (conda-based, NOT uv):

```bash
# 1. Clone shelf_gym into third_party/ (already gitignored)
git clone --recurse-submodules -j8 \
  https://github.com/NilsDengler/manipulation_enhanced_map_prediction \
  third_party/shelf_gym_repo

# 2. Create conda env (miniforge, Python 3.12)
conda create -n rob_env python=3.12
conda activate rob_env
conda install -c conda-forge "cgal<6"   # CGAL 6+ removed boost::optional, breaks skgeom

# 3. Apply patches (fix setup.py multi-package discovery, requirements, klampt Qt crash)
git apply patches/shelf_gym.patch --directory=third_party/shelf_gym_repo

# 4. Install skgeom deps in the right order
pip install "pybind11[global]==2.11.1"  # 2.12+ breaks skgeom def_property+keep_alive

# 5. Run the repo's install script FROM INSIDE the repo
cd third_party/shelf_gym_repo
bash install.sh
pip install -e . --no-build-isolation   # must use already-installed pybind11
cd ../..

# 6. Run
python nbv_env2.py
```

**Known issues resolved:**
- CGAL must be `<6` (conda-forge) — v6 removed `boost::optional` used by skgeom
- pybind11 must be `==2.11.1` — v2.12+ broke skgeom's `def_property` with `keep_alive`
- `install.sh` must run from `cd third_party/shelf_gym_repo/`, not from project root
- `configureDebugVisualizer(COV_ENABLE_RENDERING, 0)` in base_environment is re-enabled after loading
- table.urdf has NO collision (commented out) — objects fall through it; use GEOM_BOX with collision instead

**Patches in `patches/shelf_gym.patch` cover:**
- `setup.py`: `find_packages(include=['shelf_gym', 'shelf_gym.*'])` (fixes multi-pkg detection)
- `requirements.txt`: commented out training deps (wandb, sb3, lightning, tensorboard, torchvision...)
- `shelf_environment.py`: `show_vis=False` in __main__ (prevents klampt Qt/GLUT crash)
- `camera_utils.py`: `.reshape(height, width, channels)` on all PyBullet image buffers; `.astype(np.float32)` for open3d depth
