# Parallel Robotics Lab — NBV Inspection Pipeline

Next-Best-View inspection pipeline on a UR5 robot arm using PyBullet and CuRobo.

## Setup

### 1. Clone shelf_gym into third_party/

```bash
git clone --recurse-submodules -j8 https://github.com/NilsDengler/manipulation_enhanced_map_prediction third_party/shelf_gym_repo
```

### 2. Create conda environment

```bash
conda create -n rob_env python=3.12
conda activate rob_env
conda install -c conda-forge "cgal<6"
```

### 3. Apply patches and install

```bash
git apply patches/shelf_gym.patch --directory=third_party/shelf_gym_repo
cd third_party/shelf_gym_repo
pip install "pybind11[global]==2.11.1"
bash install.sh
pip install -e . --no-build-isolation
cd ../..
```

### 4. Run

```bash
python nbv_environment.py
```

## Full pipeline (scan -> reconstruct -> grasp)

`full_pipeline.py` runs the whole scan -> reconstruct -> grasp pipeline end to end in one file. Three pieces are currently placeholders for work that isn't real yet, each marked with a `TODO` and written so replacing just that function's body is enough - the surrounding orchestration doesn't need to change:

- `move_camera_to()` - motion execution. Currently PyBullet IK + a smooth adaptive glide between poses; will be replaced with CuRobo.
- `select_next_view_pose()` - next-best-view selection. Currently a fixed, precomputed orbit; will be replaced with `nbv_planner.py` (dynamic candidate generation + scoring).
- `compute_grasp_pose()` - grasp planning. Not implemented yet at all - currently a no-op stub.

Running it today still produces a real, working combined point cloud (mustard-bottle-only, back-projected and merged from every reachable view, no ICP).

```bash
python full_pipeline.py            # headless, saves the combined cloud
python full_pipeline.py --gui      # show the PyBullet GUI (watch the robot move)
python full_pipeline.py --view     # open an Open3D window with the combined cloud at the end
```

Output goes to `captures/full_pipeline/combined_pointcloud.npy` and `.ply`.
