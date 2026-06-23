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
