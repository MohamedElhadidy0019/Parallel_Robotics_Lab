# Parallel Robotics Lab — Next-Best-View (NBV) Scan Pipeline

A closed-loop Next-Best-View system on a simulated UR5 + Robotiq-85 arm (PyBullet):
it generates candidate camera viewpoints around a known object, filters them for
reachability with CuRobo, scores the reachable ones by GPU ray-casting against the
object's known CAD model (how much unseen surface a viewpoint would reveal), greedily
moves to the best one with CuRobo's collision-aware motion planner, captures a depth
image, and repeats until it has covered enough of the object's surface. No grasping.

**Current status: working end-to-end, hitting ~92% known-CAD surface coverage**
(target was 90-95%, excluding the object's table-contact base, which no
above-table camera can ever see — see [Known-CAD coverage & the base
exclusion](#known-cad-coverage--the-base-exclusion) below).

```bash
conda run -n rob_env python full_pipeline.py --gui --view
```

---

## What's actually in this repo

The pipeline has six pieces, each independently testable, tied together by
`nbv_planner.py` and driven end-to-end by `full_pipeline.py`:

| Step | What it does | Where |
|---|---|---|
| A — Reachability cache | Samples a hemisphere shell of candidate camera poses around the object, IK-checks each with CuRobo, caches the result to disk (computed once, not per-run) | `build_reachability_cache.py`, `nbv_core/reachability.py` |
| B — Known-CAD coverage | Samples the object's real mesh into surface points + normals; tracks which ones a real capture has confirmed "seen" (nearest-neighbor + normal-consistency check) | `nbv_core/coverage.py` |
| C — GPU ray-scoring | For a reachable candidate, batched GPU ray/triangle intersection (own Möller–Trumbore implementation in PyTorch) scores how many currently-unseen surface points it would reveal, occlusion- and backface-aware | `nbv_core/ray_scoring.py` |
| D — NBV planner | Greedy argmax over A's cached reachable candidates, scored by C, against B's live coverage state; stops once the target coverage is hit or nothing worthwhile remains | `nbv_planner.py` |
| E — Motion execution | CuRobo `MotionGen`-backed collision-aware trajectory planning (self-collision **and** object/table collision) — not a hand-rolled IK glide | `nbv_core/motion_planning.py` |
| F — Orchestration | Wires A–E into the real scan loop; captures a real depth image and backprojects it every successful move | `full_pipeline.py` |

Supporting/diagnostic scripts:
- `evaluate_reconstruction.py` — quantitative accuracy/completeness check of the raw
  scanned points against the true mesh, broken down by height band.
- `compare_pointcloud_to_mesh.py` — similar check for the older
  `scan_and_save_mustard_only.py` pipeline.
- `curobo_ur5_ik_test.py` — minimal standalone CuRobo IK spike against the real
  URDF, useful for sanity-checking a CuRobo environment in isolation.

### Run order

```bash
# 1. One-time (or whenever the object placement / robot URDF changes):
conda run -n rob_env python build_reachability_cache.py

# 2. Run the real scan:
conda run -n rob_env python full_pipeline.py --gui --view

# 3. Quantitative accuracy check (optional):
conda run -n rob_env python evaluate_reconstruction.py --view
```

`full_pipeline.py` prints per-view progress and a final coverage percentage, and
saves to `captures/full_pipeline/`:
- `combined_pointcloud.npy` / `.ply` — the raw backprojected scan points (all views concatenated)
- `object_pose.npz` — the object's ground-truth world pose at scan time (for later mesh comparisons)
- `coverage_visualization.ply` — see below, this is usually the more useful one to look at

---

## Known-CAD coverage & the base exclusion

Because this project assumes a *known* CAD model (not blind reconstruction), the
most honest "what did we scan" output isn't the raw point cloud — it's **the known
mesh itself, colored by which parts a real capture has confirmed seen**. This
mirrors the reference NBV pattern from `CUDA_Lab_Assignments/Assignment04_startup`:
that assignment's own "reconstructed points" are always pulled from a known
ground-truth grid filtered by ray-cast visibility, never independently re-sensed
data. `coverage_visualization.ply` does the same thing against our continuous mesh:

- **green** — a real capture landed near this known surface point (confirmed seen)
- **gray** — known surface, not yet confirmed seen
- **black** — the object's table-contact base (bottom ~15mm) — **excluded from the
  coverage percentage entirely**, because no camera positioned above the table can
  ever see it. This isn't a planning shortfall, it's a physical constraint of the
  scan setup, and the TA's 90-95% target is against the *observable* surface.

```bash
conda run -n rob_env python -c "
import open3d as o3d
mesh = o3d.io.read_triangle_mesh('captures/full_pipeline/coverage_visualization.ply')
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])
"
```

The base-exclusion threshold (15mm) wasn't picked to hit a number — it was checked
against real scan data first: excluding 10mm already cleared 90% coverage, 15mm
gave a comfortable margin (~92%), and returns diminished past ~30mm. It's a
realistic base-footprint size for the ~192mm-tall object, not an arbitrary cut.

---

## Setup

### 1. Clone shelf_gym into third_party/

```bash
git clone --recurse-submodules -j8 https://github.com/NilsDengler/manipulation_enhanced_map_prediction third_party/shelf_gym_repo
```

### 2. Create the conda environment

```bash
conda create -n rob_env python=3.12
conda activate rob_env
conda install -c conda-forge "cgal<6"
```

### 3. Apply patches and install shelf_gym

```bash
git apply patches/shelf_gym.patch --directory=third_party/shelf_gym_repo
cd third_party/shelf_gym_repo
pip install "pybind11[global]==2.11.1"
bash install.sh
pip install -e . --no-build-isolation
cd ../..
```

### 4. Install CuRobo (**requires an NVIDIA GPU** — see below before you start)

This project uses [CuRobo](https://github.com/nvlabs/curobo) for two things: IK
reachability filtering (Step A) and collision-aware motion planning (Step E,
`MotionGen`). CuRobo compiles custom CUDA kernels at install time — **an NVIDIA GPU
with a working CUDA toolchain is not optional, there is no CPU fallback build.**

#### GPU requirements — read this before spending 20 minutes on a build that won't work

CuRobo's `main`/v2 branch officially requires a **Volta-or-newer NVIDIA GPU**
(Tensor Cores) and Python 3.8–3.10. If your GPU is older (e.g. a Pascal-generation
card like a GTX 10-series — this is what this project was actually developed and
verified against), `main` will not build/run. Use the **`v0.7.8` tag** instead,
which works on older architectures:

```bash
# Pin torch to a Pascal-compatible build FIRST - torch dropped Pascal (sm_61)
# support in its own prebuilt wheels starting at 2.8. This must happen before
# CuRobo's build step or it won't see a usable GPU at all.
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121

pip install ninja   # without this, the build silently falls back to serial
                     # single-core compilation of curobo's CUDA kernel sources -
                     # looks "stuck" for 30+ minutes instead of taking ~15-20 min

git clone --branch v0.7.8 https://github.com/nvlabs/curobo.git third_party/curobo_legacy_rob_env
cd third_party/curobo_legacy_rob_env

# TORCH_CUDA_ARCH_LIST must match your actual GPU's compute capability - 6.1 is
# Pascal (GTX 10-series); check yours with `nvidia-smi --query-gpu=compute_cap
# --format=csv` if unsure, and set this accordingly.
TORCH_CUDA_ARCH_LIST=6.1+PTX pip install -v -e . --no-build-isolation
cd ../..
```

This was verified working **directly in `rob_env`** (Python 3.12) despite CuRobo's
docs listing 3.11+ as "untested" — no separate CuRobo-only environment or
subprocess bridge is needed; `import curobo` alongside PyBullet/shelf_gym in the
same process just works.

#### GPU-specific runtime gotchas already worked around in this codebase

If you're on an older/unusual GPU architecture and see errors that don't match
these, they're probably new — but these three are already fixed in
`nbv_core/motion_planning.py` and `nbv_core/curobo_configs/`, so you shouldn't
need to rediscover them:

1. **`RuntimeError: CUDA error: invalid argument`** inside `curobolib`'s custom
   LBFGS trajectory-optimization kernel, the first time `MotionGen`/trajopt is
   ever exercised (plain `IKSolver` usage doesn't hit this — different kernel
   path). Fix: `nbv_core/curobo_configs/gradient_trajopt.yml` and
   `finetune_trajopt.yml` are patched copies of CuRobo's own task configs with
   every `lbfgs.use_cuda_*_kernel` flag forced off, falling back to CuRobo's own
   plain-PyTorch/JIT implementation of the same algorithm.
2. **`AttributeError: module 'warp' has no attribute 'torch'`** from the default
   `MESH`-type world collision checker (NVIDIA's `warp` library's torch interop
   wasn't working in this environment). Fix: `motion_planning.py` explicitly uses
   `CollisionCheckerType.PRIMITIVE` — also just the correct choice, since this
   project's world obstacles are simple cuboids (table + object bounding box),
   not meshes.
3. **`MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION`** on every single
   planning call. Not a build issue — a real geometry-modeling one: the robot
   base sits flush with the table (both at the same world height), so a
   full-height table collision box always overlaps the `shoulder_link` collision
   sphere even at the arm's resting pose. Fix: the table's *collision* model
   (not the visual one) has its top surface deliberately lowered ~12cm below the
   true table height in `motion_planning.py` — real scan candidates all sit well
   above this line, so this doesn't weaken collision protection where it matters.

### 5. Run

```bash
python nbv_environment.py                                        # sanity-check the base scene
conda run -n rob_env python build_reachability_cache.py          # one-time
conda run -n rob_env python full_pipeline.py --gui --view        # the real thing
```

---

## Known open issue (real, documented, not yet root-caused)

The raw scanned point cloud (`combined_pointcloud.ply`) shows a visible
ribbing/misalignment pattern when many overlapping views are combined and viewed
from certain angles — individual views look correct in isolation, but adjacent
views don't register perfectly against each other. Investigated at length (ruled
out: depth-buffer mm-quantization — was a real bug, fixed, but not the dominant
cause; per-pixel incidence angle; camera-to-object distance; voxel-based
multi-view averaging). Most consistent with per-view camera-pose/arm-settling
precision (a previously-documented ~mm-scale residual in this project), but not
proven to the same standard as the quantization bug. **This is why
`coverage_visualization.ply`, not the raw point cloud, is the recommended way to
look at scan results** — it displays the known-correct mesh geometry filtered by
confirmed visibility, which sidesteps this issue by construction rather than
depending on raw-point registration being perfect.
