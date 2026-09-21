# Module development: NBV planner

This is the module development environment for `Parallel_Robotics_Lab`, branch `dev`.
It runs the Next-Best-View planner against PyBullet, on the host, in a conda env. It is
not the ROS container. The same `nbv_planner/` package drives both; only the robot
interface differs.

## Two ways this code runs

| | module dev (here) | ROS 2 |
| --- | --- | --- |
| entry | `python main.py <object>` | `ros2 run nbv_planner_ros inspection_node` |
| robot | `sim/env.py` `SteveSimEnv`, PyBullet | `ros/nbv_planner_ros/.../ros_robot.py` `RosRobot`, Gazebo |
| env | conda `rob_env`, python 3.12 | container, python 3.10 |
| scene | YCB object on a table, spawned in-process | Gazebo world, table and object spawned by `steve_sim_prep` |

`nbv_planner/pipeline.py` `run_inspection()` is shared. It takes a robot object and
never imports a simulator. Anything added to one robot interface must exist on the
other: `capture_observation`, `base_pose`, `table_aabb`, `current_arm_joints`,
`execute_trajectory`, `object_mesh_world`, `intrinsics`, `arm_joint_names`.

## Environment

```bash
source env.sh          # activates rob_env, points CUDA_HOME at the conda prefix
```

To rebuild it from scratch, note the channel: plain `-c nvidia` installs `nvcc` without
`cuda_runtime.h` and cuRobo then fails to compile.

```bash
conda create -n rob_env python=3.12 -y && conda activate rob_env
conda install -y -c "nvidia/label/cuda-12.1.1" cuda-toolkit
export CUDA_HOME=$CONDA_PREFIX && export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"

pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install numpy==2.5.1 scipy==1.18.0 open3d==0.19.0 trimesh==5.0.0 \
            rerun-sdk==0.36.1 pybullet==3.2.7 warp-lang==1.16.0 ninja==1.13.0
pip install --no-build-isolation -e third_party/curobo_legacy_rob_env
pip install opencv-python hydra-core omegaconf==2.3.1 iopath==0.1.10
pip install --no-deps --no-build-isolation "git+https://github.com/facebookresearch/sam2.git"
```

`--no-deps` on SAM2 is required: it declares an unpinned `torch>=2.5.1` floor it does not
exercise, and honouring it replaces the pinned 2.4.1+cu121 and breaks cuRobo with an
`undefined symbol` ImportError.

## Running

```bash
python main.py YcbMustardBottle --mode cad --views 8 --gui --viz
python main.py YcbMustardBottle --mode scan --segmenter sam --scan-views 8
```

- `--mode cad` scores views against the ground-truth CAD mesh.
- `--mode scan` has no CAD: it segments the object over a ring of views, builds a mesh
  from the points, and scores against that.
- `--mode both` scans, checks the CAD against the scan, then uses the CAD.
- `--gui` is the PyBullet window, `--viz` (alias `--rerun-viz`) is the Rerun viewer.

Outputs land in `captures/`: `scan_*.ply` (accumulated cloud), `coverage_*.ply` (mesh
coloured by coverage), `discovery_*.ply` and `object_mesh_*.ply` in scan mode.

## Pipeline

1. **Start pose** `_reach_start_pose` — cuRobo plans from the current joints to a camera
   position and look-at point given in the robot base frame. 3 attempts, verified to
   15 mm / 0.05 rad.
2. **Discovery** (scan/both only) `_discover_object` — segment from the start pose, then
   visit a ring of azimuths, merging each detection into an `ObjectEstimate`.
3. **Target surface** — sample ~4000 points with normals off the reference mesh, into a
   `CoverageTracker`.
4. **Candidates** — hemisphere of camera poses around the object, filtered by cuRobo IK.
5. **NBV loop** — CUDA ray-scoring picks the highest-gain unvisited pose, cuRobo plans a
   batch, the arm drives there, RGB-D is captured and matched against the tracker.

Detection is depth-first: RANSAC table plane, DBSCAN cluster, that cluster's 2D box as a
**prompt** for SAM2, then a min-area rectangle on the convex hull of the footprint.
SAM2 only cleans the mask; it never estimates pose.

## Open problems

Ranked. Each was measured, not guessed.

1. **The bottle cap never reaches coverage.** Of 127 cap samples, 49 have no captured
   point within `DEFAULT_SEEN_DISTANCE_THRESHOLD_M` (8 mm) and 51 more fail the normal
   consistency test because points land behind the surface. Below 95% of object height,
   zero samples fail; the body matches at 0.3 mm. Cause is grazing incidence: every
   candidate elevation is 20-55°, so the cap's side collar is never viewed squarely.
2. **16% of the reconstruction cloud is not the object.** `_capture_object_cloud` keeps
   anything above the table within `WORKSPACE_RADIUS_M` (0.25 m) of the object centre.
   Measured 63,499 stray points at a median 122 mm from the bottle. The radius is a
   hardcoded constant; the bottle's XY half-diagonal is 61 mm. Derive it from
   `mesh_world.bounds` or `estimate.box.size` instead. Apply the filter to the saved
   cloud only, never to what feeds `tracker.update()`, or coverage inflates by
   construction.
3. **Poisson loses 4.4 mm of cap height.** The scan-mode mesh matches CAD to 0.6 mm in
   width and depth but is 246.1 mm tall against 250.6 mm. Two causes in
   `ObjectEstimate.surface_mesh`: normals estimated with a 1 cm radius on a 2 cm cap, and
   `density_quantile=0.02` deleting the sparsest region, which is the cap. TSDF fusion
   (`o3d.pipelines.integration.ScalableTSDFVolume`) is the better fit since the inputs
   are already posed depth frames; it would replace that one method.
4. **The segmentation mask is discarded in the NBV loop.** Discovery segments well, then
   `del segmenter` runs and `_capture_object_cloud` backprojects raw depth.
5. **`ObjectEstimate.converged` is dead code.** `center_shift` is printed and logged to
   Rerun but never gates anything; the discovery ring always runs all `--scan-views`.
6. **The 25° ring elevation is effectively unreachable.** `ring_viewpoints` builds 6
   candidates per slot ordered 40°, 40°+5cm, 55°, 55°+5cm, 25°, 25°+5cm, and
   `object_scan.py` slices `[:3]`.

## History worth knowing

- A `blocker_spheres` argument was added to `score_candidate_views` to model the arm
  occluding its own camera, then removed from `pipeline.py` because cuRobo's collision
  spheres are deliberately larger than the arm and vetoed legitimate views. The function
  `sphere_blocked_mask` and `tests/test_self_occlusion.py` remain.
- On the ROS side, `cabinet_link`'s 5 packaged collision spheres leave 38% of the robot's
  base unmodelled, and 87% of its top 70 mm. `robot_model.refit_collision_spheres` fills
  the box from the live mesh, but the resulting spheres overrun the real box by up to
  56 mm and put the configured start pose out of IK reach. It is off by default,
  opt-in via `refit_collision_links` in `inspection_config.yaml`. The start poses need
  retuning before it can be turned on.

## Conventions

- Self-documenting code. No verbose comments, no em dashes, no emojis.
- Constants live in `nbv_planner/config.py`, not inline.
- `pipeline.py` must stay simulator-agnostic.
- After changing anything under `nbv_planner/` or `ros/`, run
  `./scripts/sync_ros_workspace.sh` to push it into the container workspace.
