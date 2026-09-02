"""Multi-stage CPU vs GPU performance benchmark for autonomous NBV inspection.

Profiles every pipeline stage:
  1. Scene & Surface Setup (CPU / PyBullet)
  2. Orbit Viewpoint Candidate Sampling (CPU / NumPy)
  3. cuRobo IK Reachability Batched Filtering (GPU / cuRobo PyTorch)
  4. Custom Native Ray Scoring Moller-Trumbore (GPU / CUDA Kernel)
  5. cuRobo Collision-Free Trajectory Optimization (GPU / cuRobo)
  6. Camera RGB-D Capture & Tabletop Margin Filter (CPU / PyBullet OpenGL)
  7. Depth Backprojection to 3D Camera/World Points (CPU / NumPy)
  8. KDTree Spatial Coverage Update (CPU / SciPy cKDTree)

Usage:
    python benchmark.py
    python benchmark.py --object YcbMustardBottle --runs 5
    python benchmark.py --all
    python benchmark.py --viz
"""

import argparse
import time
import warnings
import numpy as np
import torch
import rerun as rr

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from nbv_core.camera import backproject_depth, capture_rgbd, transform_points
from nbv_core.config import (
    BASE_EXCLUSION_HEIGHT_M,
    BASE_LINK,
    DEFAULT_YCB_OBJECT,
    EE_LINK,
    N_AZIMUTH,
    N_RADIUS,
    ROBOT_SELF_FILTER_MIN_DEPTH_M,
    TABLE_CLEARANCE_MARGIN_M,
    T_OPENGL_OPTICAL,
    URDF_PATH,
    WORKSPACE_RADIUS_M,
    WORLD_UP_Z,
)
from nbv_core.coverage import (
    CoverageTracker,
    load_ycb_mesh,
    sample_surface_points_and_normals,
    transform_mesh,
)
from nbv_core.motion_planning import _build_world_config, _get_motion_gen
from nbv_core.ray_scoring import score_candidate_views
from nbv_core.reachability import ik_filter, sample_candidate_camera_poses
from nbv_core.sim_env import SimEnv, ycb_names


def time_gpu(fn, n_warmup=1, n_runs=5):
    """Accurately measure CUDA kernel latency using torch.cuda.Event."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    for _ in range(n_runs):
        start_ev.record()
        fn()
        end_ev.record()
        torch.cuda.synchronize()
        times.append(start_ev.elapsed_time(end_ev))

    return np.mean(times), np.min(times), np.max(times)


def time_cpu(fn, n_warmup=1, n_runs=5):
    """Accurately measure CPU function latency using time.perf_counter."""
    for _ in range(n_warmup):
        fn()

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    return np.mean(times), np.min(times), np.max(times)


def run_benchmark(obj_name: str, runs: int = 5, viz: bool = False):
    """Run comprehensive CPU / GPU timing benchmark across all stages."""
    print(f"\n==================================================================")
    print(f"  BENCHMARK: Autonomous NBV Inspection Pipeline ({obj_name})")
    print(f"  Trials per stage: {runs} | GPU: {torch.cuda.get_device_name(0)}")
    print(f"==================================================================\n")

    results = []

    # 1. Scene & Surface Setup (CPU)
    def stage_setup():
        env_temp = SimEnv(render=False, ycb_object=obj_name)
        pos, orn = env_temp._p.getBasePositionAndOrientation(env_temp.obj_id, physicsClientId=env_temp.client_id)
        mesh_t = transform_mesh(load_ycb_mesh(obj_name), np.array(pos), np.array(orn), obj_name=obj_name)
        sample_surface_points_and_normals(
            mesh_t, n_samples=3600, base_exclusion_z=env_temp.table_top_z + BASE_EXCLUSION_HEIGHT_M
        )
        env_temp.close()

    mean_t, min_t, max_t = time_cpu(stage_setup, n_warmup=0, n_runs=max(1, min(runs, 3)))
    results.append(("1. Scene Setup & CAD Sampling", "CPU", mean_t, min_t, max_t, "3,600 surface pts"))

    # Instantiate long-lived env for downstream stages
    env = SimEnv(render=False, ycb_object=obj_name)
    pos, orn = env._p.getBasePositionAndOrientation(env.obj_id, physicsClientId=env.client_id)
    mesh_world = transform_mesh(load_ycb_mesh(obj_name), np.array(pos), np.array(orn), obj_name=obj_name)
    triangles_world = np.asarray(mesh_world.vertices)[np.asarray(mesh_world.faces)]
    surface_pts, surface_nrm = sample_surface_points_and_normals(
        mesh_world, n_samples=3600, base_exclusion_z=env.table_top_z + BASE_EXCLUSION_HEIGHT_M
    )
    tracker = CoverageTracker(surface_pts, surface_nrm)

    # 2. Candidate Generation (CPU)
    r_min, r_max = env.orbit_shell()
    def stage_sampling():
        return sample_candidate_camera_poses(
            env.obj_pos, radius=(r_min, r_max, N_RADIUS), n_azimuth=N_AZIMUTH, z_min_world=env.table_top_z
        )

    mean_t, min_t, max_t = time_cpu(stage_sampling, n_warmup=1, n_runs=runs)
    t_cand, q_cand = stage_sampling()
    results.append(("2. Orbit Candidate View Generation", "CPU", mean_t, min_t, max_t, f"{len(t_cand)} poses"))

    # 3. cuRobo IK Reachability (GPU)
    t_base, q_base = env.base_pose()
    def stage_ik():
        return ik_filter(URDF_PATH, BASE_LINK, EE_LINK, t_cand, q_cand, t_base, q_base)

    mean_t, min_t, max_t = time_cpu(stage_ik, n_warmup=1, n_runs=runs)
    reachable, _ = stage_ik()
    reach_idx = np.where(reachable)[0]
    results.append(("3. cuRobo IK Reachability Batch", "GPU", mean_t, min_t, max_t, f"{len(reach_idx)}/{len(t_cand)} reachable"))

    # 4. Custom CUDA Ray Scoring (GPU)
    cand_pos = t_cand[reach_idx]
    unseen_pts = tracker.get_unseen_points()
    unseen_nrm = tracker.get_unseen_normals()
    def stage_ray_scoring():
        score_candidate_views(cand_pos, unseen_pts, unseen_nrm, triangles_world)

    mean_t, min_t, max_t = time_gpu(stage_ray_scoring, n_warmup=2, n_runs=runs * 2)
    scores, _ = score_candidate_views(cand_pos, unseen_pts, unseen_nrm, triangles_world)
    results.append(("4. CUDA Ray Scoring (Moller-Trumbore)", "GPU", mean_t, min_t, max_t, f"{len(cand_pos)} x {len(unseen_pts)} rays"))

    # 5. cuRobo Motion Planning Trajectory Optimization (GPU)
    from curobo.types.math import Pose
    from curobo.types.robot import JointState
    from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig
    from nbv_core.reachability import quaternion_xyzw_to_wxyz, world_poses_to_base_link_frame

    motion_gen = _get_motion_gen(env)
    motion_gen.update_world(_build_world_config(env))
    best_cand = reach_idx[np.argmax(scores)]
    t_tgt_b, q_tgt_b = world_poses_to_base_link_frame(
        t_cand[best_cand:best_cand+1], q_cand[best_cand:best_cand+1], t_base, q_base
    )
    goal = Pose(
        position=motion_gen.tensor_args.to_device(np.ascontiguousarray(t_tgt_b, dtype=np.float32)),
        quaternion=motion_gen.tensor_args.to_device(np.ascontiguousarray(quaternion_xyzw_to_wxyz(q_tgt_b), dtype=np.float32)),
    )
    live = [env._p.getJointState(env.robot_id, i, physicsClientId=env.client_id)[0] for i in env.arm_joint_indices]
    q_start = JointState.from_position(
        motion_gen.tensor_args.to_device(np.asarray(live, dtype=np.float32)[None, :]), joint_names=env.arm_joint_names
    )
    def stage_curobo_plan():
        return motion_gen.plan_single(q_start, goal, MotionGenPlanConfig(max_attempts=15))

    mean_t, min_t, max_t = time_cpu(stage_curobo_plan, n_warmup=1, n_runs=runs)
    results.append(("5. cuRobo Trajectory Optimization", "GPU", mean_t, min_t, max_t, "UR5 collision-free"))

    # 6. RGB-D Capture & Tabletop Filter (CPU/EGL)
    t_cam = t_cand[best_cand]
    def stage_capture():
        return capture_rgbd(t_cam, env.obj_pos, WORLD_UP_Z, env.intrinsics, physics_client_id=env.client_id)

    mean_t, min_t, max_t = time_cpu(stage_capture, n_warmup=1, n_runs=runs)
    rgb, depth_m, view_mat, _ = stage_capture()
    results.append(("6. RGB-D Camera Capture (EGL/PyBullet)", "CPU", mean_t, min_t, max_t, "640x480 frames"))

    # 7. Depth Backprojection (CPU)
    def stage_backproject():
        pts_c, _ = backproject_depth(
            depth_m, env.intrinsics, rgb=rgb, min_depth=ROBOT_SELF_FILTER_MIN_DEPTH_M, max_depth=env.intrinsics.far * 0.9
        )
        T_world_cam = np.linalg.inv(view_mat) @ T_OPENGL_OPTICAL
        pts_w = transform_points(pts_c, T_world_cam)
        is_above = pts_w[:, 2] >= (env.table_top_z + TABLE_CLEARANCE_MARGIN_M)
        in_xy = np.linalg.norm(pts_w[:, :2] - env.obj_pos[:2], axis=-1) < WORKSPACE_RADIUS_M
        return pts_w[is_above & in_xy]

    mean_t, min_t, max_t = time_cpu(stage_backproject, n_warmup=1, n_runs=runs)
    cloud = stage_backproject()
    results.append(("7. Depth Backprojection & Tabletop Filter", "CPU", mean_t, min_t, max_t, f"{len(cloud):,} 3D pts"))

    # 8. KDTree Coverage Update (CPU)
    def stage_kdtree():
        return tracker.update(cloud)

    mean_t, min_t, max_t = time_cpu(stage_kdtree, n_warmup=1, n_runs=runs)
    results.append(("8. KDTree Spatial Coverage Matching", "CPU", mean_t, min_t, max_t, "Radius < 8mm, dev < 45 deg"))

    env.close()

    # Print Formatted Results Table
    total_loop_ms = sum(r[2] for r in results[2:])
    gpu_loop_ms = sum(r[2] for r in results[2:] if r[1] == "GPU")
    cpu_loop_ms = total_loop_ms - gpu_loop_ms

    print(f"+---------------------------------------------+--------+-----------+-----------+------------------------------+")
    print(f"| Pipeline Stage                              | Device | Mean (ms) | Min..Max  | Details / Throughput         |")
    print(f"+---------------------------------------------+--------+-----------+-----------+------------------------------+")
    for name, dev, mean_v, min_v, max_v, details in results:
        print(f"| {name:<43} | {dev:^6} | {mean_v:8.2f}  | {min_v:4.1f}..{max_v:<4.1f} | {details:<28} |")
    print(f"+---------------------------------------------+--------+-----------+-----------+------------------------------+")
    print(f"| Per-View Loop Time (Stages 3-8)             | BOTH   | {total_loop_ms:8.2f}  |    -      | 100.0% of decision cycle     |")
    print(f"|   -> GPU Acceleration Portion (cuRobo+CUDA) | GPU    | {gpu_loop_ms:8.2f}  |    -      | {gpu_loop_ms/total_loop_ms*100:5.1f}% of decision loop     |")
    print(f"|   -> CPU Host Portion (Capture+KDTree)      | CPU    | {cpu_loop_ms:8.2f}  |    -      | {cpu_loop_ms/total_loop_ms*100:5.1f}% of decision loop     |")
    print(f"+---------------------------------------------+--------+-----------+-----------+------------------------------+\n")

    if viz:
        rr.init(f"nbv_benchmark_{obj_name}", spawn=True)
        categories = [r[0] for r in results]
        latencies = [r[2] for r in results]
        rr.log("benchmark/bar_chart", rr.BarChart(latencies))
        doc_lines = [
            f"# Performance Benchmark: `{obj_name}`",
            "",
            "| Stage | Device | Mean Latency | Min..Max | Notes |",
            "|:---|:---:|:---:|:---:|:---|",
        ]
        for name, dev, mean_v, min_v, max_v, details in results:
            doc_lines.append(f"| {name} | **{dev}** | {mean_v:.2f} ms | {min_v:.1f}..{max_v:.1f} ms | {details} |")
        doc_lines.extend([
            "",
            f"- **Decision Loop Latency**: {total_loop_ms:.1f} ms",
            f"- **GPU Workload Share**: {gpu_loop_ms/total_loop_ms*100:.1f}%",
            f"- **CPU Workload Share**: {cpu_loop_ms/total_loop_ms*100:.1f}%",
        ])
        rr.log("benchmark/summary", rr.TextDocument("\n".join(doc_lines), media_type="text/markdown"))
        print("Logged benchmark results to Rerun viewer.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("object", nargs="?", default=None, choices=ycb_names(), help="Target YCB object name")
    parser.add_argument("--object", dest="opt_object", default=None, choices=ycb_names(), help="Target YCB object name")
    parser.add_argument("--runs", type=int, default=5, help="Number of timing trials per stage")
    parser.add_argument("--all", action="store_true", help="Benchmark all supported YCB objects")
    parser.add_argument("--viz", action="store_true", help="Send benchmark graphs to Rerun")
    args = parser.parse_args()

    target_obj = args.opt_object or args.object or DEFAULT_YCB_OBJECT

    if args.all:
        for obj in ycb_names():
            run_benchmark(obj, runs=args.runs, viz=args.viz)
    else:
        run_benchmark(target_obj, runs=args.runs, viz=args.viz)
