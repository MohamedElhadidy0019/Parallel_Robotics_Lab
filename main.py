"""End-to-end autonomous Next-Best-View (NBV) scanning loop with cuRobo and custom CUDA scoring.

Usage:
    python main.py
    python main.py YcbMustardBottle --views 8
    python main.py YcbChipsCan --views 10 --gui
    python main.py YcbMustardBottle --viz
    python main.py YcbMustardBottle --start-pos 0.5 0.0 1.15 --look-at 0.785 0.0 0.85
"""

import argparse
import io
import os
import sys
import time
import warnings
import numpy as np
import open3d as o3d
import torch

def _silence_c_output():
    """Route low-level C/C++ stdout and stderr (PyBullet/OpenGL threads) to /dev/null while keeping Python sys.stdout / sys.stderr intact."""
    if getattr(sys, "_c_output_silenced", False):
        return
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        real_out = os.dup(1)
        real_err = os.dup(2)
        sys.stdout = io.TextIOWrapper(open(real_out, "wb", buffering=0), encoding="utf-8", write_through=True)
        sys.stderr = io.TextIOWrapper(open(real_err, "wb", buffering=0), encoding="utf-8", write_through=True)
        null_fd = os.open(os.devnull, os.O_RDWR)
        os.dup2(null_fd, 1)
        os.dup2(null_fd, 2)
        os.close(null_fd)
        sys._c_output_silenced = True
    except Exception:
        pass

_silence_c_output()

# Suppress noisy library warnings (gymnasium box precision, torch arch list, rerun, deprecations)
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5;8.0;8.6;8.9;9.0")
os.environ.setdefault("RERUN_LOG", "warn")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", module="gymnasium")
warnings.filterwarnings("ignore", module="torch")
warnings.filterwarnings("ignore", module="rerun")

from nbv_planner.camera import (
    backproject_depth,
    transform_points,
)
from nbv_planner.config import (
    BASE_EXCLUSION_HEIGHT_M,
    BASE_LINK,
    DEFAULT_YCB_OBJECT,
    EE_LINK,
    MAX_POSE_ERROR_M,
    N_AZIMUTH,
    N_RADIUS,
    ROBOT_SELF_FILTER_MIN_DEPTH_M,
    START_CAMERA_POSITION_BASE,
    START_LOOK_AT_BASE,
    START_ROTATION_TOL_RAD,
    START_SAFETY_RADIUS_M,
    TABLE_CLEARANCE_MARGIN_M,
    URDF_PATH,
    WORKSPACE_RADIUS_M,
    ycb_names,
)
from nbv_planner.coverage import (
    CoverageTracker,
    build_coverage_colored_mesh,
    load_ycb_mesh,
    sample_surface_points_and_normals,
    transform_mesh,
)
from nbv_planner.motion_planning import build_world_config, plan_motion_batch
from nbv_planner.ray_scoring import score_candidate_views
from nbv_planner.reachability import ik_filter, sample_candidate_camera_poses
from nbv_planner.start_pose import pose_errors, start_camera_pose_world
from nbv_planner.viz import NBVVisualizer
from sim.env import SteveSimEnv as SimEnv


def _capture_object_cloud(env: SimEnv):
    """Capture RGB-D at the measured camera pose and return filtered world-frame object points + the observation."""
    observation = env.capture_observation()
    pts_cam, _ = backproject_depth(
        observation.depth_m,
        env.intrinsics,
        rgb=observation.rgb,
        min_depth=ROBOT_SELF_FILTER_MIN_DEPTH_M,
        max_depth=env.intrinsics.far * 0.9,
        drop_edges=True,
    )
    if pts_cam.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32), observation

    pts_world = transform_points(pts_cam, observation.world_from_camera)

    # Real-world tabletop filter: drop points below table surface, keep workspace XY radius (no height limit)
    is_above_table = pts_world[:, 2] >= (env.table_surface_z + TABLE_CLEARANCE_MARGIN_M)
    in_workspace_xy = np.linalg.norm(pts_world[:, :2] - env.obj_pos[:2], axis=-1) < WORKSPACE_RADIUS_M

    return pts_world[is_above_table & in_workspace_xy], observation


def _reach_start_pose(
    env: SimEnv,
    start_position_base: np.ndarray,
    look_at_base: np.ndarray,
    visualizer: NBVVisualizer,
):
    """Plan with cuRobo to the start pose (camera facing the look-at point), drive there, and verify.

    Returns (camera position world, look-at point world, observation captured at the start pose).
    """
    t_base, q_base = env.base_pose()
    t_start, q_start, look_at_world = start_camera_pose_world(start_position_base, look_at_base, t_base, q_base)

    # The object is not known yet: keep a conservative box around the look-at point clear of the arm.
    lo, hi = env.table_aabb
    world_cfg = build_world_config(
        t_base_world=t_base,
        q_base_world_xyzw=q_base,
        table_lo=lo,
        table_hi=hi,
        t_obj_world=look_at_world,
        q_obj_world_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        obj_dims=np.full(3, 2.0 * START_SAFETY_RADIUS_M),
    )
    ok, _, trajectory, opt_ms = plan_motion_batch(
        t_targets_world=t_start[None, :],
        q_targets_world=q_start[None, :],
        current_joints=env.current_arm_joints(),
        t_base_world=t_base,
        q_base_world_xyzw=q_base,
        world_config=world_cfg,
        arm_joint_names=env.arm_joint_names,
        enable_graph=False,
    )
    if not ok or trajectory is None:
        raise RuntimeError(
            f"cuRobo found no collision-free path to the start pose {np.round(start_position_base, 3).tolist()} "
            f"looking at {np.round(look_at_base, 3).tolist()} (base frame). Adjust --start-pos / --look-at."
        )

    t_exec_0 = time.perf_counter()
    env.execute_trajectory(trajectory, visualizer=visualizer)
    exec_ms = (time.perf_counter() - t_exec_0) * 1000.0
    pos_err, rot_err = pose_errors(env.camera_world_transform(), t_start, q_start)
    if pos_err > MAX_POSE_ERROR_M or rot_err > START_ROTATION_TOL_RAD:
        raise RuntimeError(
            f"Arm stopped {pos_err * 1000:.1f} mm / {np.degrees(rot_err):.1f} deg away from the start pose"
        )

    observation = env.capture_observation()
    distance = np.linalg.norm(look_at_world - t_start)
    offset = t_start - look_at_world
    elevation = np.degrees(np.arctan2(offset[2], np.linalg.norm(offset[:2])))
    print(
        f"      Reached start pose: camera at ({t_start[0]:.3f}, {t_start[1]:.3f}, {t_start[2]:.3f}) facing "
        f"({look_at_world[0]:.3f}, {look_at_world[1]:.3f}, {look_at_world[2]:.3f}) | {distance:.2f} m, "
        f"{elevation:.0f} deg elevation | error {pos_err * 1000:.1f} mm / {np.degrees(rot_err):.1f} deg | "
        f"plan {opt_ms:.0f} ms, drive {exec_ms:.0f} ms"
    )
    return t_start, look_at_world, observation


def run_nbv_scan(
    obj_name: str = DEFAULT_YCB_OBJECT,
    max_views: int = 8,
    target_coverage: float = 0.95,
    n_surface_samples: int = 4000,
    gui: bool = False,
    viz: bool = False,
    start_position_base: np.ndarray = START_CAMERA_POSITION_BASE,
    look_at_base: np.ndarray = START_LOOK_AT_BASE,
) -> dict:
    """Run full autonomous NBV scan pipeline for a YCB object."""
    print(f"=== Autonomous NBV Scan: {obj_name} ===")
    env = SimEnv(render=gui, ycb_object=obj_name)
    try:
        visualizer = NBVVisualizer(obj_name, enabled=viz)
        visualizer.init_scene(env)

        # [Stage 1/5] Start Pose: face the object before the planner runs
        print("[1/5] Moving arm to start pose with cuRobo...")
        visualizer.update_setup_stage("Start Pose", "RUNNING", "cuRobo planning...")
        start_position, look_at_world, start_observation = _reach_start_pose(
            env, start_position_base, look_at_base, visualizer
        )
        visualizer.log_start_pose(start_observation, look_at_world)
        visualizer.update_setup_stage("Start Pose", "DONE", "reached ({:.2f}, {:.2f}, {:.2f})".format(*start_position))

        # [Stage 2/5] Target Object & Surface Sampling
        print(f"[2/5] Loading CAD & sampling target surface: {obj_name}...")
        # PyBullet reports the inertial (COM) frame, so the URDF inertial offset must be undone.
        mesh = load_ycb_mesh(obj_name)
        pos, orn = env._p.getBasePositionAndOrientation(env.obj_id, physicsClientId=env.client_id)
        mesh_world = transform_mesh(mesh, np.array(pos), np.array(orn), obj_name=obj_name, is_inertial_frame=True)
        triangles_world = np.asarray(mesh_world.vertices[mesh_world.faces], dtype=np.float32)

        env.obj_pos = np.array((mesh_world.bounds[0] + mesh_world.bounds[1]) / 2.0, dtype=np.float64)
        base_exclusion_z = float(mesh_world.vertices[:, 2].min()) + BASE_EXCLUSION_HEIGHT_M
        surface_pts, surface_nrm = sample_surface_points_and_normals(
            mesh_world, n_samples=n_surface_samples, base_exclusion_z=base_exclusion_z
        )
        tracker = CoverageTracker(surface_pts, surface_nrm)

        visualizer.log_cad_mesh(mesh_world)
        visualizer.update_setup_stage("Scene & Target Surface", "DONE", f"{len(surface_pts):,} samples")
        print(f"      Target surface: {len(surface_pts):,} samples")

        # [Stage 3/5] Kinematics & Candidate Filtering
        print("[3/5] Sampling orbit viewpoints & checking cuRobo reachability...")
        r_min, r_max = env.orbit_shell()
        t_cand, q_cand = sample_candidate_camera_poses(
            env.obj_pos,
            radius=(r_min, r_max, N_RADIUS),
            n_azimuth=N_AZIMUTH,
            z_min_world=env.table_surface_z + 0.02,
        )
        t_base, q_base = env.base_pose()
        visualizer.update_setup_stage("Orbit Viewpoints", "DONE", f"{len(t_cand)} generated")
        visualizer.update_setup_stage("cuRobo IK Reachability", "RUNNING", "checking kinematics...")

        t0_ik = time.perf_counter()
        reachable, _ = ik_filter(
            URDF_PATH, BASE_LINK, EE_LINK, t_cand, q_cand, t_base, q_base
        )
        ik_ms = (time.perf_counter() - t0_ik) * 1000.0
        reach_idx = np.where(reachable)[0]
        print(f"      Candidates: {len(t_cand)} generated -> {len(reach_idx)} reachable by UR5 ({ik_ms:.0f} ms)")
        visualizer.update_setup_stage("cuRobo IK Reachability", "DONE", f"{len(reach_idx)} reachable ({ik_ms:.0f} ms)")

        if len(reach_idx) == 0:
            print("Error: No reachable candidate viewpoints found.")
            return {"coverage": 0.0, "views": 0, "points": 0}

        # [Stage 4/5] Autonomous NBV Scanning Loop
        print(f"[4/5] Running NBV loop (max {max_views} views, target {target_coverage * 100:.0f}%):")
        accumulated_clouds = []
        visited = np.zeros(len(t_cand), dtype=bool)
        views_executed = 0

        # Construct cuRobo collision world: table slab + axis-aligned box around the CAD mesh
        lo, hi = env.table_aabb
        world_cfg = build_world_config(
            t_base_world=t_base,
            q_base_world_xyzw=q_base,
            table_lo=lo,
            table_hi=hi,
            t_obj_world=env.obj_pos,
            q_obj_world_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            obj_dims=mesh_world.bounds[1] - mesh_world.bounds[0],
        )

        while views_executed < max_views:
            cov_current = tracker.coverage_fraction()
            if cov_current >= target_coverage:
                print(f"      Target coverage {target_coverage * 100:.1f}% reached.")
                break

            unvisited_reach = [i for i in reach_idx if not visited[i]]
            if not unvisited_reach:
                print("      All reachable viewpoints visited.")
                break

            unseen_pts = tracker.get_unseen_points()
            unseen_nrm = tracker.get_unseen_normals()
            if len(unseen_pts) == 0:
                print("      100% surface coverage confirmed.")
                break

            # Parallel GPU ray scoring
            cand_positions = t_cand[unvisited_reach]
            visualizer.start_view(view_idx=views_executed + 1, max_views=max_views)

            t0 = time.perf_counter()
            scores, _ = score_candidate_views(
                cand_positions, unseen_pts, unseen_nrm, triangles_world
            )
            score_ms = (time.perf_counter() - t0) * 1000.0
            visualizer.update_view_stage("CUDA Ray Scoring", "DONE", score_ms)
            visualizer.update_view_stage("cuRobo Batch Opt", "RUNNING")

            # Take top K non-zero gain candidates for batched planning (cap at 4 for VRAM efficiency)
            sorted_local = np.argsort(scores)[::-1]
            valid_sorted = [l for l in sorted_local if scores[l] > 0][:4]
            if not valid_sorted:
                print("      No remaining candidates with positive gain.")
                break

            top_cand_indices = [unvisited_reach[l] for l in valid_sorted]
            target_t = t_cand[top_cand_indices]
            target_q = q_cand[top_cand_indices]

            # Plan collision-free trajectory using cuRobo on GPU
            ok, best_k, trajectory, opt_ms = plan_motion_batch(
                t_targets_world=target_t,
                q_targets_world=target_q,
                current_joints=env.current_arm_joints(),
                t_base_world=t_base,
                q_base_world_xyzw=q_base,
                world_config=world_cfg,
                arm_joint_names=env.arm_joint_names,
                enable_graph=False,
            )
            visualizer.update_view_stage("cuRobo Batch Opt", "DONE", opt_ms)
            visualizer.update_view_stage("Arm Waypoint Drive (120Hz)", "RUNNING")

            if ok and trajectory is not None:
                t_achieved, exec_ms = env.execute_trajectory(trajectory, visualizer=visualizer)
                err = float(np.linalg.norm(t_achieved - target_t[best_k]))
                reached = err <= MAX_POSE_ERROR_M
            else:
                exec_ms = 0.0
                reached = False

            visualizer.update_view_stage("Arm Waypoint Drive (120Hz)", "DONE", exec_ms)
            visualizer.update_view_stage("RGB-D Camera Capture", "RUNNING")

            # Mark evaluated candidates as visited
            num_to_mark = best_k + 1 if reached else len(top_cand_indices)
            for c_idx in top_cand_indices[:num_to_mark]:
                visited[c_idx] = True

            if not reached:
                print("      Top batch candidates blocked by table collision, trying next candidates...")
                continue

            best_cand_idx = top_cand_indices[best_k]
            cand_score = int(scores[valid_sorted[best_k]])

            t_cap_0 = time.perf_counter()
            cloud, observation = _capture_object_cloud(env)
            cap_ms = (time.perf_counter() - t_cap_0) * 1000.0
            visualizer.update_view_stage("RGB-D Camera Capture", "DONE", cap_ms)
            visualizer.update_view_stage("KDTree Coverage Match", "RUNNING")

            if len(cloud) == 0:
                continue

            t_cov_0 = time.perf_counter()
            newly_seen = tracker.update(cloud)
            cov_ms = (time.perf_counter() - t_cov_0) * 1000.0
            visualizer.update_view_stage("KDTree Coverage Match", "DONE", cov_ms)

            accumulated_clouds.append(cloud)
            views_executed += 1
            cov_now = tracker.coverage_fraction()

            visualizer.complete_view(
                cand_idx=best_cand_idx,
                gain=cand_score,
                newly_seen=newly_seen,
                cov_pct=cov_now * 100.0,
            )

            # Live streaming to Rerun (3D views, cameras, point cloud)
            visualizer.log_step(
                step_idx=views_executed,
                cand_idx=best_cand_idx,
                gain=cand_score,
                newly_seen=newly_seen,
                score_ms=score_ms,
                view_matrix=observation.view_matrix,
                intrinsics=env.intrinsics,
                tracker=tracker,
                new_cloud=cloud,
            )

            print(
                f"  [View {views_executed}/{max_views}] Cand #{best_cand_idx:03d} "
                f"(gain={cand_score:4d}) -> +{newly_seen:4d} seen | "
                f"Coverage: {cov_now * 100:4.1f}% | Scored in {score_ms:4.0f}ms"
            )

            # Flush GPU cache between views to maintain clean VRAM headroom
            torch.cuda.empty_cache()

        # [Stage 5/5] Export Results
        os.makedirs("captures", exist_ok=True)
        final_cov = tracker.coverage_fraction()
        total_pts = sum(len(c) for c in accumulated_clouds)

        if accumulated_clouds:
            full_cloud = np.concatenate(accumulated_clouds, axis=0)
            cloud_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(full_cloud.astype(np.float64)))
            o3d.io.write_point_cloud(os.path.join("captures", f"scan_{obj_name}.ply"), cloud_o3d)

        cov_mesh_o3d = build_coverage_colored_mesh(mesh_world, tracker, base_exclusion_z=base_exclusion_z)
        cov_mesh_path = os.path.join("captures", f"coverage_{obj_name}.ply")
        o3d.io.write_triangle_mesh(cov_mesh_path, cov_mesh_o3d)

        print(f"[5/5] Complete: {views_executed} views executed | Reconstructed {total_pts:,} pts | Final Coverage: {final_cov * 100:.1f}%")
        print(f"      Saved: captures/scan_{obj_name}.ply & captures/coverage_{obj_name}.ply\n")

        if gui:
            try:
                print("[GUI] Scan complete. Press Enter to close PyBullet window...")
                input()
            except (EOFError, KeyboardInterrupt):
                pass

        return {
            "coverage": final_cov,
            "views": views_executed,
            "points": total_pts,
            "start_pose": start_position.tolist(),
        }

    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("object", nargs="?", default=DEFAULT_YCB_OBJECT, choices=ycb_names())
    parser.add_argument("--views", "--frames", type=int, default=8, help="Maximum number of scan viewpoints")
    parser.add_argument("--target-cov", type=float, default=0.95, help="Target coverage fraction (0-1)")
    parser.add_argument("--samples", type=int, default=4000, help="Surface sampling resolution")
    parser.add_argument("--gui", action="store_true", help="Enable PyBullet live simulation window")
    parser.add_argument("--rerun-viz", "--viz", dest="viz", action="store_true", help="Enable live Rerun 3D viewer")
    parser.add_argument(
        "--start-pos", nargs=3, type=float, metavar=("X", "Y", "Z"), default=START_CAMERA_POSITION_BASE,
        help="Start pose camera position, robot base frame (m)",
    )
    parser.add_argument(
        "--look-at", nargs=3, type=float, metavar=("X", "Y", "Z"), default=START_LOOK_AT_BASE,
        help="Point the camera faces at the start pose (roughly the object), robot base frame (m)",
    )
    args = parser.parse_args()

    run_nbv_scan(
        obj_name=args.object,
        max_views=args.views,
        target_coverage=args.target_cov,
        n_surface_samples=args.samples,
        gui=args.gui,
        viz=args.viz,
        start_position_base=args.start_pos,
        look_at_base=args.look_at,
    )
