"""End-to-end autonomous Next-Best-View (NBV) scanning loop with cuRobo and custom CUDA scoring.

Usage:
    python main.py
    python main.py YcbMustardBottle --views 8
    python main.py YcbChipsCan --views 10 --gui
    python main.py YcbMustardBottle --viz
"""

import argparse
import os
import time
import warnings
import numpy as np
import open3d as o3d
import torch

# Suppress noisy library warnings (gymnasium box precision, torch arch list, rerun, deprecations)
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5;8.0;8.6;8.9;9.0")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", module="gymnasium")
warnings.filterwarnings("ignore", module="torch")
warnings.filterwarnings("ignore", module="rerun")

from nbv_core.camera import (
    backproject_depth,
    capture_rgbd,
    transform_points,
)
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
    build_coverage_colored_mesh,
    load_ycb_mesh,
    sample_surface_points_and_normals,
    transform_mesh,
)
from nbv_core.motion_planning import move_camera_to_batch
from nbv_core.ray_scoring import score_candidate_views
from nbv_core.reachability import ik_filter, sample_candidate_camera_poses
from nbv_core.sim_env import SimEnv, ycb_names
from nbv_core.viz import NBVVisualizer


def _capture_object_cloud(env: SimEnv):
    """Capture RGB-D from camera link and return filtered world-frame object points + raw frames."""
    link_state = env._p.getLinkState(env.robot_id, env.camera_link, physicsClientId=env.client_id)
    t_cam = np.array(link_state[0])

    rgb, depth_m, view_matrix, _ = capture_rgbd(
        t_cam, env.obj_pos, WORLD_UP_Z, env.intrinsics, physics_client_id=env.client_id
    )
    pts_cam, _ = backproject_depth(
        depth_m,
        env.intrinsics,
        rgb=rgb,
        min_depth=ROBOT_SELF_FILTER_MIN_DEPTH_M,
        max_depth=env.intrinsics.far * 0.9,
        drop_edges=True,
    )
    if pts_cam.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32), rgb, depth_m, view_matrix, t_cam

    T_world_cam = np.linalg.inv(view_matrix) @ T_OPENGL_OPTICAL
    pts_world = transform_points(pts_cam, T_world_cam)

    # Real-world tabletop filter: drop points below table surface, keep workspace XY radius (no height limit)
    is_above_table = pts_world[:, 2] >= (env.table_top_z + TABLE_CLEARANCE_MARGIN_M)
    in_workspace_xy = np.linalg.norm(pts_world[:, :2] - env.obj_pos[:2], axis=-1) < WORKSPACE_RADIUS_M

    return pts_world[is_above_table & in_workspace_xy], rgb, depth_m, view_matrix, t_cam


def _save_ply(path: str, points: np.ndarray) -> None:
    """Save raw point cloud to ASCII PLY."""
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for p in points:
            f.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")


def run_nbv_scan(
    obj_name: str = DEFAULT_YCB_OBJECT,
    max_views: int = 8,
    target_coverage: float = 0.95,
    n_surface_samples: int = 4000,
    gui: bool = False,
    viz: bool = False,
) -> dict:
    """Run full autonomous NBV scan pipeline for a YCB object."""
    print(f"=== Autonomous NBV Scan: {obj_name} ===")

    # [Stage 1/4] Scene & Surface Setup
    t_start_scan = time.perf_counter()
    env = SimEnv(render=gui, ycb_object=obj_name)
    try:
        mesh = load_ycb_mesh(obj_name)
        pos, orn = env._p.getBasePositionAndOrientation(env.obj_id, physicsClientId=env.client_id)
        mesh_world = transform_mesh(mesh, np.array(pos), np.array(orn), obj_name=obj_name)
        triangles_world = np.asarray(mesh_world.vertices[mesh_world.faces], dtype=np.float32)

        env.obj_pos = np.array((mesh_world.bounds[0] + mesh_world.bounds[1]) / 2.0, dtype=np.float64)
        base_exclusion_z = float(mesh_world.vertices[:, 2].min()) + BASE_EXCLUSION_HEIGHT_M
        surface_pts, surface_nrm = sample_surface_points_and_normals(
            mesh_world, n_samples=n_surface_samples, base_exclusion_z=base_exclusion_z
        )
        tracker = CoverageTracker(surface_pts, surface_nrm)

        visualizer = NBVVisualizer(obj_name, enabled=viz)
        visualizer.init_scene(env, mesh_world)
        visualizer.update_setup_stage("Scene & Target Surface", "DONE", f"{len(surface_pts):,} samples (>15mm base excluded)")
        print(f"      Target surface: {len(surface_pts)} samples (base excluded >{BASE_EXCLUSION_HEIGHT_M*1000:.0f}mm)")

        # [Stage 2/4] Kinematics & Candidate Filtering
        print("[2/4] Sampling orbit viewpoints & checking cuRobo reachability...")
        r_min, r_max = env.orbit_shell()
        t_cand, q_cand = sample_candidate_camera_poses(
            env.obj_pos,
            radius=(r_min, r_max, N_RADIUS),
            n_azimuth=N_AZIMUTH,
            z_min_world=env.table_top_z,
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

        # [Stage 3/4] Autonomous NBV Scanning Loop
        print(f"[3/4] Running NBV loop (max {max_views} views, target {target_coverage * 100:.0f}%):")
        accumulated_clouds = []
        executed_cams = []
        visited = np.zeros(len(t_cand), dtype=bool)
        views_executed = 0

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

            ok, t_achieved, best_k, opt_ms, exec_ms = move_camera_to_batch(
                env, target_t, target_q, visualizer=visualizer, return_timing=True
            )
            visualizer.update_view_stage("cuRobo Batch Opt", "DONE", opt_ms)
            visualizer.update_view_stage("Arm Waypoint Drive (120Hz)", "DONE", exec_ms)
            visualizer.update_view_stage("RGB-D Camera Capture", "RUNNING")

            # Mark evaluated candidates as visited
            num_to_mark = best_k + 1 if ok else len(top_cand_indices)
            for c_idx in top_cand_indices[:num_to_mark]:
                visited[c_idx] = True

            if not ok:
                print("      Top batch candidates blocked by table collision, trying next candidates...")
                continue

            best_cand_idx = top_cand_indices[best_k]
            cand_score = int(scores[valid_sorted[best_k]])

            t_cap_0 = time.perf_counter()
            cloud, rgb, depth_m, view_matrix, t_cam = _capture_object_cloud(env)
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
            executed_cams.append(t_achieved)
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
                view_matrix=view_matrix,
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

        # [Stage 4/4] Export Results
        os.makedirs("captures", exist_ok=True)
        final_cov = tracker.coverage_fraction()
        total_pts = sum(len(c) for c in accumulated_clouds)

        if accumulated_clouds:
            full_cloud = np.concatenate(accumulated_clouds, axis=0)
            cloud_ply_path = os.path.join("captures", f"scan_{obj_name}.ply")
            _save_ply(cloud_ply_path, full_cloud)

        cov_mesh_o3d = build_coverage_colored_mesh(mesh_world, tracker, base_exclusion_z=base_exclusion_z)
        cov_mesh_path = os.path.join("captures", f"coverage_{obj_name}.ply")
        o3d.io.write_triangle_mesh(cov_mesh_path, cov_mesh_o3d)

        print(f"[4/4] Complete: {views_executed} views executed | Reconstructed {total_pts:,} pts | Final Coverage: {final_cov * 100:.1f}%")
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
        }

    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("object", nargs="?", default=DEFAULT_YCB_OBJECT, choices=ycb_names())
    parser.add_argument("--views", type=int, default=8, help="Maximum number of scan viewpoints")
    parser.add_argument("--target-cov", type=float, default=0.95, help="Target coverage fraction (0-1)")
    parser.add_argument("--samples", type=int, default=4000, help="Surface sampling resolution")
    parser.add_argument("--gui", action="store_true", help="Enable PyBullet live simulation window")
    parser.add_argument("--rerun-viz", "--viz", dest="viz", action="store_true", help="Enable live Rerun 3D viewer")
    args = parser.parse_args()

    run_nbv_scan(
        obj_name=args.object,
        max_views=args.views,
        target_coverage=args.target_cov,
        n_surface_samples=args.samples,
        gui=args.gui,
        viz=args.viz,
    )