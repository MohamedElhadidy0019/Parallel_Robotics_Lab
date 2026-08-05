"""
The whole scan -> reconstruct pipeline in one file, so the full shape is
visible at a glance.

move_camera_to() and select_next_view_pose() were originally two swap-point
stand-ins in this file (PyBullet-IK glide, fixed orbit); both are now real:
move_camera_to is CuRobo MotionGen-backed collision-aware trajectory
planning (nbv_core/motion_planning.py), select_next_view_pose is the real
greedy NBV planner (nbv_planner.py, Step D of the project plan). Grasping is
out of scope (TA's scope update - no grasping needed), so the third
original swap point, compute_grasp_pose(), has been removed rather than
implemented.

Running this script produces a real, working combined point cloud
(mustard-bottle-only, same filtering/back-projection as
scan_and_save_mustard_only.py, no ICP), scanned via the dynamic NBV
planner instead of a fixed orbit. Requires reachability/reachability_cache.npz
to already exist - build it first via:
    conda run -n rob_env python build_reachability_cache.py

Usage:
  python full_pipeline.py            # headless, save the combined cloud
  python full_pipeline.py --view     # also open an Open3D window at the end
"""
import argparse
import os

import numpy as np
import open3d as o3d

from nbv_environment import NBVEnv2
from nbv_core.camera_geometry import capture_rgb_and_depth, compute_intrinsics, backproject_depth, edge_discontinuity_mask
from nbv_core.motion_planning import move_camera_to
from nbv_planner import get_coverage_fraction, select_next_view_pose

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures", "full_pipeline")
EDGE_DISCONTINUITY_THRESHOLD_M = 0.02


def capture_and_backproject_view(env: NBVEnv2, view_index: int) -> np.ndarray:
    """
    Capture one view, keep only mustard-bottle pixels, back-project to world-frame points.
    Real/final, not a placeholder.

    An incidence-angle filter (dropping grazing-angle points relative to the known true
    surface) was tried here and reverted - a controlled test (same raw captures, filter on vs.
    off) showed the points it drops are NOT systematically worse than the points it keeps
    (sometimes the opposite), so per-pixel incidence angle does not actually predict accuracy
    here, despite the real regional correlation between local surface orientation and error -
    see project memory for the full investigation. Left as a documented dead end, not
    reintroduced speculatively.
    """
    _, depth_mm, t_cam_world, q_cam_world, body_ids = capture_rgb_and_depth(
        env._p, env.robot_id, env.camera_link,
        env.camera.width, env.camera.height, env.camera.fov,
        env.camera.near, env.camera.far,
    )

    depth_m = depth_mm.astype(np.float64) / 1000.0
    is_mustard = body_ids == env.obj_id
    is_not_edge_artifact = edge_discontinuity_mask(depth_m, threshold_m=EDGE_DISCONTINUITY_THRESHOLD_M)
    is_mustard_and_trustworthy = is_mustard & is_not_edge_artifact

    fx, fy, cx, cy = compute_intrinsics(env.camera.width, env.camera.height, env.camera.fov)
    points_world_grid = backproject_depth(depth_mm, t_cam_world, q_cam_world, fx, fy, cx, cy)
    points_world = points_world_grid[is_mustard_and_trustworthy]

    print(f"View {view_index}: {len(points_world)} mustard-only points")
    return points_world


def run_scan_and_reconstruct(env: NBVEnv2) -> np.ndarray:
    accumulated_points: list[np.ndarray] = []
    view_index = 0
    while True:
        next_view = select_next_view_pose(env, accumulated_points, view_index)
        if next_view is None:
            break
        t_target_world, q_target_world = next_view

        reached, t_achieved_world = move_camera_to(env, t_target_world, q_target_world)
        if not reached:
            pose_error_m = float(np.linalg.norm(t_achieved_world - t_target_world))
            print(f"View {view_index}: skipped - camera settled {pose_error_m * 1000:.1f}mm "
                  f"from the intended pose (likely beyond the arm's reach here)")
        else:
            accumulated_points.append(capture_and_backproject_view(env, view_index))

        view_index += 1

    return np.concatenate(accumulated_points, axis=0) if accumulated_points else np.zeros((0, 3))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gui", action="store_true", help="Show the PyBullet GUI (watch the robot move).")
    parser.add_argument("--view", action="store_true", help="Open an Open3D window with the combined cloud at the end.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    env = NBVEnv2(render=args.gui)

    points_world = run_scan_and_reconstruct(env)
    print(f"\nReconstructed {len(points_world)} points total")
    coverage_fraction = get_coverage_fraction(env)
    if coverage_fraction is not None:
        print(f"Final known-CAD surface coverage: {coverage_fraction * 100:.1f}%")

    npy_path = os.path.join(OUTPUT_DIR, "combined_pointcloud.npy")
    np.save(npy_path, points_world)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_world)
    ply_path = os.path.join(OUTPUT_DIR, "combined_pointcloud.ply")
    o3d.io.write_point_cloud(ply_path, pcd)

    # Saved alongside the cloud (same convention as scan_and_save_mustard_only.py) so
    # evaluate_reconstruction.py can place the ground-truth mesh in this exact scan's world
    # frame - object placement is deterministic in practice but this avoids ever assuming so.
    t_obj_world, q_obj_world = env._p.getBasePositionAndOrientation(env.obj_id)
    pose_path = os.path.join(OUTPUT_DIR, "object_pose.npz")
    np.savez(pose_path, t_obj_world=np.array(t_obj_world), q_obj_world=np.array(q_obj_world))
    print(f"Saved point cloud to {npy_path} and {ply_path}, object pose to {pose_path}")

    if args.view:
        # PyBullet's EGL renderer (used for every depth capture above) and Open3D's
        # interactive viewer both want exclusive access to the same GPU/X resource in
        # one process - release PyBullet's hold on it first or draw_geometries() fails
        # with a GLX BadAccess error.
        env.close()
        print("Opening Open3D viewer (drag to rotate, scroll to zoom, close the window to continue)...")
        o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()
