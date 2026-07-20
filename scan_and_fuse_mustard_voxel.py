"""
Same reachability-filtered orbit scan as scan_and_save_mustard_only.py, but
instead of just concatenating each view's raw points, fuses them into an
OccupancyVoxelGrid (nbv_core.voxel_grid) - one point per voxel regardless of
how many raw depth pixels landed there. This is the fix for the "terracing"
artifact visible when zooming into the raw concatenated cloud: each view
samples the continuous surface on its own discrete pixel grid from a
different angle, so even a perfectly-posed camera would produce points that
don't exactly coincide between views. Voxel fusion collapses all of that
down to one point per small neighborhood instead of keeping every
raw, slightly-offset sample.

Requires scan_and_save_mustard_only.py's underlying orbit/reachability
machinery (move_through_orbit, already in nbv_environment.py) - this script
doesn't depend on scan_and_save_mustard_only.py having been run first, it
does its own scan.

Usage:
  python scan_and_fuse_mustard_voxel.py
"""
import os

import numpy as np

from nbv_environment import NBVEnv2
from nbv_core.camera_geometry import capture_rgb_and_depth, compute_intrinsics, backproject_depth, edge_discontinuity_mask
from nbv_core.voxel_grid import OccupancyVoxelGrid

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures", "scan_and_save_mustard_only")
NUM_VIEWS = 20
ORBIT_HEIGHT_M = 1.2
VOXEL_SIZE_M = 0.003
GRID_MARGIN_M = 0.02


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    env = NBVEnv2(render=False)

    aabb_min, aabb_max = env._p.getAABB(env.obj_id)
    grid = OccupancyVoxelGrid.from_aabb(aabb_min, aabb_max, margin=GRID_MARGIN_M, voxel_size=VOXEL_SIZE_M)
    print(f"Voxel grid: size={grid.size} voxel_size={VOXEL_SIZE_M} total_cells={np.prod(grid.size)}")

    def integrate_view(view_index: int) -> None:
        _, depth_mm, t_cam_world, q_cam_world, body_ids = capture_rgb_and_depth(
            env._p, env.robot_id, env.camera_link,
            env.camera.width, env.camera.height, env.camera.fov, env.camera.near, env.camera.far,
        )
        depth_m = depth_mm.astype(np.float64) / 1000.0
        is_mustard = body_ids == env.obj_id
        is_ok = is_mustard & edge_discontinuity_mask(depth_m, threshold_m=0.02)

        fx, fy, cx, cy = compute_intrinsics(env.camera.width, env.camera.height, env.camera.fov)
        points_world_grid = backproject_depth(depth_mm, t_cam_world, q_cam_world, fx, fy, cx, cy)
        points_world = points_world_grid[is_ok]

        grid.integrate_points(points_world, t_cam_world, max_range=env.camera.far)
        print(f"View {view_index}: {len(points_world)} raw mustard points integrated - "
              f"occupied voxels so far: {grid.get_occupied_count()}")

    env.move_through_orbit(n_views=NUM_VIEWS, height=ORBIT_HEIGHT_M, on_stop=integrate_view)

    fused_points = grid.get_occupied_points()
    fused_path = os.path.join(OUTPUT_DIR, "fused_voxel_pointcloud.npy")
    np.save(fused_path, fused_points)
    print(f"\nFused point cloud: {len(fused_points)} points (one per occupied voxel) -> {fused_path}")

    t_obj_world, q_obj_world = env._p.getBasePositionAndOrientation(env.obj_id)
    np.savez(
        os.path.join(OUTPUT_DIR, "object_pose.npz"),
        t_obj_world=np.array(t_obj_world),
        q_obj_world=np.array(q_obj_world),
    )

    print("Done.")


if __name__ == "__main__":
    main()
