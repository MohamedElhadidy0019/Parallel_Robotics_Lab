"""
Runs the same fixed 8-view orbit scan used elsewhere in the project
(NBVEnv2.move_through_orbit), and at each stop walks through the capture
pipeline as separate, visible steps - unlike NBVEnv2.capture_frame(), which
bundles all of this behind one call:

  1. Render the camera and convert its raw depth buffer into real depth in
     millimeters (nbv_core.camera_geometry.capture_rgb_and_depth).
  2. Back-project every pixel of that depth image into a 3D point in world
     coordinates (nbv_core.camera_geometry.backproject_depth).
  3. Drop points we know are bad: too close (e.g. the gripper), too far
     (background - no real surface hit), or sitting on a depth discontinuity
     ("flying pixels" at silhouette edges - see edge_discontinuity_mask).
  4. Save the depth image and the (filtered) point cloud to disk.

No voxel grid, no NBV planning here - just capture, convert, filter, save,
one view at a time, with each step printed so it's traceable.

Usage: python scan_and_save.py
"""
import os

import numpy as np

from nbv_environment import NBVEnv2
from nbv_core.camera_geometry import capture_rgb_and_depth, compute_intrinsics, backproject_depth, edge_discontinuity_mask
from nbv_core.io_utils import save_depth_image

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures", "scan_and_save")
NUM_VIEWS = 8
ORBIT_HEIGHT_M = 1.2
MIN_DEPTH_M = 0.15  # points closer than this are almost certainly the gripper, not the scene


def save_view(view_index: int, env: NBVEnv2) -> None:
    # Step 1: render the camera, get real depth in millimeters.
    # (capture_rgb_and_depth also returns an RGB image and a per-pixel body-ID
    # mask, unused here since this script is only about the depth -> point
    # cloud pipeline - `_` is the conventional "deliberately discarded" name.)
    _, depth_mm, t_cam_world, q_cam_world, _ = capture_rgb_and_depth(
        env._p, env.robot_id, env.camera_link,
        env.camera.width, env.camera.height, env.camera.fov,
        env.camera.near, env.camera.far,
    )
    print(f"View {view_index}: captured a {depth_mm.shape[1]}x{depth_mm.shape[0]} depth image "
          f"(camera at {np.round(t_cam_world, 2)})")

    # Step 2: turn every pixel of that depth image into a 3D point in world space.
    fx, fy, cx, cy = compute_intrinsics(env.camera.width, env.camera.height, env.camera.fov)
    points_world_grid = backproject_depth(depth_mm, t_cam_world, q_cam_world, fx, fy, cx, cy)
    print(f"View {view_index}: back-projected into a {points_world_grid.shape[0]}x{points_world_grid.shape[1]} "
          f"grid of 3D points (one per pixel)")

    # Step 3: keep only points we trust - see MIN_DEPTH_M and edge_discontinuity_mask above.
    depth_m = depth_mm.astype(np.float64) / 1000.0
    is_in_range = (depth_m > MIN_DEPTH_M) & (depth_m < env.camera.far * 0.98)
    is_not_edge_artifact = edge_discontinuity_mask(depth_m)
    is_trustworthy = is_in_range & is_not_edge_artifact

    points_world = points_world_grid[is_trustworthy]
    print(f"View {view_index}: kept {len(points_world)} / {is_trustworthy.size} points after filtering "
          f"(dropped background/too-close/edge-artifact pixels)")

    # Step 4: save both.
    depth_image_path = os.path.join(OUTPUT_DIR, f"depth_{view_index:02d}.png")
    save_depth_image(depth_mm, depth_image_path, near_m=env.camera.near, far_m=env.camera.far)

    point_cloud_path = os.path.join(OUTPUT_DIR, f"pointcloud_{view_index:02d}.npy")
    np.save(point_cloud_path, points_world)

    print(f"View {view_index}: saved {depth_image_path} and {point_cloud_path}\n")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    env = NBVEnv2(render=False)

    env.move_through_orbit(
        n_views=NUM_VIEWS,
        height=ORBIT_HEIGHT_M,
        on_stop=lambda view_index: save_view(view_index, env),
    )

    print(f"Done. {NUM_VIEWS} views saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
