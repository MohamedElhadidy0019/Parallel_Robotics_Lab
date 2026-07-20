"""
Same fixed 8-view orbit scan as scan_and_save.py, but this time keeps ONLY
pixels/points that belong to the mustard bottle (env.obj_id) - table, robot,
floor, and background are all dropped. Uses PyBullet's own per-pixel
segmentation mask (nbv_core.camera_geometry.capture_rgb_and_depth's
`body_ids` output) to know which pixel is which object, rather than the
depth-based heuristics (near/far range, edge discontinuity) used elsewhere.

Worth knowing: this is a sim-only shortcut. A real depth camera has no
ground-truth "this pixel is the mustard bottle" signal - PyBullet hands it to
us for free because it already knows exactly which object it rendered at
each pixel. Fine to use here for isolating what we want to look at, just not
something a real robot's perception pipeline could do this way.

After scanning, combines all per-view clouds into one (previously a separate
script, combine_mustard_pointclouds.py - folded in here so the whole
scan -> combine -> export flow is one command) and exports it as a .ply
(previously save_combined_pointcloud_ply.py). Both retired scripts moved to
archive/.

Usage:
  python scan_and_save_mustard_only.py            # headless scan, default 640x480 camera
  python scan_and_save_mustard_only.py --gui       # show the PyBullet GUI (watch the robot move)
  python scan_and_save_mustard_only.py --view      # open an Open3D window with the combined cloud at the end
  python scan_and_save_mustard_only.py --width 3200 --height 2400
                                                    # much denser capture (~5x default resolution per
                                                    # axis) - shrinks the per-pixel footprint at the
                                                    # working distance from ~1.0mm down to ~0.2mm, so
                                                    # cross-view point-to-point comparisons at a 1mm
                                                    # threshold stop being limited by sampling density
                                                    # itself. Slower to render. Saved to a
                                                    # resolution-suffixed output dir so it doesn't
                                                    # clobber a default-resolution scan.
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from nbv_environment import NBVEnv2
from nbv_core.camera_geometry import capture_rgb_and_depth, compute_intrinsics, backproject_depth, edge_discontinuity_mask
from nbv_core.io_utils import save_depth_image
from shelf_gym.utils.camera_utils import Camera

CAPTURES_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")
NUM_VIEWS = 20
ORBIT_HEIGHT_M = 1.2
DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480

# Pixels tagged as the mustard bottle can still occasionally sit on a bad,
# interpolated depth value right at the object's own silhouette edge (see
# edge_discontinuity_mask) - keep that filter too, on top of the mask, so a
# correct-object-label doesn't let a bad depth value slip through.
EDGE_DISCONTINUITY_THRESHOLD_M = 0.02

# Set by main() before anything else runs - module-level so save_view/combine_and_export
# (called via move_through_orbit's on_stop callback) don't need it threaded through.
OUTPUT_DIR = None


def save_view(view_index: int, env: NBVEnv2) -> None:
    # Step 1: render the camera - depth in millimeters, plus a per-pixel body-ID mask.
    _, depth_mm, t_cam_world, q_cam_world, body_ids = capture_rgb_and_depth(
        env._p, env.robot_id, env.camera_link,
        env.camera.width, env.camera.height, env.camera.fov,
        env.camera.near, env.camera.far,
    )

    # Step 2: which pixels are the mustard bottle, and only the mustard bottle?
    depth_m = depth_mm.astype(np.float64) / 1000.0
    is_mustard = body_ids == env.obj_id
    is_not_edge_artifact = edge_discontinuity_mask(depth_m, threshold_m=EDGE_DISCONTINUITY_THRESHOLD_M)
    is_mustard_and_trustworthy = is_mustard & is_not_edge_artifact
    print(f"View {view_index}: {is_mustard.sum()} / {is_mustard.size} pixels are the mustard bottle "
          f"({is_mustard_and_trustworthy.sum()} kept after also dropping edge artifacts)")

    # Step 3: back-project every pixel, then keep only the mustard ones.
    fx, fy, cx, cy = compute_intrinsics(env.camera.width, env.camera.height, env.camera.fov)
    points_world_grid = backproject_depth(depth_mm, t_cam_world, q_cam_world, fx, fy, cx, cy)
    points_world = points_world_grid[is_mustard_and_trustworthy]

    # Step 4: save both - blank out every non-mustard pixel in the depth
    # image too (set to the far plane, so it renders as a uniform background
    # color and only the mustard bottle's shape stands out).
    depth_mm_mustard_only = np.where(is_mustard_and_trustworthy, depth_mm, env.camera.far * 1000.0)

    depth_image_path = os.path.join(OUTPUT_DIR, f"depth_mustard_{view_index:02d}.png")
    save_depth_image(depth_mm_mustard_only, depth_image_path, near_m=env.camera.near, far_m=env.camera.far)

    point_cloud_path = os.path.join(OUTPUT_DIR, f"pointcloud_mustard_{view_index:02d}.npy")
    np.save(point_cloud_path, points_world)

    print(f"View {view_index}: saved {depth_image_path} and {point_cloud_path} "
          f"({len(points_world)} mustard-only points)\n")


def save_combined_by_height_figure(points: np.ndarray) -> None:
    """The original view: colored by height (z). Good for judging overall shape."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=points[:, 2], cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(f"Combined mustard-only point cloud, by height ({len(points)} pts)")
    path = os.path.join(OUTPUT_DIR, "combined_mustard_pointcloud.png")
    fig.savefig(path, dpi=150)
    print(f"Saved height-colored visualization to {path}")


def save_combined_by_view_figure(points: np.ndarray, view_index_per_point: np.ndarray) -> None:
    """
    Colored by which view each point came from. If the views actually agree
    with each other, colors should be thoroughly interspersed across one
    coherent shape. If instead you can see separately-colored, offset/rotated
    copies of a similar shape, the views are NOT lining up correctly.
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        s=1, c=view_index_per_point, cmap='tab10', vmin=0, vmax=9,
    )
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(f"Combined mustard-only point cloud, by view index ({len(points)} pts)")
    fig.colorbar(scatter, ax=ax, label="view index", shrink=0.6)
    path = os.path.join(OUTPUT_DIR, "combined_mustard_pointcloud_by_view.png")
    fig.savefig(path, dpi=150)
    print(f"Saved per-view-colored visualization to {path}")


def combine_and_export() -> np.ndarray:
    """
    Concatenates all per-view clouds just saved (no deduplication - overlapping
    views repeat points), saves the combined .npy + .ply + the two diagnostic
    PNGs. Returns the combined points (for the optional --view step).
    """
    paths = sorted(glob.glob(os.path.join(OUTPUT_DIR, "pointcloud_mustard_*.npy")))
    per_view_points = [np.load(p) for p in paths]
    for i, points in enumerate(per_view_points):
        print(f"View {i}: {len(points)} points")

    combined_points = np.concatenate(per_view_points, axis=0)
    view_index_per_point = np.concatenate([
        np.full(len(points), i) for i, points in enumerate(per_view_points)
    ])
    print(f"\nCombined: {len(combined_points)} points from {len(per_view_points)} views "
          f"(no deduplication yet - overlapping views repeat points)")

    npy_path = os.path.join(OUTPUT_DIR, "combined_mustard_pointcloud.npy")
    np.save(npy_path, combined_points)
    print(f"Saved combined point cloud to {npy_path}")

    save_combined_by_height_figure(combined_points)
    save_combined_by_view_figure(combined_points, view_index_per_point)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(combined_points)
    ply_path = os.path.join(OUTPUT_DIR, "combined_mustard_pointcloud.ply")
    o3d.io.write_point_cloud(ply_path, pcd)
    print(f"Saved combined point cloud to {ply_path}")

    return combined_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gui", action="store_true", help="Show the PyBullet GUI (watch the robot move).")
    parser.add_argument("--view", action="store_true", help="Open an Open3D window with the combined cloud at the end.")
    parser.add_argument("--width", type=int, default=DEFAULT_CAMERA_WIDTH, help="Camera image width in pixels.")
    parser.add_argument("--height", type=int, default=DEFAULT_CAMERA_HEIGHT, help="Camera image height in pixels.")
    return parser.parse_args()


def main() -> None:
    global OUTPUT_DIR
    args = parse_args()

    is_default_resolution = (args.width == DEFAULT_CAMERA_WIDTH and args.height == DEFAULT_CAMERA_HEIGHT)
    dir_name = "scan_and_save_mustard_only" if is_default_resolution else f"scan_and_save_mustard_only_{args.width}x{args.height}"
    OUTPUT_DIR = os.path.join(CAPTURES_ROOT, dir_name)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    env = NBVEnv2(render=args.gui)
    env.camera = Camera(width=args.width, height=args.height)
    print(f"Camera resolution: {args.width}x{args.height} -> output dir: {OUTPUT_DIR}")

    env.move_through_orbit(
        n_views=NUM_VIEWS,
        height=ORBIT_HEIGHT_M,
        on_stop=lambda view_index: save_view(view_index, env),
    )

    # Ground-truth pose of the mustard bottle at scan time - lets
    # compare_pointcloud_to_mesh.py place the ground-truth mesh in the same
    # world frame as the saved point clouds without needing a live sim.
    t_obj_world, q_obj_world = env._p.getBasePositionAndOrientation(env.obj_id)
    np.savez(
        os.path.join(OUTPUT_DIR, "object_pose.npz"),
        t_obj_world=np.array(t_obj_world),
        q_obj_world=np.array(q_obj_world),
    )

    print(f"Done. {NUM_VIEWS} views saved to {OUTPUT_DIR}\n")

    combined_points = combine_and_export()

    if args.view:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(combined_points)
        print("Opening Open3D viewer (drag to rotate, scroll to zoom, close the window to continue)...")
        o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()
