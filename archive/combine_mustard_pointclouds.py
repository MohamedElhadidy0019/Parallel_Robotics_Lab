"""
Combines the per-view mustard-only point clouds saved by
scan_and_save_mustard_only.py into a single point cloud, and saves a
visualization so you can check the combined shape looks like a complete
bottle - it should trace much more of the surface than any single view
alone, since each view only sees the side facing the camera.

This is the simplest possible way to combine point clouds: just concatenate
them, no deduplication of overlapping points between views. If the result
looks too noisy/redundant, voxel-grid deduplication (nbv_core.voxel_grid,
already used elsewhere in the project) is the natural next step - this
script exists first to check whether that's even needed.

Requires scan_and_save_mustard_only.py to have been run first (this script
only reads its saved pointcloud_mustard_XX.npy files, it doesn't run the sim).

Usage:
  python combine_mustard_pointclouds.py            # save .npy + a PNG snapshot
  python combine_mustard_pointclouds.py --view      # also open an interactive
                                                     # Open3D window (needs a
                                                     # display - won't work
                                                     # over a headless/SSH
                                                     # session without X11)
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures", "scan_and_save_mustard_only")
OUTPUT_DIR = INPUT_DIR  # save alongside the per-view files


def load_all_view_point_clouds() -> list[np.ndarray]:
    paths = sorted(glob.glob(os.path.join(INPUT_DIR, "pointcloud_mustard_*.npy")))
    if not paths:
        raise FileNotFoundError(
            f"No pointcloud_mustard_*.npy files found in {INPUT_DIR} - "
            "run scan_and_save_mustard_only.py first."
        )
    return [np.load(path) for path in paths]


def view_in_open3d(points: np.ndarray, view_index_per_point: np.ndarray) -> None:
    """
    Open an interactive Open3D window for the combined point cloud, colored
    by which view each point came from (same categorical coloring as
    save_combined_by_view_figure below) - purely a visualization tool (mouse
    to rotate/zoom, close the window to continue), not part of the actual
    reconstruction pipeline, so it's fine to reach for a library here.
    """
    import open3d as o3d

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    colors = plt.get_cmap('tab10')(view_index_per_point % 10)[:, :3]
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    print("Opening Open3D viewer (drag to rotate, scroll to zoom, close the window to continue)...")
    o3d.visualization.draw_geometries([point_cloud])


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
    Colored by which of the 8 views each point came from (tab10: up to 10
    distinct colors). If the views actually agree with each other, colors
    should be thoroughly interspersed across one coherent shape. If instead
    you can see 8 separately-colored, offset/rotated copies of a similar
    shape, the views are NOT lining up correctly in world coordinates.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--view", action="store_true",
        help="Open an interactive Open3D window showing the combined point cloud.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    per_view_points = load_all_view_point_clouds()
    for i, points in enumerate(per_view_points):
        print(f"View {i}: {len(points)} points")

    combined_points = np.concatenate(per_view_points, axis=0)
    # Same length as combined_points - which view each point came from, so we
    # can tell apart "8 views blending into one shape" from "8 views that
    # don't actually agree with each other" when we look at the plot.
    view_index_per_point = np.concatenate([
        np.full(len(points), i) for i, points in enumerate(per_view_points)
    ])
    print(f"\nCombined: {len(combined_points)} points from {len(per_view_points)} views "
          f"(no deduplication yet - overlapping views repeat points)")

    combined_path = os.path.join(OUTPUT_DIR, "combined_mustard_pointcloud.npy")
    np.save(combined_path, combined_points)
    print(f"Saved combined point cloud to {combined_path}")

    save_combined_by_height_figure(combined_points)
    save_combined_by_view_figure(combined_points, view_index_per_point)

    if args.view:
        view_in_open3d(combined_points, view_index_per_point)


if __name__ == "__main__":
    main()
