"""
Incrementally ICP-registers the mustard-only per-view point clouds one view
at a time: view 0 is the reference frame, view 1 is ICP-registered against
it and folded in, view 2 is registered against the (view0+view1)
accumulated cloud, and so on. After EVERY step this opens an Open3D window
showing the accumulated cloud so far (colored by view index) and prints, both
before and after ICP: what % of the new view's points land within 1/2/3/5mm
of the existing accumulated cloud (the "do the points actually lie on top of
each other" question - target is 2mm), plus point-to-mesh distance stats
against the ground-truth mesh. Close each window to advance to the next view.

Uses full 6-DOF point-to-point ICP by default. A prior investigation
(archive/combine_mustard_pointclouds_icp.py) found full ICP catastrophically
diverges on this bottle's smooth, near-cylindrical surface (mean point-to-mesh
error went 6mm -> 212mm) because rotation is poorly constrained by a
self-similar curved surface (the aperture problem) - a view "slides" around
the curve to a numerically tighter but geometrically wrong fit. That's
exactly the failure mode this script lets you watch happen, view by view.
Pass --translation-only to use the safer variant instead (restricts every
ICP step to a pure translation, sidestepping the aperture problem entirely).

Usage:
  python icp_incremental_view.py                       # full 6-DOF point-to-point ICP
  python icp_incremental_view.py --translation-only     # matches archive/ script's safer approach
  python icp_incremental_view.py --max-corr-dist 0.01   # tighter/looser correspondence search radius
"""
import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

import compare_pointcloud_to_mesh as cptm

CAPTURES_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")
DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480
DEFAULT_MAX_CORRESPONDENCE_DISTANCE_M = 0.015
OVERLAP_THRESHOLDS_MM = (1.0, 2.0, 3.0, 5.0)


def captures_dir_for_resolution(width: int, height: int) -> str:
    is_default = (width == DEFAULT_CAMERA_WIDTH and height == DEFAULT_CAMERA_HEIGHT)
    dir_name = "scan_and_save_mustard_only" if is_default else f"scan_and_save_mustard_only_{width}x{height}"
    return os.path.join(CAPTURES_ROOT, dir_name)


def load_per_view_points(captures_dir: str) -> list[np.ndarray]:
    paths = sorted(glob.glob(os.path.join(captures_dir, "pointcloud_mustard_*.npy")))
    return [np.load(p) for p in paths]


def overlap_stats(source_points: np.ndarray, target_points: np.ndarray) -> str:
    """% of source_points with a nearest neighbor in target_points within each threshold - this is
    the "do the points actually lie on top of each other" question, independent of the ground-truth mesh."""
    tree = cKDTree(target_points)
    d_m, _ = tree.query(source_points, k=1)
    parts = [f"<={t:.0f}mm={100.0 * np.mean(d_m <= t / 1000.0):.1f}%" for t in OVERLAP_THRESHOLDS_MM]
    return " ".join(parts)


def mesh_distance_stats(points_world: np.ndarray, mesh: o3d.geometry.TriangleMesh) -> str:
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    query = o3d.core.Tensor(points_world.astype(np.float32))
    d_mm = scene.compute_distance(query).numpy() * 1000.0
    return f"mean={d_mm.mean():.2f}mm median={np.median(d_mm):.2f}mm rms={np.sqrt((d_mm ** 2).mean()):.2f}mm"


def translation_only_icp(
    source_points: np.ndarray, target_pcd: o3d.geometry.PointCloud,
    max_corr_dist: float, max_iters: int = 30, tol: float = 1e-5,
) -> np.ndarray:
    rng = np.random.default_rng(0)
    tree = o3d.geometry.KDTreeFlann(target_pcd)
    target_points = np.asarray(target_pcd.points)
    current_points = source_points.copy()
    for _ in range(max_iters):
        sample_idx = rng.choice(len(current_points), size=min(2000, len(current_points)), replace=False)
        deltas = []
        for pt in current_points[sample_idx]:
            _, nn_idx, dist2 = tree.search_knn_vector_3d(pt, 1)
            if dist2[0] < max_corr_dist ** 2:
                deltas.append(target_points[nn_idx[0]] - pt)
        if len(deltas) < 10:
            break
        step = np.mean(deltas, axis=0)
        current_points = current_points + step
        if np.linalg.norm(step) < tol:
            break
    return current_points


def full_icp(source_points: np.ndarray, target_pcd: o3d.geometry.PointCloud, max_corr_dist: float) -> np.ndarray:
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_points)
    result = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, max_corr_dist, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    R_icp = result.transformation[:3, :3]
    rotation_deg = np.degrees(np.arccos(np.clip((np.trace(R_icp) - 1) / 2, -1, 1)))
    print(f"    ICP fitness={result.fitness:.3f} inlier_rmse={result.inlier_rmse * 1000:.2f}mm "
          f"rotation_applied={rotation_deg:.2f}deg")
    source_pcd.transform(result.transformation)
    return np.asarray(source_pcd.points)


def view_step(points: np.ndarray, view_index_per_point: np.ndarray, up_to_view: int) -> None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = plt.get_cmap('tab10')(view_index_per_point % 10)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    print(f"  Opening viewer for views 0..{up_to_view} (close window to continue)...")
    o3d.visualization.draw_geometries([pcd], window_name=f"Accumulated through view {up_to_view}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--translation-only", action="store_true",
                         help="Use translation-only ICP (safer, matches archive/ script).")
    parser.add_argument("--max-corr-dist", type=float, default=DEFAULT_MAX_CORRESPONDENCE_DISTANCE_M,
                         help="Max correspondence distance in meters (default 0.015).")
    parser.add_argument("--width", type=int, default=DEFAULT_CAMERA_WIDTH,
                         help="Camera width the scan was captured at (selects the matching captures dir).")
    parser.add_argument("--height", type=int, default=DEFAULT_CAMERA_HEIGHT,
                         help="Camera height the scan was captured at (selects the matching captures dir).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    captures_dir = captures_dir_for_resolution(args.width, args.height)
    cptm.INPUT_DIR = captures_dir  # so load_ground_truth_mesh_world() reads THIS run's object_pose.npz, not the default-resolution one

    per_view_points = load_per_view_points(captures_dir)
    print(f"Loaded {len(per_view_points)} views from {captures_dir}")
    print(f"Mode: {'translation-only' if args.translation_only else 'full 6-DOF point-to-point'} ICP, "
          f"max_corr_dist={args.max_corr_dist * 1000:.1f}mm\n")

    mesh = cptm.load_ground_truth_mesh_world()

    accumulated_pcd = o3d.geometry.PointCloud()
    accumulated_pcd.points = o3d.utility.Vector3dVector(per_view_points[0])
    combined_points = [per_view_points[0]]
    view_index_per_point = [np.zeros(len(per_view_points[0]), dtype=int)]

    print(f"View 0 (reference, no ICP): {mesh_distance_stats(per_view_points[0], mesh)}")
    view_step(per_view_points[0], view_index_per_point[0], 0)

    for i in range(1, len(per_view_points)):
        target_points_before = np.asarray(accumulated_pcd.points)
        print(f"\nView {i}: registering against accumulated views 0..{i - 1} "
              f"({len(target_points_before)} pts)...")
        print(f"  overlap BEFORE ICP: {overlap_stats(per_view_points[i], target_points_before)}")

        if args.translation_only:
            registered_points = translation_only_icp(per_view_points[i], accumulated_pcd, args.max_corr_dist)
        else:
            registered_points = full_icp(per_view_points[i], accumulated_pcd, args.max_corr_dist)

        print(f"  overlap AFTER ICP:  {overlap_stats(registered_points, target_points_before)}")

        combined_points.append(registered_points)
        view_index_per_point.append(np.full(len(registered_points), i))

        accumulated_pcd.points = o3d.utility.Vector3dVector(
            np.concatenate([target_points_before, registered_points], axis=0)
        )

        all_points_so_far = np.concatenate(combined_points, axis=0)
        print(f"  vs ground-truth mesh: {mesh_distance_stats(all_points_so_far, mesh)}")
        view_step(all_points_so_far, np.concatenate(view_index_per_point), i)

    print("\nDone.")


if __name__ == "__main__":
    main()
