"""
Refines the per-view mustard-only point clouds' mutual alignment with
incremental TRANSLATION-ONLY ICP before combining, instead of plain
concatenation (combine_mustard_pointclouds.py's approach).

Why translation-only, not full 6-DOF ICP: tried full point-to-plane ICP
first and it actively made things much worse (mean mesh error 6mm -> 212mm).
Diagnosed why: the bottle's surface is largely a smooth cylinder, which
barely constrains rotation about/near its own curvature - full ICP "slid" a
view by an 18.89deg rotation to a numerically tighter but geometrically
wrong fit (inlier RMSE dropped from 4.9mm to 0.8mm at the WRONG alignment),
and that error compounded catastrophically across the incremental chain
(single-view corrections growing from ~100mm to over 1600mm by the last
view). Restricting each step to a pure translation sidesteps this aperture
problem entirely (translation is well-constrained even on a smooth
cylinder) and is a better physical match anyway for the leading hypothesis
(depth-buffer-precision bias pushing points along their viewing ray, which
looks like a small translation, not a rotation).

Why do this at all: camera pose is now essentially exact (see
nbv_environment.py's _snap_to_joint_targets - forward-kinematics-verified
sub-0.1mm/0deg residual against the intended viewpoint), yet a small
~0-3.6mm per-view bias against the ground-truth mesh remains (see
project_pointcloud_layering_bug memory - camera pose/IK/settling were ruled
out as the cause via three separate experiments, still unexplained,
suspected depth-buffer precision).

Usage:
  python combine_mustard_pointclouds_icp.py            # save .npy + a PNG,
                                                          # print before/after
                                                          # error vs the
                                                          # ground-truth mesh
  python combine_mustard_pointclouds_icp.py --view      # also open an
                                                          # interactive
                                                          # Open3D window
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from combine_mustard_pointclouds import load_all_view_point_clouds, OUTPUT_DIR
from compare_pointcloud_to_mesh import load_ground_truth_mesh_world

# Misalignment is already down to a few mm (camera pose is essentially
# exact) - a tight correspondence distance keeps ICP from latching onto the
# wrong part of the (self-similar, roughly cylindrical) bottle surface.
MAX_CORRESPONDENCE_DISTANCE_M = 0.015
ACCUMULATED_VOXEL_SIZE_M = 0.002  # keeps the growing registration target from getting unboundedly dense
CORRESPONDENCE_SAMPLE_SIZE = 2000  # per iteration - plenty to estimate a robust mean translation, much faster than using every point
MAX_ITERATIONS = 30
CONVERGENCE_TOL_M = 1e-5


def translation_only_icp(
    source_points: np.ndarray, target_pcd: o3d.geometry.PointCloud
) -> tuple[np.ndarray, np.ndarray]:
    """
    Point-to-point ICP restricted to a pure translation each iteration (no
    rotation estimated, ever) - see module docstring for why. Each
    iteration: sample points, find their nearest neighbor in the target,
    move the WHOLE cloud by the mean of those correspondence vectors,
    repeat until the step size is negligible or the iteration budget runs out.

    Returns: (registered_points, total_translation_m)
    """
    rng = np.random.default_rng(0)
    tree = o3d.geometry.KDTreeFlann(target_pcd)
    target_points = np.asarray(target_pcd.points)

    current_points = source_points.copy()
    total_translation = np.zeros(3)

    for _ in range(MAX_ITERATIONS):
        sample_idx = rng.choice(len(current_points), size=min(CORRESPONDENCE_SAMPLE_SIZE, len(current_points)), replace=False)
        deltas = []
        for pt in current_points[sample_idx]:
            _, idx, dist2 = tree.search_knn_vector_3d(pt, 1)
            if dist2[0] < MAX_CORRESPONDENCE_DISTANCE_M ** 2:
                deltas.append(target_points[idx[0]] - pt)
        if len(deltas) < 10:
            break
        step = np.mean(deltas, axis=0)
        current_points = current_points + step
        total_translation += step
        if np.linalg.norm(step) < CONVERGENCE_TOL_M:
            break

    return current_points, total_translation


def register_incrementally(
    per_view_points: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """
    First view is the reference frame (identity - it's already at an
    essentially-exact camera pose, nothing to correct against yet). Every
    later view is registered via translation-only ICP against the
    accumulated cloud from all prior views, then folded in.

    Returns: (combined_points, view_index_per_point, translation_mm_per_view)
    - translation_mm_per_view is how far ICP moved each view, useful for
    judging whether the correction was actually small (as expected if
    camera pose is the near-exact starting point) or suspiciously large.
    """
    accumulated_pcd = o3d.geometry.PointCloud()
    accumulated_pcd.points = o3d.utility.Vector3dVector(per_view_points[0])
    combined_points = [per_view_points[0]]
    view_index_per_point = [np.zeros(len(per_view_points[0]), dtype=int)]
    translation_mm_per_view = [0.0]

    for i in range(1, len(per_view_points)):
        registered_points, translation_m = translation_only_icp(per_view_points[i], accumulated_pcd)
        translation_mm_per_view.append(np.linalg.norm(translation_m) * 1000)

        combined_points.append(registered_points)
        view_index_per_point.append(np.full(len(registered_points), i))

        accumulated_pcd.points = o3d.utility.Vector3dVector(
            np.concatenate([np.asarray(accumulated_pcd.points), registered_points], axis=0)
        )
        accumulated_pcd = accumulated_pcd.voxel_down_sample(ACCUMULATED_VOXEL_SIZE_M)

    return (
        np.concatenate(combined_points, axis=0),
        np.concatenate(view_index_per_point),
        translation_mm_per_view,
    )


def mesh_distance_stats(points_world: np.ndarray) -> None:
    mesh = load_ground_truth_mesh_world()
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    query = o3d.core.Tensor(points_world.astype(np.float32))
    d_mm = scene.compute_distance(query).numpy() * 1000.0
    print(f"  mean={d_mm.mean():.2f}mm median={np.median(d_mm):.2f}mm "
          f"rms={np.sqrt((d_mm**2).mean()):.2f}mm within5mm={(d_mm<5).mean()*100:.1f}%")


def save_by_view_figure(points: np.ndarray, view_index_per_point: np.ndarray) -> None:
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        s=1, c=view_index_per_point, cmap='tab10', vmin=0, vmax=9,
    )
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.set_title(f"ICP-registered point cloud, by view index ({len(points)} pts)")
    fig.colorbar(scatter, ax=ax, label="view index", shrink=0.6)
    path = os.path.join(OUTPUT_DIR, "combined_mustard_pointcloud_icp_by_view.png")
    fig.savefig(path, dpi=150)
    print(f"Saved per-view-colored visualization to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", action="store_true", help="Open an interactive Open3D window.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    per_view_points = load_all_view_point_clouds()
    print(f"Loaded {len(per_view_points)} views")

    raw_combined = np.concatenate(per_view_points, axis=0)
    print("\nBEFORE ICP (plain concatenation) vs ground-truth mesh:")
    mesh_distance_stats(raw_combined)

    combined_points, view_index_per_point, translation_mm_per_view = register_incrementally(per_view_points)
    print("\nPer-view ICP correction (translation magnitude):")
    for i, t_mm in enumerate(translation_mm_per_view):
        print(f"  view {i}: {t_mm:.2f}mm")

    print("\nAFTER ICP vs ground-truth mesh:")
    mesh_distance_stats(combined_points)

    icp_path = os.path.join(OUTPUT_DIR, "combined_mustard_pointcloud_icp.npy")
    np.save(icp_path, combined_points)
    print(f"\nSaved ICP-registered combined point cloud to {icp_path}")

    save_by_view_figure(combined_points, view_index_per_point)

    if args.view:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(combined_points)
        colors = plt.get_cmap('tab10')(view_index_per_point % 10)[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)
        print("Opening Open3D viewer (colored by view, drag to rotate, close window to continue)...")
        o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()
