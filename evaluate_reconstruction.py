"""
Automated, quantitative reconstruction-quality check against the known-CAD ground truth mesh -
replaces "does this screenshot look about right" with real numbers, split by height band so
regional defects (e.g. the bottle's top/bottom vs. its middle) show up as numbers, not just a
visual impression.

Two complementary directions, both needed:
  - ACCURACY (point-to-mesh): for every scanned point, distance to the nearest true surface
    point. Catches noise/outliers - points that ended up somewhere they shouldn't have.
  - COMPLETENESS (mesh-to-point): for every sampled true-surface point, distance to the nearest
    scanned point. Catches holes/gaps - real surface area the scan never captured, which is
    what produces the sparse, faceted "multiple flat sides" look in poorly-covered regions
    (few real samples there get triangulated/rendered as a rough polygon instead of a smooth
    surface, not a mesh-quality problem, a coverage-density one).

Both are computed overall AND per height band (N_HEIGHT_BANDS equal-height slices from the
object's min to max world z) - this is what actually answers "is the bottom/top worse than the
middle," not just eyeballing a screenshot from one angle.

Requires full_pipeline.py to have been run first (reads its saved combined_pointcloud.npy and
object_pose.npz).

Usage:
  python evaluate_reconstruction.py            # print stats
  python evaluate_reconstruction.py --view     # also open an interactive Open3D window
"""
import argparse
import os

import numpy as np
import open3d as o3d

from nbv_core.coverage import load_object_mesh_world, sample_surface_points_and_normals

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(PROJECT_ROOT, "captures", "full_pipeline")
N_HEIGHT_BANDS = 5
N_MESH_SAMPLES = 20000
ACCURACY_THRESHOLDS_MM = (2.0, 5.0, 10.0)
COMPLETENESS_THRESHOLDS_MM = (2.0, 5.0, 10.0)


def load_scanned_points_and_pose() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cloud_path = os.path.join(INPUT_DIR, "combined_pointcloud.npy")
    pose_path = os.path.join(INPUT_DIR, "object_pose.npz")
    if not os.path.exists(cloud_path) or not os.path.exists(pose_path):
        raise FileNotFoundError(
            f"{cloud_path} / {pose_path} not found - run full_pipeline.py first."
        )
    points_world = np.load(cloud_path)
    pose = np.load(pose_path)
    return points_world, pose["t_obj_world"], pose["q_obj_world"]


def point_to_mesh_distances(points_world: np.ndarray, mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    query = o3d.core.Tensor(points_world.astype(np.float32))
    return scene.compute_distance(query).numpy()


def nearest_neighbor_distances(query_points: np.ndarray, reference_points: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree
    distances, _ = cKDTree(reference_points).query(query_points)
    return distances


def print_distance_stats(label: str, distances_m: np.ndarray, thresholds_mm: tuple[float, ...]) -> None:
    d_mm = distances_m * 1000.0
    if len(d_mm) == 0:
        print(f"  {label}: no points in this band")
        return
    within = "  ".join(f"<{t:.0f}mm={100 * (d_mm < t).mean():5.1f}%" for t in thresholds_mm)
    print(f"  {label}: n={len(d_mm):6d}  mean={d_mm.mean():6.2f}mm  median={np.median(d_mm):6.2f}mm  "
          f"rms={np.sqrt((d_mm ** 2).mean()):6.2f}mm  max={d_mm.max():7.2f}mm  {within}")


def evaluate_by_height_band(
    points_world: np.ndarray, accuracy_dist_m: np.ndarray,
    mesh_samples_world: np.ndarray, completeness_dist_m: np.ndarray,
    z_min: float, z_max: float,
) -> None:
    edges = np.linspace(z_min, z_max, N_HEIGHT_BANDS + 1)
    print(f"\nBy height band ({N_HEIGHT_BANDS} equal-height slices, world z {z_min:.3f} to {z_max:.3f}):")
    for i in range(N_HEIGHT_BANDS):
        lo, hi = edges[i], edges[i + 1]
        band_label = f"z in [{lo:.3f},{hi:.3f})"
        in_band_points = (points_world[:, 2] >= lo) & (points_world[:, 2] < hi)
        in_band_samples = (mesh_samples_world[:, 2] >= lo) & (mesh_samples_world[:, 2] < hi)
        print_distance_stats(f"  accuracy     {band_label}", accuracy_dist_m[in_band_points], ACCURACY_THRESHOLDS_MM)
        print_distance_stats(f"  completeness {band_label}", completeness_dist_m[in_band_samples], COMPLETENESS_THRESHOLDS_MM)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", action="store_true", help="Open an interactive Open3D window.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    points_world, t_obj_world, q_obj_world = load_scanned_points_and_pose()
    mesh_world = load_object_mesh_world(t_obj_world, q_obj_world)
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh_world.vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh_world.faces)
    mesh_o3d.compute_vertex_normals()

    mesh_samples_world, _ = sample_surface_points_and_normals(mesh_world, n_samples=N_MESH_SAMPLES)

    accuracy_dist_m = point_to_mesh_distances(points_world, mesh_o3d)
    completeness_dist_m = nearest_neighbor_distances(mesh_samples_world, points_world)

    print(f"Scanned points: {len(points_world)}   True-surface samples: {len(mesh_samples_world)}")
    print("\nOverall:")
    print_distance_stats("accuracy     (scan -> true surface)", accuracy_dist_m, ACCURACY_THRESHOLDS_MM)
    print_distance_stats("completeness (true surface -> scan)", completeness_dist_m, COMPLETENESS_THRESHOLDS_MM)

    z_min, z_max = mesh_world.vertices[:, 2].min(), mesh_world.vertices[:, 2].max()
    evaluate_by_height_band(points_world, accuracy_dist_m, mesh_samples_world, completeness_dist_m, z_min, z_max)

    if args.view:
        d_mm = accuracy_dist_m * 1000.0
        vmax = np.percentile(d_mm, 99)
        import matplotlib.pyplot as plt
        colors = plt.get_cmap('viridis')(np.clip(d_mm / vmax, 0, 1))[:, :3]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_world)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Solid, semi-transparent true-surface mesh (not just a wireframe) so the scan points
        # are visible sitting inside/outside/through it, not just next to it - lets you see
        # directly whether points are systematically offset in one direction, not just that
        # they're offset. Needs the newer draw()/MaterialRecord API (draw_geometries doesn't
        # support mesh transparency) - defaultLitTransparency shader + alpha in base_color.
        mesh_material = o3d.visualization.rendering.MaterialRecord()
        mesh_material.shader = "defaultLitTransparency"
        mesh_material.base_color = [0.15, 0.35, 0.95, 0.35]  # saturated blue, alpha=0.35 - flat gray was too
                                                              # washed out under lighting to read surface
                                                              # curvature/shading through

        point_material = o3d.visualization.rendering.MaterialRecord()
        point_material.shader = "defaultUnlit"
        point_material.point_size = 4.0

        print("\nOpening Open3D viewer (translucent surface = ground truth, colored points = scan "
              "colored by accuracy error - viridis, dark=accurate/bright=off - drag to rotate, "
              "close window to continue)...")
        o3d.visualization.draw([
            {"name": "ground_truth_mesh", "geometry": mesh_o3d, "material": mesh_material},
            {"name": "scanned_points", "geometry": pcd, "material": point_material},
        ], title="Reconstruction vs. ground truth", bg_color=(1.0, 1.0, 1.0, 1.0), show_ui=True)


if __name__ == "__main__":
    main()
