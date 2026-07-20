"""
Compares the saved mustard-bottle point cloud against the object's actual
ground-truth mesh (collision_vhacd.obj - the same coarse mesh PyBullet
renders and collides with, NOT the finer YCB scan mesh - see note below) to
get a real, ground-truth-backed accuracy number instead of just eyeballing
plots for coherence.

How this is possible: the mesh is a static asset with a known, fixed shape
in its own local frame. To compare it against the point cloud (world-frame),
the mesh just needs to be placed in that same world frame:
  1. t_obj_world/q_obj_world - where the object actually settled at scan
     time, saved by scan_and_save_mustard_only.py (env._p.getBasePositionAndOrientation).
  2. R_mesh_baselink - the mesh's own fixed local offset from the URDF's
     <visual>/<collision> <origin rpy="0 0 1.57"> tag (translation is zero).
Composing these puts every mesh vertex into world coordinates, the same
frame the point cloud is already in. Then for every scanned point we ask
"how far is this from the nearest point on the true surface?" (point-to-mesh
distance, via Open3D's RaycastingScene) - a direct per-point accuracy
number, not just a visual coherence check.

Note on mesh choice: model_textureless.urdf's <visual> AND <collision> both
point at collision_vhacd.obj (a coarse ~90-triangle convex-decomposition
proxy) - that's the literal shape PyBullet rendered to produce our depth
images, so it's the internally-consistent ground truth here, even though
it's coarser than the original YCB scan mesh. Expect a small baseline error
from its own faceting, not just reconstruction noise.

Requires scan_and_save_mustard_only.py to have been run first (reads its
saved pointcloud_mustard_*.npy and object_pose.npz).

Usage:
  python compare_pointcloud_to_mesh.py            # print stats + save a PNG
  python compare_pointcloud_to_mesh.py --view      # also open an interactive
                                                     # Open3D window (mesh
                                                     # wireframe + point cloud
                                                     # colored by error)
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pybullet as pb
import trimesh

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(PROJECT_ROOT, "captures", "scan_and_save_mustard_only")
MESH_PATH = os.path.join(
    PROJECT_ROOT,
    "third_party/shelf_gym_repo/shelf_gym/meshes/urdf/ycb_objects/YcbMustardBottle/collision_vhacd.obj",
)
# From model_textureless.urdf's <visual>/<collision> <origin rpy="0 0 1.57">.
R_MESH_BASELINK = np.array(pb.getMatrixFromQuaternion(pb.getQuaternionFromEuler([0, 0, 1.57]))).reshape(3, 3)

# model_textureless.urdf's <inertial><origin rpy="0 0 0.1" xyz="0.005 0.005 -0.015"/> is
# NOT zero. PyBullet's getBasePositionAndOrientation() returns the pose of the link's
# INERTIAL frame, not the URDF link/visual origin - a well-known pybullet gotcha for any
# body whose <inertial><origin> differs from identity. Composing t_obj_world/q_obj_world
# directly with the visual mesh's local vertices (as if it were the link origin) silently
# mis-places the ground-truth mesh by this inertial offset: 5.73deg rotation (the 0.1 rad)
# + ~17mm translation at this object's scale - confirmed by comparing point-to-mesh error
# with and without this correction (mean error dropped ~30-60% once corrected).
R_LINK_INERTIAL = np.array(pb.getMatrixFromQuaternion(pb.getQuaternionFromEuler([0, 0, 0.1]))).reshape(3, 3)
T_LINK_INERTIAL = np.array([0.005, 0.005, -0.015])


def load_scanned_points() -> np.ndarray:
    """
    Loads whichever candidate point cloud was produced most recently, not a
    fixed fused > combined > per-view preference - a stale fused_voxel_
    pointcloud.npy from an earlier scan_and_fuse_mustard_voxel.py run would
    otherwise silently outrank a fresh combined_mustard_pointcloud.npy from
    the current run, comparing last run's points against this run's mesh
    placement (two different physical object poses).
    """
    fused_path = os.path.join(INPUT_DIR, "fused_voxel_pointcloud.npy")
    combined_path = os.path.join(INPUT_DIR, "combined_mustard_pointcloud.npy")
    candidates = [p for p in (fused_path, combined_path) if os.path.exists(p)]
    if candidates:
        newest_path = max(candidates, key=os.path.getmtime)
        return np.load(newest_path)
    paths = sorted(glob.glob(os.path.join(INPUT_DIR, "pointcloud_mustard_*.npy")))
    if not paths:
        raise FileNotFoundError(
            f"No point cloud files found in {INPUT_DIR} - run scan_and_save_mustard_only.py first."
        )
    return np.concatenate([np.load(p) for p in paths], axis=0)


def load_ground_truth_mesh_world() -> o3d.geometry.TriangleMesh:
    pose_path = os.path.join(INPUT_DIR, "object_pose.npz")
    if not os.path.exists(pose_path):
        raise FileNotFoundError(
            f"{pose_path} not found - re-run scan_and_save_mustard_only.py "
            "(it now saves the object's ground-truth pose alongside the point clouds)."
        )
    pose = np.load(pose_path)
    t_obj_world = pose["t_obj_world"]
    q_obj_world = pose["q_obj_world"]
    R_inertial_world = np.array(pb.getMatrixFromQuaternion(q_obj_world)).reshape(3, 3)

    # Undo the inertial-frame offset to recover the true link-origin pose (see
    # R_LINK_INERTIAL/T_LINK_INERTIAL above) - worldTlink = worldTinertial * inertialTlink.
    R_baselink_world = R_inertial_world @ R_LINK_INERTIAL.T
    t_obj_world = R_inertial_world @ (-R_LINK_INERTIAL.T @ T_LINK_INERTIAL) + t_obj_world

    # collision_vhacd.obj is a VHACD convex decomposition: 4 separate hull
    # pieces, 3 of which are non-triangular faces (1 quad + 2 pentagons).
    # Open3D's read_triangle_mesh silently DROPS non-triangle faces instead
    # of triangulating them ("Skipping non-triangle primitive geometry"),
    # leaving real holes in the ground truth and producing a spurious
    # systematic offset in the distance comparison. trimesh triangulates
    # n-gons instead of dropping them (98 faces recovered vs Open3D's 90).
    tm = trimesh.load(MESH_PATH, process=False)
    if isinstance(tm, trimesh.Scene):
        tm = trimesh.util.concatenate(list(tm.geometry.values()))
    v_mesh = np.asarray(tm.vertices)

    v_world = (R_baselink_world @ (R_MESH_BASELINK @ v_mesh.T)).T + t_obj_world
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(v_world)
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(tm.faces))
    mesh.compute_vertex_normals()
    return mesh


def point_to_mesh_distances(points_world: np.ndarray, mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    scene = o3d.t.geometry.RaycastingScene()
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene.add_triangles(mesh_t)
    query = o3d.core.Tensor(points_world.astype(np.float32))
    return scene.compute_distance(query).numpy()


def print_stats(distances_m: np.ndarray) -> None:
    d_mm = distances_m * 1000.0
    print(f"Point-to-mesh distance over {len(d_mm)} points:")
    print(f"  mean   = {d_mm.mean():.2f} mm")
    print(f"  median = {np.median(d_mm):.2f} mm")
    print(f"  rms    = {np.sqrt((d_mm ** 2).mean()):.2f} mm")
    print(f"  max    = {d_mm.max():.2f} mm")
    for threshold_mm in (2.0, 5.0, 10.0):
        frac = (d_mm < threshold_mm).mean() * 100
        print(f"  within {threshold_mm:>4.1f}mm: {frac:5.1f}%")


def save_error_figure(points_world: np.ndarray, distances_m: np.ndarray) -> None:
    d_mm = distances_m * 1000.0
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    vmax = np.percentile(d_mm, 99)  # clip outliers so a few bad points don't wash out the colorscale
    scatter = ax.scatter(
        points_world[:, 0], points_world[:, 1], points_world[:, 2],
        s=1, c=d_mm, cmap='viridis', vmin=0, vmax=vmax,
    )
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(f"Point-to-mesh distance vs ground truth ({len(points_world)} pts, "
                 f"mean={d_mm.mean():.1f}mm)")
    fig.colorbar(scatter, ax=ax, label="distance to true surface (mm)", shrink=0.6)
    path = os.path.join(INPUT_DIR, "pointcloud_vs_mesh_error.png")
    fig.savefig(path, dpi=150)
    print(f"Saved error visualization to {path}")


def view_in_open3d(points_world: np.ndarray, distances_m: np.ndarray, mesh: o3d.geometry.TriangleMesh) -> None:
    d_mm = distances_m * 1000.0
    vmax = np.percentile(d_mm, 99)
    colors = plt.get_cmap('viridis')(np.clip(d_mm / vmax, 0, 1))[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_world)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    mesh_wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    mesh_wireframe.paint_uniform_color([0.2, 0.2, 0.2])

    print("Opening Open3D viewer (gray wireframe = ground-truth mesh, colored "
          "points = scan colored by error, drag to rotate, close window to continue)...")
    o3d.visualization.draw_geometries([mesh_wireframe, pcd])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", action="store_true", help="Open an interactive Open3D window.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    points_world = load_scanned_points()
    mesh = load_ground_truth_mesh_world()
    distances_m = point_to_mesh_distances(points_world, mesh)

    print_stats(distances_m)
    save_error_figure(points_world, distances_m)

    if args.view:
        view_in_open3d(points_world, distances_m, mesh)


if __name__ == "__main__":
    main()
