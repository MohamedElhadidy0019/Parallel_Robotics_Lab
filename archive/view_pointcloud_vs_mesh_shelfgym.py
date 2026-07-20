"""
Same interactive viewer as view_pointcloud_vs_mesh.py (gray mesh + red
points), but for the shelf_gym-pipeline scan saved by
scan_and_save_mustard_only_shelfgym.py instead of this project's own
pipeline. Separate script, doesn't touch the existing viewer or its data.

Requires scan_and_save_mustard_only_shelfgym.py to have been run first.

Usage:
  python view_pointcloud_vs_mesh_shelfgym.py
"""
import os

import numpy as np
import open3d as o3d
import pybullet as pb
import trimesh

from compare_pointcloud_to_mesh import MESH_PATH, R_MESH_BASELINK, R_LINK_INERTIAL, T_LINK_INERTIAL
from scan_and_save_mustard_only_shelfgym import OUTPUT_DIR


def load_scanned_points() -> np.ndarray:
    path = os.path.join(OUTPUT_DIR, "combined_mustard_pointcloud.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found - run scan_and_save_mustard_only_shelfgym.py first.")
    return np.load(path)


def load_ground_truth_mesh_world() -> o3d.geometry.TriangleMesh:
    pose = np.load(os.path.join(OUTPUT_DIR, "object_pose.npz"))
    t_obj_world = pose["t_obj_world"]
    q_obj_world = pose["q_obj_world"]
    R_inertial_world = np.array(pb.getMatrixFromQuaternion(q_obj_world)).reshape(3, 3)
    R_baselink_world = R_inertial_world @ R_LINK_INERTIAL.T
    t_baselink_world = R_inertial_world @ (-R_LINK_INERTIAL.T @ T_LINK_INERTIAL) + t_obj_world

    tm = trimesh.load(MESH_PATH, process=False)
    if isinstance(tm, trimesh.Scene):
        tm = trimesh.util.concatenate(list(tm.geometry.values()))
    v_mesh = np.asarray(tm.vertices)
    v_world = (R_baselink_world @ (R_MESH_BASELINK @ v_mesh.T)).T + t_baselink_world

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(v_world)
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(tm.faces))
    mesh.compute_vertex_normals()
    return mesh


def main() -> None:
    points_world = load_scanned_points()
    mesh = load_ground_truth_mesh_world()
    mesh.paint_uniform_color([0.75, 0.75, 0.75])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_world)
    pcd.paint_uniform_color([0.9, 0.1, 0.1])

    print("Opening Open3D viewer (shelf_gym pipeline) - gray = ground-truth mesh, "
          "red = scanned point cloud. Drag to rotate, close window to exit.")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="shelf_gym pipeline: point cloud (red) vs mesh (gray)", width=1000, height=800)
    vis.add_geometry(mesh)
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.point_size = 3.0
    opt.mesh_show_wireframe = True
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
