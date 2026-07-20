"""
Opens an interactive Open3D window showing the scanned point cloud and the
ground-truth mesh together, in the same world frame, so you can directly
see how they differ by eye - rotate/zoom around both at once, rather than
reading distance numbers.

Gray solid (+ wireframe) = ground-truth mesh (collision_vhacd.obj, placed
via the same object_pose.npz + URDF-offset transform as
compare_pointcloud_to_mesh.py - reused from there so both scripts always
agree on where the mesh sits).
Red points = the scanned/combined point cloud.

Requires scan_and_save_mustard_only.py to have been run first (it now
combines and saves the point cloud itself).

Usage:
  python view_pointcloud_vs_mesh.py
"""
import open3d as o3d

from compare_pointcloud_to_mesh import load_scanned_points, load_ground_truth_mesh_world


def main() -> None:
    points_world = load_scanned_points()
    mesh = load_ground_truth_mesh_world()
    mesh.paint_uniform_color([0.75, 0.75, 0.75])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_world)
    pcd.paint_uniform_color([0.9, 0.1, 0.1])

    print("Opening Open3D viewer - gray = ground-truth mesh, red = scanned "
          "point cloud. Drag to rotate, scroll to zoom, close window to exit.")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point cloud (red) vs ground-truth mesh (gray)", width=1000, height=800)
    vis.add_geometry(mesh)
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.point_size = 3.0
    opt.mesh_show_wireframe = True
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
