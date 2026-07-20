# python3 -c "
# import open3d as o3d
# from compare_pointcloud_to_mesh import load_ground_truth_mesh_world
# mesh = load_ground_truth_mesh_world()
# mesh.paint_uniform_color([0.7, 0.7, 0.7])
# o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True)
# "

# python3 -c "
# import open3d as o3d
# pcd = o3d.io.read_point_cloud('captures/scan_and_save_mustard_only/combined_mustard_pointcloud.ply')
# o3d.visualization.draw_geometries([pcd])
# "

python3 -c "
import glob, numpy as np, open3d as o3d
paths = sorted(glob.glob('captures/scan_and_save_mustard_only/pointcloud_mustard_*.npy'))
for p in paths:
    pts = np.load(p)
    print(f'{p}: {len(pts)} points (close window to see next)')
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    o3d.visualization.draw_geometries([pcd], window_name=p)
"
