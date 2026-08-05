conda run -n rob_env python -c "
import open3d as o3d
pcd = o3d.io.read_point_cloud('/home/mohamed/repos/Parallel_Robotics_Lab/captures/full_pipeline/combined_pointcloud.ply')
o3d.visualization.draw_geometries([pcd])
"