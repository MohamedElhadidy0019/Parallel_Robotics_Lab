"""
Saves the combined mustard-bottle point cloud (produced by
combine_mustard_pointclouds.py) as a standard .ply file, viewable in any
generic point cloud tool (MeshLab, CloudCompare, etc.), not just this
project's own Open3D scripts.

Requires scan_and_save_mustard_only.py and combine_mustard_pointclouds.py
to have been run first.

Usage: python save_combined_pointcloud_ply.py
"""
import os

import numpy as np
import open3d as o3d

from combine_mustard_pointclouds import INPUT_DIR


def main() -> None:
    npy_path = os.path.join(INPUT_DIR, "combined_mustard_pointcloud.npy")
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"{npy_path} not found - run combine_mustard_pointclouds.py first.")
    points = np.load(npy_path)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    ply_path = os.path.join(INPUT_DIR, "combined_mustard_pointcloud.ply")
    o3d.io.write_point_cloud(ply_path, pcd)
    print(f"Saved {len(points)} points to {ply_path}")


if __name__ == "__main__":
    main()
