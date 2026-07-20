"""
Saves the combined mustard-bottle point cloud produced by
scan_and_save_mustard_only_shelfgym.py (the shelf_gym-vendor-pipeline scan)
as a standard .ply file, viewable in any generic point cloud tool
(MeshLab, CloudCompare, etc.).

Requires scan_and_save_mustard_only_shelfgym.py to have been run first.

Usage: python save_combined_pointcloud_ply_shelfgym.py
"""
import os

import numpy as np
import open3d as o3d

from scan_and_save_mustard_only_shelfgym import OUTPUT_DIR


def main() -> None:
    npy_path = os.path.join(OUTPUT_DIR, "combined_mustard_pointcloud.npy")
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"{npy_path} not found - run scan_and_save_mustard_only_shelfgym.py first.")
    points = np.load(npy_path)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    ply_path = os.path.join(OUTPUT_DIR, "combined_mustard_pointcloud.ply")
    o3d.io.write_point_cloud(ply_path, pcd)
    print(f"Saved {len(points)} points to {ply_path}")


if __name__ == "__main__":
    main()
