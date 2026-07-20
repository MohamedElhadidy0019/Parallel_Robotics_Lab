import numpy as np
import glob, os
from scan_and_save_mustard_only_shelfgym import OUTPUT_DIR

pose = np.load(os.path.join(OUTPUT_DIR, "object_pose.npz"))
obj_center = pose["t_obj_world"]

paths = sorted(glob.glob(os.path.join(OUTPUT_DIR, "pointcloud_mustard_*.npy")))
print(f"{'view':>4} {'n':>6} {'centroid_xy_angle_deg':>22} {'centroid_dist_from_obj_mm':>26}")
for p in paths:
    idx = int(os.path.basename(p).split('_')[-1].split('.')[0])
    pts = np.load(p)
    if len(pts) == 0:
        continue
    centroid = pts.mean(axis=0)
    rel = centroid - obj_center
    angle_deg = np.degrees(np.arctan2(rel[1], rel[0]))
    dist_mm = np.linalg.norm(rel[:2]) * 1000
    print(f"{idx:>4} {len(pts):>6} {angle_deg:>22.2f} {dist_mm:>26.2f}")

print(f"\nExpected: theta sweeps from 180deg to 360deg across views 0-19 (linspace),")
print(f"so centroid_xy_angle should sweep by a similar ~180deg range if points really")
print(f"track the true camera viewpoint. If it stays roughly constant instead, points")
print(f"from different real viewpoints are landing in the same world region.")
