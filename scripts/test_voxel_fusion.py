"""
Milestone 2 check: run the fixed 8-view orbit through capture_frame() +
OccupancyVoxelGrid.integrate_points(), and verify the reconstruction assembles
correctly and progressively across views (no NBV planning yet - motion is the
same proven fixed sweep as orbit_and_capture).

Usage: python scripts/test_voxel_fusion.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nbv_environment import NBVEnv2

SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "captures", "voxel_fusion_test")


def main():
    env = NBVEnv2(render=False)
    n_views = 8
    grid = env.scan_fixed_views_and_integrate(n_views=n_views, height=1.2, save_dir=SAVE_DIR)

    counts = []
    for i in range(n_views):
        pts = np.load(os.path.join(SAVE_DIR, f"occupied_after_view_{i:02d}.npy"))
        counts.append(len(pts))
    print(f"Occupied-voxel count per view: {counts}")

    non_decreasing = all(counts[i] <= counts[i + 1] for i in range(len(counts) - 1))
    print("PASS - occupied count is monotonically non-decreasing across views" if non_decreasing
          else "FAIL - occupied count decreased somewhere (OCCUPIED should be sticky, this shouldn't happen)")

    # Progression snapshot: view 0 (first look), a middle view, and the final fused result
    snapshot_indices = sorted(set([0, n_views // 2, n_views - 1]))
    fig = plt.figure(figsize=(6 * len(snapshot_indices), 6))
    for plot_i, view_i in enumerate(snapshot_indices):
        pts = np.load(os.path.join(SAVE_DIR, f"occupied_after_view_{view_i:02d}.npy"))
        ax = fig.add_subplot(1, len(snapshot_indices), plot_i + 1, projection='3d')
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, c=pts[:, 2], cmap='viridis')
        ax.set_title(f"After view {view_i} ({len(pts)} occupied voxels)")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
    out_path = os.path.join(SAVE_DIR, "fusion_progression.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved fusion progression figure to {out_path}")


if __name__ == "__main__":
    main()
