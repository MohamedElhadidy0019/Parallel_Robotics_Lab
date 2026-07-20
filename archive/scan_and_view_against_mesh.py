"""
Runs the full pipeline end to end, from a fresh sim to the scanned cloud
sitting over the ground-truth mesh in an interactive viewer:

  1. scan_and_save_mustard_only.main() - drives the arm through the fixed
     orbit, saves mustard-only per-view point clouds + object_pose.npz to
     captures/scan_and_save_mustard_only/.
  2. combine_mustard_pointclouds.main() - concatenates all per-view clouds
     into combined_mustard_pointcloud.npy.
  3. view_pointcloud_vs_mesh.main() - opens an Open3D window with the
     combined cloud (red) over the ground-truth mesh (gray wireframe +
     solid), using compare_pointcloud_to_mesh.py's inertial-frame-corrected
     mesh placement.

Just glues together the three already-working scripts in order - no new
scan/combine/compare logic here.

Usage:
  python scan_and_view_against_mesh.py
"""
import scan_and_save_mustard_only
import combine_mustard_pointclouds
import view_pointcloud_vs_mesh


def main() -> None:
    print("=== Step 1/3: scanning ===")
    scan_and_save_mustard_only.main()

    print("\n=== Step 2/3: combining views ===")
    combine_mustard_pointclouds.main()

    print("\n=== Step 3/3: viewing cloud vs mesh ===")
    view_pointcloud_vs_mesh.main()


if __name__ == "__main__":
    main()
