#!/usr/bin/env bash
# Runs the scan -> view-against-mesh pipeline end to end:
#   1. scan_and_save_mustard_only.py - orbit scan, mustard-only per-view
#      point clouds + object_pose.npz, combines them into
#      combined_mustard_pointcloud.npy/.ply (all in one script now)
#   2. view_pointcloud_vs_mesh.py - opens an Open3D window with the
#      combined cloud (red) over the ground-truth mesh (gray)
#
# Usage: ./run_scan_pipeline.sh
set -e

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "=== Step 1/2: scanning + combining ==="
conda run -n rob_env python scan_and_save_mustard_only.py

echo
echo "=== Step 2/2: viewing cloud vs mesh ==="
conda run -n rob_env python view_pointcloud_vs_mesh.py
