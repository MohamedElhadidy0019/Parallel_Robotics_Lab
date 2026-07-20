#!/usr/bin/env bash
# Same as run_scan_and_export_ply.sh, but using the shelf_gym-VENDOR capture
# pipeline instead of this project's own (see scan_and_save_mustard_only_shelfgym.py
# for what that swaps out and why):
#   1. scan_and_save_mustard_only_shelfgym.py - orbit scan using shelf_gym's
#      own Camera.get_cam_in_hand()/get_pointcloud(), combines all views,
#      prints point-to-mesh error stats
#   2. save_combined_pointcloud_ply_shelfgym.py - saves that combined cloud
#      as a standard .ply file
#   3. view_pointcloud_vs_mesh_shelfgym.py - opens an Open3D window with the
#      combined cloud (red) over the ground-truth mesh (gray)
#
# Output: captures/scan_and_save_mustard_only_shelfgym/combined_mustard_pointcloud.ply
#
# Usage:
#   ./run_scan_and_export_ply_shelfgym.sh              # scan headless, then view
#   ./run_scan_and_export_ply_shelfgym.sh --gui-scan    # also watch the robot move while scanning
set -e

cd "$(dirname "${BASH_SOURCE[0]}")"

SCAN_ARGS=()
if [[ "$1" == "--gui-scan" ]]; then
    SCAN_ARGS=(--gui)
fi

echo "=== Step 1/3: scanning (shelf_gym vendor pipeline) ==="
conda run -n rob_env python scan_and_save_mustard_only_shelfgym.py "${SCAN_ARGS[@]}"

echo
echo "=== Step 2/3: exporting to .ply ==="
conda run -n rob_env python save_combined_pointcloud_ply_shelfgym.py

echo
echo "=== Step 3/3: viewing cloud vs mesh ==="
conda run -n rob_env python view_pointcloud_vs_mesh_shelfgym.py
