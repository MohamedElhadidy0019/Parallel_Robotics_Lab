#!/usr/bin/env bash
# Runs the scan -> combine -> PLY export pipeline - now all one script:
#   scan_and_save_mustard_only.py - orbit scan, per-view point clouds,
#   combines them, and saves both combined_mustard_pointcloud.npy and .ply.
#
# Output: captures/scan_and_save_mustard_only/combined_mustard_pointcloud.ply
#
# Usage:
#   ./run_scan_and_export_ply.sh          # headless
#   ./run_scan_and_export_ply.sh --gui     # watch the robot move
#   ./run_scan_and_export_ply.sh --view    # also open an Open3D window at the end
set -e

cd "$(dirname "${BASH_SOURCE[0]}")"

conda run -n rob_env python scan_and_save_mustard_only.py "$@"
