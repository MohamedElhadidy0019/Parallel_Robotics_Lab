#!/usr/bin/env bash
# Runs the real NBV pipeline end to end:
#   1. build_reachability_cache.py - only if reachability/reachability_cache.npz
#      is missing (skip if it already exists - it's a one-time precompute,
#      only needs rebuilding if the object placement / robot URDF changes)
#   2. full_pipeline.py - the real dynamic NBV scan
#   3. opens captures/full_pipeline/coverage_visualization.ply (known mesh,
#      green=seen/gray=unseen/black=excluded base) - the useful result to look at
#
# Usage:
#   ./run_full_pipeline.sh                # headless scan, then view coverage mesh
#   ./run_full_pipeline.sh --gui           # watch the robot move
#   ./run_full_pipeline.sh --gui --view    # also pop the raw point cloud (diagnostic only)
set -e

cd "$(dirname "${BASH_SOURCE[0]}")"

CACHE_PATH="reachability/reachability_cache.npz"

if [ -f "$CACHE_PATH" ]; then
    echo "=== Step 1/3: reachability cache already exists at $CACHE_PATH, skipping ==="
else
    echo "=== Step 1/3: building reachability cache (one-time) ==="
    conda run -n rob_env python build_reachability_cache.py
fi

echo
echo "=== Step 2/3: running the NBV scan ==="
conda run -n rob_env python full_pipeline.py "$@"

echo
echo "=== Step 3/3: opening coverage visualization ==="
conda run -n rob_env python -c "
import open3d as o3d
mesh = o3d.io.read_triangle_mesh('captures/full_pipeline/coverage_visualization.ply')
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])
"
