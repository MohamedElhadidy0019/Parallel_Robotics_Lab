#!/usr/bin/env bash
# Copy the ROS packages and the planner library into the container workspace.
# Everything else (sim, tests, third_party, captures) stays in this repo.
set -euo pipefail

SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-$HOME/dev/steve_ros2_ws_student/src/parallel_robotics_lab}"

mkdir -p "$TARGET"
rsync -a --delete \
  --exclude '__pycache__' --exclude '*.pyc' --exclude 'csrc/build' --exclude '*.egg-info' \
  "$SOURCE/ros" "$SOURCE/nbv_planner" "$TARGET/"
cp "$SOURCE/pyproject.toml" "$TARGET/pyproject.toml"

echo "Synced to $TARGET ($(du -sh "$TARGET" | cut -f1))"
