"""NBV Planner: Autonomous Next-Best-View 3D scanning and inspection planner."""

from nbv_planner.api import NBVPlanner
from nbv_planner.camera import (
    CameraIntrinsics,
    backproject_depth,
    transform_points,
)
from nbv_planner.coverage import CoverageTracker
from nbv_planner.motion_planning import (
    TrajectoryPlan,
    move_camera_to,
    move_camera_to_batch,
)
from nbv_planner.ray_scoring import score_candidate_views
from nbv_planner.reachability import ik_filter, sample_candidate_camera_poses
from nbv_planner.viz import NBVVisualizer

__all__ = [
    "CameraIntrinsics",
    "CoverageTracker",
    "NBVPlanner",
    "NBVVisualizer",
    "TrajectoryPlan",
    "backproject_depth",
    "ik_filter",
    "move_camera_to",
    "move_camera_to_batch",
    "sample_candidate_camera_poses",
    "score_candidate_views",
    "transform_points",
]
