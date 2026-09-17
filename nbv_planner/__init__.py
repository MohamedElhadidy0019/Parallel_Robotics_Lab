"""NBV Planner: Autonomous Next-Best-View 3D scanning and inspection planner."""

import importlib
from typing import Any

__all__ = [
    "CameraIntrinsics",
    "CoverageTracker",
    "NBVPlanner",
    "NBVVisualizer",
    "TrajectoryPlan",
    "backproject_depth",
    "build_world_config",
    "ik_filter",
    "plan_motion_batch",
    "plan_motion_single",
    "sample_candidate_camera_poses",
    "score_candidate_views",
    "transform_points",
    "warmup_motion_gen",
]

_LAZY_IMPORTS = {
    "CameraIntrinsics": ("nbv_planner.camera", "CameraIntrinsics"),
    "backproject_depth": ("nbv_planner.camera", "backproject_depth"),
    "transform_points": ("nbv_planner.camera", "transform_points"),
    "CoverageTracker": ("nbv_planner.coverage", "CoverageTracker"),
    "NBVPlanner": ("nbv_planner.api", "NBVPlanner"),
    "NBVVisualizer": ("nbv_planner.viz", "NBVVisualizer"),
    "TrajectoryPlan": ("nbv_planner.motion_planning", "TrajectoryPlan"),
    "build_world_config": ("nbv_planner.motion_planning", "build_world_config"),
    "plan_motion_batch": ("nbv_planner.motion_planning", "plan_motion_batch"),
    "plan_motion_single": ("nbv_planner.motion_planning", "plan_motion_single"),
    "warmup_motion_gen": ("nbv_planner.motion_planning", "warmup_motion_gen"),
    "ik_filter": ("nbv_planner.reachability", "ik_filter"),
    "sample_candidate_camera_poses": ("nbv_planner.reachability", "sample_candidate_camera_poses"),
    "score_candidate_views": ("nbv_planner.ray_scoring", "score_candidate_views"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        mod = importlib.import_module(module_name)
        val = getattr(mod, attr_name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
