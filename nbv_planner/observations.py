"""Calibrated RGB-D observations and world-coordinate geometry."""
from dataclasses import dataclass
import numpy as np
from nbv_planner.camera import CameraIntrinsics
from nbv_planner.config import T_OPENGL_OPTICAL


@dataclass(frozen=True)
class Observation:
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: CameraIntrinsics
    world_from_camera: np.ndarray
    timestamp: float

    @property
    def view_matrix(self) -> np.ndarray:
        return T_OPENGL_OPTICAL @ np.linalg.inv(self.world_from_camera)

    @property
    def camera_position(self) -> np.ndarray:
        return self.world_from_camera[:3, 3]


def project_world_points(points_world: np.ndarray, observation: Observation) -> np.ndarray:
    transform = observation.world_from_camera
    points_camera = (np.asarray(points_world) - transform[:3, 3]) @ transform[:3, :3]
    if not np.isfinite(points_camera).all() or np.any(points_camera[:, 2] <= observation.intrinsics.near):
        raise ValueError('Target projection crosses the camera near plane')
    intrinsics = observation.intrinsics
    return points_camera[:, :2] / points_camera[:, 2:3] * [intrinsics.fx, intrinsics.fy] + [intrinsics.cx, intrinsics.cy]
