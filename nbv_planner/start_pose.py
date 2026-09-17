"""Start pose: where the camera goes, facing the object, before the NBV planner runs.

Given as a camera position and a look-at point in the robot base frame, so it needs no object
mesh or exact pose -- only roughly where the object will be.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from nbv_planner.reachability import camera_lookat_quaternion_xyzw


def start_camera_pose_world(
    position_base: np.ndarray,
    look_at_base: np.ndarray,
    t_base_world: np.ndarray,
    q_base_world_xyzw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Base-frame start pose -> (camera position world, camera quaternion xyzw world, look-at point world)."""
    position_base = np.asarray(position_base, dtype=float)
    look_at_base = np.asarray(look_at_base, dtype=float)
    if np.linalg.norm(look_at_base - position_base) < 1e-3:
        raise ValueError("Start pose position and look-at point must differ")

    R_base = Rotation.from_quat(q_base_world_xyzw)
    position_world = R_base.apply(position_base) + t_base_world
    look_at_world = R_base.apply(look_at_base) + t_base_world
    return position_world, camera_lookat_quaternion_xyzw(position_world, look_at_world), look_at_world


def pose_errors(world_from_camera: np.ndarray, position: np.ndarray, quaternion_xyzw: np.ndarray) -> tuple[float, float]:
    """(position error m, rotation error rad) between a measured camera transform and a target pose."""
    position_error = float(np.linalg.norm(world_from_camera[:3, 3] - np.asarray(position)))
    rotation = Rotation.from_matrix(world_from_camera[:3, :3]).inv() * Rotation.from_quat(quaternion_xyzw)
    return position_error, float(rotation.magnitude())
