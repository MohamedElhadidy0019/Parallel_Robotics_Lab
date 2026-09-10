"""Synthetic camera rendering for PyBullet simulation."""

from typing import Tuple
import numpy as np
import pybullet as p

from nbv_planner.camera import CameraIntrinsics


def depth_buffer_to_linear(depth_buf: np.ndarray, near: float, far: float) -> np.ndarray:
    """Convert PyBullet's [0, 1] depth buffer to metres."""
    return far * near / (far - (far - near) * depth_buf)


def capture_rgbd(
    cam_pos: np.ndarray,
    target_pos: np.ndarray,
    up_vector: np.ndarray,
    intrinsics: CameraIntrinsics,
    physics_client_id: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Render one RGB-D frame from PyBullet simulation.

    Returns (rgb (H, W, 3), depth_m (H, W), view_matrix_4x4, proj_matrix_4x4).
    """
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=cam_pos.tolist(),
        cameraTargetPosition=target_pos.tolist(),
        cameraUpVector=up_vector.tolist(),
        physicsClientId=physics_client_id,
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=intrinsics.fov,
        aspect=intrinsics.width / intrinsics.height,
        nearVal=intrinsics.near,
        farVal=intrinsics.far,
        physicsClientId=physics_client_id,
    )
    _, _, rgb_img, depth_img, _ = p.getCameraImage(
        width=intrinsics.width,
        height=intrinsics.height,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
        physicsClientId=physics_client_id,
    )

    rgb = np.array(rgb_img, dtype=np.uint8)[:, :, :3]  # drop alpha channel
    depth_buf = np.array(depth_img, dtype=np.float32)
    depth_m = depth_buffer_to_linear(depth_buf, intrinsics.near, intrinsics.far).astype(np.float32)

    # Convert flat column-major matrices to 4x4 NumPy arrays
    view_np = np.array(view_matrix, dtype=np.float32).reshape((4, 4), order="F")
    proj_np = np.array(proj_matrix, dtype=np.float32).reshape((4, 4), order="F")
    return rgb, depth_m, view_np, proj_np
