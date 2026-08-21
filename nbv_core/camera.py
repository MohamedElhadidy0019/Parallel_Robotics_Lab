"""Render RGB-D in PyBullet and turn the depth image into 3D points."""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pybullet as p

from nbv_core.config import (
    DEFAULT_CAMERA_FAR,
    DEFAULT_CAMERA_FOV,
    DEFAULT_CAMERA_HEIGHT,
    DEFAULT_CAMERA_NEAR,
    DEFAULT_CAMERA_WIDTH,
)


@dataclass
class CameraIntrinsics:
    """Pinhole camera parameters. fov is vertical, in degrees."""

    width: int = DEFAULT_CAMERA_WIDTH
    height: int = DEFAULT_CAMERA_HEIGHT
    fov: float = DEFAULT_CAMERA_FOV
    near: float = DEFAULT_CAMERA_NEAR
    far: float = DEFAULT_CAMERA_FAR

    @property
    def fy(self) -> float:
        """Vertical focal length, in pixels."""
        return self.height / (2.0 * np.tan(np.radians(self.fov) / 2.0))

    @property
    def fx(self) -> float:
        """Horizontal focal length, in pixels."""
        return self.fy  # square pixels

    @property
    def cx(self) -> float:
        """Optical centre, x."""
        return self.width / 2.0

    @property
    def cy(self) -> float:
        """Optical centre, y."""
        return self.height / 2.0


def depth_buffer_to_linear(depth_buf: np.ndarray, near: float, far: float) -> np.ndarray:
    """Convert PyBullet's [0, 1] depth buffer to metres."""
    return far * near / (far - (far - near) * depth_buf)


def edge_discontinuity_mask(depth_m: np.ndarray, threshold_m: float = 0.02) -> np.ndarray:
    """True where a pixel's depth agrees with its four neighbours."""
    pad = np.pad(depth_m, 1, mode="edge")
    c = pad[1:-1, 1:-1]
    diffs = np.stack([
        np.abs(c - pad[:-2, 1:-1]),
        np.abs(c - pad[2:, 1:-1]),
        np.abs(c - pad[1:-1, :-2]),
        np.abs(c - pad[1:-1, 2:]),
    ])
    return diffs.max(axis=0) < threshold_m


def backproject_depth(
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    rgb: Optional[np.ndarray] = None,
    max_depth: Optional[float] = None,
    drop_edges: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Depth image to (N, 3) points in the camera frame, plus their colours.

    Camera frame is X right, Y down, Z forward.
    """
    h, w = depth_m.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    z = depth_m
    x = (u - intrinsics.cx) * z / intrinsics.fx
    y = (v - intrinsics.cy) * z / intrinsics.fy

    valid = (z > 0) & np.isfinite(z)
    if max_depth is not None:
        valid &= z <= max_depth
    if drop_edges:
        valid &= edge_discontinuity_mask(depth_m)

    points_cam = np.stack([x[valid], y[valid], z[valid]], axis=-1).astype(np.float32)
    colors = rgb[valid] if rgb is not None else None
    return points_cam, colors


def transform_points(points: np.ndarray, transform_4x4: np.ndarray) -> np.ndarray:
    """Apply a 4x4 rigid transform to (N, 3) points."""
    if points.shape[0] == 0:
        return points.copy().astype(np.float32)

    R = transform_4x4[:3, :3]
    t = transform_4x4[:3, 3:4]
    return (R @ points.T + t).T


def capture_rgbd(
    cam_pos: np.ndarray,
    target_pos: np.ndarray,
    up_vector: np.ndarray,
    intrinsics: CameraIntrinsics,
    physics_client_id: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Render one view. Returns (rgb, depth in metres, view matrix, projection matrix)."""
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

    rgb = np.array(rgb_img, dtype=np.uint8)[:, :, :3]  # drop alpha
    depth_buf = np.array(depth_img, dtype=np.float32)
    depth_m = depth_buffer_to_linear(depth_buf, intrinsics.near, intrinsics.far).astype(np.float32)

    # PyBullet returns these flat and column-major.
    view_np = np.array(view_matrix, dtype=np.float32).reshape((4, 4), order="F")
    proj_np = np.array(proj_matrix, dtype=np.float32).reshape((4, 4), order="F")
    return rgb, depth_m, view_np, proj_np
