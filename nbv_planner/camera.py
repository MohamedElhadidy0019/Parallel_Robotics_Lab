"""Pure geometric and pinhole camera utilities for 3D point backprojection."""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from nbv_planner.config import (
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
    min_depth: Optional[float] = None,
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
    if min_depth is not None:
        valid &= z >= min_depth
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
