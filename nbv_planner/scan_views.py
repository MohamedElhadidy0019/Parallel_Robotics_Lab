"""Camera poses on a ring around an estimated object box."""

import numpy as np

from nbv_planner.camera import CameraIntrinsics
from nbv_planner.detection import TableBox
from nbv_planner.reachability import camera_lookat_quaternion_xyzw


def ring_radius(box: TableBox, intrinsics: CameraIntrinsics, margin: float = 1.3, min_radius: float = 0.30) -> float:
    """Distance at which the box's bounding sphere fits the vertical field of view."""
    return max(margin * float(np.linalg.norm(box.size)) / 2.0 / np.sin(np.radians(intrinsics.fov) / 2.0), min_radius)


def ring_viewpoints(box: TableBox, intrinsics: CameraIntrinsics, start_position: np.ndarray, n_views: int = 8,
                    elevations_deg: tuple[float, ...] = (40.0, 55.0, 25.0),
                    radius_steps: tuple[float, ...] = (0.0, 0.05)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Candidate poses per azimuth slot, ordered as a loop starting next to the start camera.

    Returns (slot index (N,), positions (N,3), quaternions xyzw (N,4)); within a slot, candidates are in preference order.
    """
    radius = ring_radius(box, intrinsics)
    start_offset = np.asarray(start_position)[:2] - box.center[:2]
    start_azimuth = np.arctan2(start_offset[1], start_offset[0])
    slots, positions, quaternions = [], [], []
    for slot in range(1, n_views + 1):
        azimuth = start_azimuth + 2.0 * np.pi * slot / (n_views + 1)
        for elevation in np.radians(elevations_deg):
            for step in radius_steps:
                r = radius + step
                eye = box.center + r * np.array([np.cos(elevation) * np.cos(azimuth),
                                                 np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])
                slots.append(slot)
                positions.append(eye)
                quaternions.append(camera_lookat_quaternion_xyzw(eye, box.center))
    return np.asarray(slots), np.asarray(positions), np.asarray(quaternions)
