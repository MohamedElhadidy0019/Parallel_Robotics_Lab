"""Start pose math: base-frame pose to world, look-at orientation, and pose error.

Pure math -- no robot, no sim, no GPU.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from nbv_planner.start_pose import pose_errors, start_camera_pose_world

BASE_T = np.array([1.0, -0.5, 0.2])
BASE_Q = Rotation.from_euler("z", 90, degrees=True).as_quat()


def test_start_pose_is_expressed_in_base_frame():
    """A base rotated 90 deg about Z maps base +X to world +Y."""
    t, _, look_at = start_camera_pose_world([0.5, 0.0, 1.0], [0.8, 0.0, 0.7], BASE_T, BASE_Q)

    assert np.allclose(t, BASE_T + [0.0, 0.5, 1.0], atol=1e-9)
    assert np.allclose(look_at, BASE_T + [0.0, 0.8, 0.7], atol=1e-9)


def test_start_camera_faces_look_at_point():
    t, q, look_at = start_camera_pose_world([0.5, 0.1, 1.15], [0.785, 0.0, 0.85], BASE_T, BASE_Q)
    R = Rotation.from_quat(q).as_matrix()

    assert np.allclose(R[:, 2], (look_at - t) / np.linalg.norm(look_at - t), atol=1e-6)
    assert R[2, 1] < 0.0  # image "down" (+Y optical) points down in the world: camera is upright


def test_start_pose_rejects_degenerate_look_at():
    with pytest.raises(ValueError):
        start_camera_pose_world([0.5, 0.0, 1.0], [0.5, 0.0, 1.0], BASE_T, BASE_Q)


def test_pose_errors():
    T = np.eye(4)
    T[:3, 3] = [0.1, 0.2, 0.3]
    q_off = Rotation.from_euler("x", 0.04).as_quat()

    pos_err, rot_err = pose_errors(T, [0.1, 0.2, 0.31], q_off)

    assert pos_err == pytest.approx(0.01)
    assert rot_err == pytest.approx(0.04)
