"""Spheres blocking the camera's line of sight. Needs a GPU."""

import numpy as np
import pytest

from nbv_planner.ray_scoring import score_candidate_views, sphere_blocked_mask

CAMERA = np.array([[0.0, 0.0, 0.5]], dtype=np.float32)
POINT = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
NORMAL = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
NO_TRIANGLES = np.zeros((0, 3, 3), dtype=np.float32)


def spheres(*entries):
    return np.array([list(entries)], dtype=np.float32)


def test_sphere_on_the_line_blocks():
    assert sphere_blocked_mask(CAMERA, POINT, spheres([0.0, 0.0, 0.25, 0.05]))[0, 0]


def test_sphere_beside_the_line_does_not_block():
    assert not sphere_blocked_mask(CAMERA, POINT, spheres([0.2, 0.0, 0.25, 0.05]))[0, 0]


def test_sphere_behind_the_point_does_not_block():
    assert not sphere_blocked_mask(CAMERA, POINT, spheres([0.0, 0.0, -0.3, 0.05]))[0, 0]


def test_sphere_holding_the_camera_is_ignored():
    """The camera sits inside the arm's own wrist sphere, which cannot occlude it."""
    assert not sphere_blocked_mask(CAMERA, POINT, spheres([0.0, 0.0, 0.52, 0.06]))[0, 0]


def test_gripper_sphere_just_ahead_of_the_camera_blocks():
    assert sphere_blocked_mask(CAMERA, POINT, spheres([0.0, 0.0, 0.38, 0.05]))[0, 0]


def test_disabled_spheres_are_ignored():
    assert not sphere_blocked_mask(CAMERA, POINT, spheres([0.0, 0.0, 0.25, -0.05]))[0, 0]


def test_blocked_points_drop_out_of_the_score():
    unblocked, _ = score_candidate_views(CAMERA, POINT, NORMAL, NO_TRIANGLES, backface_margin=0.0)
    blocked, mask = score_candidate_views(CAMERA, POINT, NORMAL, NO_TRIANGLES, backface_margin=0.0,
                                          blocker_spheres=spheres([0.0, 0.0, 0.25, 0.05]))
    assert unblocked[0] == 1
    assert blocked[0] == 0
    assert mask.shape == (1, 1) and not mask[0, 0]
