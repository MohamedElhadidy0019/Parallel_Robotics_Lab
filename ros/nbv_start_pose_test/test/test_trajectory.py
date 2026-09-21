from types import SimpleNamespace

import numpy as np
import pytest

from nbv_start_pose_test.trajectory import build_trajectory, duration


def test_time_dilation_preserves_derivatives_and_joint_order():
    plan = SimpleNamespace(
        joint_names=["b", "a"],
        position=np.array([[[0., 1.], [2., 3.], [4., 5.]]]),
        velocity=np.ones((1, 3, 2)) * 2,
        acceleration=np.ones((1, 3, 2)) * 4,
    )
    msg = build_trajectory(plan, ["a", "b"], 0.04, 0.25)
    assert list(msg.points[1].positions) == [3., 2.]
    assert list(msg.points[1].velocities) == [0.5, 0.5]
    assert list(msg.points[1].accelerations) == [0.25, 0.25]
    assert msg.points[0].time_from_start == duration(0)
    assert msg.points[-1].time_from_start == duration(0.32)
    assert msg.header.stamp.sec == 0


@pytest.mark.parametrize("scale", [0, -1, 1.1, float("nan")])
def test_reject_bad_speed(scale):
    with pytest.raises(ValueError):
        build_trajectory(None, [], 0.04, scale)


def test_reject_nonfinite_plan():
    plan = SimpleNamespace(joint_names=["a"], position=np.array([[0.], [np.nan]]))
    with pytest.raises(ValueError):
        build_trajectory(plan, ["a"], 0.04, 1.0)


def test_duration_rounds_without_invalid_nanosecond_field():
    assert duration(0.9999999999).sec == 1
    assert duration(0.9999999999).nanosec == 0
