"""Preserve cuRobo's sampled trajectory, with uniform time dilation."""

import numpy as np
from builtin_interfaces.msg import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


def duration(seconds):
    ns = round(float(seconds) * 1_000_000_000)
    return Duration(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)


def build_trajectory(plan, joint_names, dt, speed_scale):
    if not 0 < speed_scale <= 1 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("Require 0 < speed_scale <= 1 and a positive interpolation dt")
    order = [plan.joint_names.index(name) for name in joint_names]

    def array(value):
        value = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
        if value.ndim == 3 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 2 or not np.isfinite(value).all():
            raise ValueError("Expected one finite trajectory with shape (steps, joints)")
        return value[:, order]

    positions = array(plan.position)
    if len(positions) < 2:
        raise ValueError("cuRobo returned fewer than two trajectory samples")
    velocities = array(plan.velocity) if plan.velocity is not None else None
    accelerations = array(plan.acceleration) if plan.acceleration is not None else None
    msg = JointTrajectory(joint_names=list(joint_names))
    # Zero header stamp means start when accepted; no wall/sim-clock stamp mismatch.
    for i, position in enumerate(positions):
        point = JointTrajectoryPoint(positions=position.tolist(),
                                     time_from_start=duration(i * dt / speed_scale))
        if velocities is not None:
            point.velocities = (velocities[i] * speed_scale).tolist()
        if accelerations is not None:
            point.accelerations = (accelerations[i] * speed_scale ** 2).tolist()
        msg.points.append(point)
    return msg
