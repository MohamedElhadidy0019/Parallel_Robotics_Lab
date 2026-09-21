#!/usr/bin/env python3
"""Report the arm's pose, and optionally drive it to a joint target, to pick a start pose by hand.

    python3 jog_arm.py                                  report only
    python3 jog_arm.py 0 -1.2 1.5 -1.8 -1.57 0          move there, then report
"""

import sys
import time

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

BASE_FRAME = "base_link"
CAMERA_FRAME = "camera_link"
CONTROLLER = "/joint_trajectory_controller/follow_joint_trajectory"


def arm_joint_names(state: JointState) -> list[str]:
    order = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")
    named = {key: name for key in order for name in state.name if key + "_joint" in name}
    return [named[key] for key in order if key in named]


def main() -> None:
    targets = [float(v) for v in sys.argv[1:]]
    rclpy.init()
    node = Node("jog_arm")
    buffer = Buffer()
    TransformListener(buffer, node)

    state = {}
    node.create_subscription(JointState, "/joint_states", lambda msg: state.update(msg=msg), 10)
    deadline = time.monotonic() + 10.0
    while "msg" not in state and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    if "msg" not in state:
        raise SystemExit("No /joint_states in 10 s: is the controller up?")

    names = arm_joint_names(state["msg"])
    values = dict(zip(state["msg"].name, state["msg"].position))
    current = np.array([values[n] for n in names])

    if targets:
        if len(targets) != len(names):
            raise SystemExit(f"Expected {len(names)} joint values for {names}")
        client = ActionClient(node, FollowJointTrajectory, CONTROLLER)
        if not client.wait_for_server(timeout_sec=10.0):
            raise SystemExit(f"No action server at {CONTROLLER}")

        trajectory = JointTrajectory(joint_names=names)
        for fraction in np.linspace(0.0, 1.0, 40)[1:]:
            seconds = 8.0 * fraction
            trajectory.points.append(JointTrajectoryPoint(
                positions=(current + (np.array(targets) - current) * fraction).tolist(),
                time_from_start=Duration(sec=int(seconds), nanosec=int((seconds % 1) * 1e9)),
            ))
        goal = client.send_goal_async(FollowJointTrajectory.Goal(trajectory=trajectory))
        rclpy.spin_until_future_complete(node, goal)
        result = goal.result().get_result_async()
        rclpy.spin_until_future_complete(node, result, timeout_sec=20.0)
        print(f"move: error_code {result.result().result.error_code}")
        for _ in range(40):
            rclpy.spin_once(node, timeout_sec=0.05)
        values = dict(zip(state["msg"].name, state["msg"].position))
        current = np.array([values[n] for n in names])

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        try:
            tf = buffer.lookup_transform(BASE_FRAME, CAMERA_FRAME, rclpy.time.Time())
            break
        except Exception:
            tf = None
    if tf is None:
        raise SystemExit(f"No TF {BASE_FRAME} -> {CAMERA_FRAME}")

    t = tf.transform.translation
    q = tf.transform.rotation
    forward = Rotation.from_quat([q.x, q.y, q.z, q.w]).apply([0.0, 0.0, 1.0])
    position = np.array([t.x, t.y, t.z])

    print(f"joints     {names}")
    print(f"           {np.round(current, 4).tolist()}")
    print(f"camera in {BASE_FRAME}: [{t.x:.3f}, {t.y:.3f}, {t.z:.3f}]")
    for distance in (0.3, 0.4, 0.5):
        look = position + forward * distance
        print(f"  looking at (at {distance:.1f} m): [{look[0]:.3f}, {look[1]:.3f}, {look[2]:.3f}]")
    print("\nPaste into inspection_config.yaml:")
    print(f"  start_camera_position_base: [{t.x:.3f}, {t.y:.3f}, {t.z:.3f}]")
    look = position + forward * 0.4
    print(f"  start_look_at_base: [{look[0]:.3f}, {look[1]:.3f}, {look[2]:.3f}]")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
