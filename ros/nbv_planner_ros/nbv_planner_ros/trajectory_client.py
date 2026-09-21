"""FollowJointTrajectory action client that preserves the timing cuRobo planned."""

import time
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTolerance
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

DEFAULT_PATH_TOLERANCE_RAD = 0.15
DEFAULT_GOAL_TOLERANCE_RAD = 0.02


def duration(seconds: float) -> Duration:
    nanoseconds = round(float(seconds) * 1_000_000_000)
    return Duration(sec=nanoseconds // 1_000_000_000, nanosec=nanoseconds % 1_000_000_000)


class TrajectoryClient:
    """Action client dispatching joint trajectories to the robot controller."""

    DEFAULT_JOINTS = (
        "ur5shoulder_pan_joint",
        "ur5shoulder_lift_joint",
        "ur5elbow_joint",
        "ur5wrist_1_joint",
        "ur5wrist_2_joint",
        "ur5wrist_3_joint",
    )

    def __init__(
        self,
        node: Node,
        action_name: str = "/joint_trajectory_controller/follow_joint_trajectory",
        joint_names: tuple[str, ...] = DEFAULT_JOINTS,
    ):
        self.node = node
        self.action_name = action_name
        self.joint_names = list(joint_names)
        self.client = ActionClient(self.node, FollowJointTrajectory, self.action_name)

    def wait_for_server(self, timeout_sec: float = 10.0) -> bool:
        """Wait for the action server to become available."""
        ok = self.client.wait_for_server(timeout_sec=timeout_sec)
        if not ok:
            self.node.get_logger().error(f"Action server {self.action_name} not available!")
        return ok

    def build_joint_trajectory(
        self,
        trajectory: np.ndarray,
        dt: float,
        speed_scale: float = 1.0,
    ) -> JointTrajectory:
        """Space an (N, 6) joint path at the planner's own sample interval, stretched uniformly.

        Positions only. Finite-differenced velocities clipped to a limit the positions do not
        respect describe a different motion from the one the positions describe, and the controller
        splines through both, which overshoots the waypoints it was given.
        """
        trajectory = np.asarray(trajectory, dtype=np.float64)
        if trajectory.ndim != 2 or trajectory.shape[1] != len(self.joint_names):
            raise ValueError(f"Trajectory shape {trajectory.shape} != (N, {len(self.joint_names)})")
        if len(trajectory) < 2:
            raise ValueError("Trajectory needs at least two samples")
        if not np.isfinite(trajectory).all():
            raise ValueError("Trajectory contains non-finite joint values")
        if not np.isfinite(dt) or dt <= 0 or not 0 < speed_scale <= 1:
            raise ValueError("Require a positive sample interval and 0 < speed_scale <= 1")

        # Zero header stamp means start when accepted. A stamp taken from this node's clock is
        # wrong the moment the node and the controller disagree about which clock that is.
        message = JointTrajectory(joint_names=list(self.joint_names))
        for i, position in enumerate(trajectory):
            message.points.append(JointTrajectoryPoint(
                positions=position.tolist(),
                time_from_start=duration(i * dt / speed_scale),
            ))
        return message

    def execute(
        self,
        trajectory: np.ndarray,
        dt: float,
        speed_scale: float = 1.0,
        settle_sec: float = 0.5,
        on_tick=None,
        path_tolerance: float = DEFAULT_PATH_TOLERANCE_RAD,
    ) -> tuple[bool, str, float]:
        """Send the trajectory and block until the controller reports a result.

        Returns (success, message, duration_ms).
        """
        if not self.wait_for_server(timeout_sec=5.0):
            return False, "Action server unavailable", 0.0

        message = self.build_joint_trajectory(trajectory, dt=dt, speed_scale=speed_scale)
        goal = FollowJointTrajectory.Goal(trajectory=message, goal_time_tolerance=duration(3.0))
        for name in self.joint_names:
            goal.path_tolerance.append(JointTolerance(name=name, position=path_tolerance))
            goal.goal_tolerance.append(JointTolerance(name=name, position=DEFAULT_GOAL_TOLERANCE_RAD))

        started = time.perf_counter()
        pending = self.client.send_goal_async(goal)
        deadline = time.time() + 5.0
        while not pending.done() and time.time() < deadline and rclpy.ok():
            time.sleep(0.02)
        if not pending.done():
            return False, "Timed out waiting for goal response", 0.0

        handle = pending.result()
        if not handle or not handle.accepted:
            return False, "Goal was rejected by controller", 0.0

        final = message.points[-1].time_from_start
        planned_sec = final.sec + final.nanosec / 1e9
        result_future = handle.get_result_async()
        deadline = time.time() + planned_sec + settle_sec + 10.0
        while not result_future.done() and time.time() < deadline and rclpy.ok():
            time.sleep(0.05)
            if on_tick is not None:
                on_tick()

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if not result_future.done():
            # An abandoned goal keeps driving the arm while the caller reads its pose and moves on.
            self._cancel(handle)
            return False, f"Trajectory execution timed out after {planned_sec:.1f} s of motion", elapsed_ms

        result = result_future.result()
        if not result:
            self._cancel(handle)
            return False, "Controller returned no result", elapsed_ms

        code = result.result.error_code
        if code == FollowJointTrajectory.Result.SUCCESSFUL:
            return True, "Success", elapsed_ms
        return False, f"Controller error code {code}: {result.result.error_string}", elapsed_ms

    def _cancel(self, handle) -> None:
        """Stop a goal the caller has given up on, and give the executor time to deliver it."""
        future = handle.cancel_goal_async()
        deadline = time.time() + 3.0
        while not future.done() and time.time() < deadline and rclpy.ok():
            time.sleep(0.02)
