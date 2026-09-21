"""Connect to an existing simulation, plan once, execute, and verify the camera pose."""

import os
import threading
import time
import traceback

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTolerance
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState as JointStateMsg

from nbv_planner.config import START_SAFETY_RADIUS_M
from nbv_planner.motion_planning import build_world_config, get_motion_gen
from nbv_planner.start_pose import pose_errors, start_camera_pose_world
from nbv_planner.reachability import quaternion_xyzw_to_wxyz, world_poses_to_base_link_frame
from nbv_planner_ros.ros_robot import RosRobot
from nbv_start_pose_test.trajectory import build_trajectory, duration


class StartPoseNode(Node):
    def __init__(self):
        super().__init__("nbv_start_pose_test", parameter_overrides=[
            Parameter("use_sim_time", value=True)])
        defaults = {"config_file": "", "viz": True, "rerun_url": "", "plan_only": False,
                    "speed_scale": 0.25, "keep_alive": True, "input_timeout": 60.0,
                    "stall_timeout": 60.0, "position_tolerance": 0.02,
                    "rotation_tolerance": 0.05, "path_tolerance": 0.15}
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.failed = False
        self.done = threading.Event()
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.sample = None
        self.sample_number = 0
        self.goal_handle = None
        self.viewer = None
        self.feedback_error = "no controller feedback received"

    def param(self, name):
        return self.get_parameter(name).value

    def receive_joints(self, msg):
        values = dict(zip(msg.name, msg.position))
        if not all(name in values for name in self.robot.arm_joint_names):
            return
        positions = np.array([values[name] for name in self.robot.arm_joint_names])
        with self.lock:
            self.sample_number += 1
            self.sample = (self.sample_number, time.monotonic(),
                           msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec, positions)

    def snapshot(self):
        with self.lock:
            return self.sample

    def alive(self):
        if self.stop.is_set() or not rclpy.ok():
            raise RuntimeError("Interrupted")

    def wait_still(self):
        deadline = time.monotonic() + self.param("input_timeout")
        previous = None
        stable = 0
        while time.monotonic() < deadline:
            self.alive()
            sample = self.snapshot()
            if sample is not None and (previous is None or sample[0] != previous[0]):
                if previous is not None and sample[2] > previous[2]:
                    dt = (sample[2] - previous[2]) / 1e9
                    stable = stable + 1 if np.max(np.abs(sample[3] - previous[3])) / dt < 0.01 else 0
                    if stable >= 5:
                        return sample[3].copy()
                previous = sample
            time.sleep(0.05)
        raise RuntimeError("No fresh, stationary arm joint states; check Gazebo and competing goals")

    def wait_future(self, future, timeout):
        deadline = time.monotonic() + timeout
        while not future.done():
            self.alive()
            if time.monotonic() > deadline:
                raise RuntimeError("Action response timed out")
            time.sleep(0.02)
        return future.result()

    def feedback(self, msg):
        values = msg.feedback.error.positions
        names = msg.feedback.joint_names
        if values:
            i = int(np.argmax(np.abs(values)))
            self.feedback_error = f"largest joint error: {names[i]} = {values[i]:.4f} rad"

    def execute(self, trajectory):
        client = ActionClient(self, FollowJointTrajectory,
                              self.cfg.get("controller_action", "/joint_trajectory_controller/follow_joint_trajectory"))
        if not client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError("Joint trajectory action server unavailable")
        goal = FollowJointTrajectory.Goal(trajectory=trajectory, goal_time_tolerance=duration(3.0))
        for name in trajectory.joint_names:
            goal.path_tolerance.append(JointTolerance(name=name, position=self.param("path_tolerance")))
            goal.goal_tolerance.append(JointTolerance(name=name, position=0.02))
        pending = client.send_goal_async(goal, feedback_callback=self.feedback)
        # If the response arrives after shutdown/timeout, cancel any late acceptance too.
        abandoned = threading.Event()

        def cancel_late(future):
            if abandoned.is_set() and not future.cancelled() and future.exception() is None:
                handle = future.result()
                if handle and handle.accepted:
                    handle.cancel_goal_async()

        pending.add_done_callback(cancel_late)
        try:
            self.goal_handle = self.wait_future(pending, 10.0)
        except Exception:
            abandoned.set()
            if pending.done():
                cancel_late(pending)
            raise
        if not self.goal_handle.accepted:
            raise RuntimeError("Controller rejected trajectory")
        future = self.goal_handle.get_result_async()
        last_clock = start_clock = self.get_clock().now().nanoseconds
        last_progress = time.monotonic()
        final = trajectory.points[-1].time_from_start
        sim_limit = final.sec + final.nanosec / 1e9 + 10.0
        try:
            while not future.done():
                self.alive()
                now = self.get_clock().now().nanoseconds
                if now < last_clock:
                    raise RuntimeError("ROS clock moved backwards during execution")
                if now > last_clock:
                    last_progress = time.monotonic()
                if time.monotonic() - last_progress > self.param("stall_timeout"):
                    raise RuntimeError("Gazebo clock stalled during execution")
                sample = self.snapshot()
                if sample is None or time.monotonic() - sample[1] > self.param("stall_timeout"):
                    raise RuntimeError("Joint feedback stopped during execution")
                if (now - start_clock) / 1e9 > sim_limit:
                    raise RuntimeError("Trajectory exceeded simulation-time deadline")
                last_clock = now
                time.sleep(0.05)
            result = future.result().result
            if result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
                raise RuntimeError(f"Controller {result.error_code}: {result.error_string}; {self.feedback_error}")
        except Exception:
            cancellation = self.goal_handle.cancel_goal_async()
            # Give the executor time to deliver cancellation before the process exits.
            deadline = time.monotonic() + 3.0
            while not cancellation.done() and time.monotonic() < deadline and rclpy.ok():
                time.sleep(0.02)
            raise
        finally:
            self.goal_handle = None

    def update_viewer(self):
        try:
            self.viewer.update()
        except Exception as error:
            self.get_logger().warning(f"Viewer update: {error}", throttle_duration_sec=10.0)

    def run(self):
        try:
            self.run_motion()
        except Exception:
            self.failed = True
            self.get_logger().error(traceback.format_exc())
        finally:
            self.done.set()

    def run_motion(self):
        from curobo.types.math import Pose
        from curobo.types.robot import JointState
        from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

        path = self.param("config_file") or os.path.join(
            get_package_share_directory("nbv_planner_ros"), "config", "inspection_config.yaml")
        with open(path) as stream:
            self.cfg = yaml.safe_load(stream)["inspection"]
        position = np.asarray(self.cfg["start_camera_position_base"], dtype=float)
        look_at = np.asarray(self.cfg["start_look_at_base"], dtype=float)
        if position.shape != (3,) or look_at.shape != (3,) or not np.isfinite([position, look_at]).all():
            raise ValueError("Start position and look-at must be finite 3-vectors")
        self.get_logger().info(f"Config: {path}; start={position.tolist()}, look_at={look_at.tolist()} (base frame)")
        self.robot = RosRobot(self, self.cfg)
        self.create_subscription(JointStateMsg, self.cfg.get("joint_states_topic", "/joint_states"),
                                 self.receive_joints, 10)
        self.wait_still()
        pe, re = self.robot.camera_kinematics_error()
        if pe > 0.01 or re > 0.05:
            raise RuntimeError(f"Live camera FK mismatch: {pe * 1000:.1f} mm / {np.degrees(re):.2f} deg")
        self.get_logger().info(f"Live camera FK verified: {pe * 1000:.1f} mm / {np.degrees(re):.2f} deg")
        if self.param("viz"):
            from nbv_start_pose_test.viewer import RobotViewer
            self.viewer = RobotViewer(self.robot, self.param("rerun_url"))
            self.create_timer(0.2, self.update_viewer)

        base_t, base_q = self.robot.base_pose()
        target_t, target_q, look_world = start_camera_pose_world(position, look_at, base_t, base_q)
        lo, hi = self.robot.table_aabb
        world = build_world_config(base_t, base_q, table_lo=lo, table_hi=hi,
                                   t_obj_world=look_world,
                                   obj_dims=np.full(3, 2 * START_SAFETY_RADIUS_M))
        self.get_logger().info("Warming up reused cuRobo model; robot remains stationary")
        mg = get_motion_gen(world)
        joints = self.wait_still()  # Warmup can take minutes; never use pre-warmup joint values.
        start = JointState.from_position(mg.tensor_args.to_device(joints[None].astype(np.float32)),
                                         joint_names=list(self.robot.arm_joint_names))
        start = start.get_ordered_joint_state(mg.kinematics.joint_names)
        target_base, quaternion_base = world_poses_to_base_link_frame(
            target_t[None], target_q[None], base_t, base_q)
        goal = Pose(position=mg.tensor_args.to_device(target_base.astype(np.float32)),
                    quaternion=mg.tensor_args.to_device(quaternion_xyzw_to_wxyz(quaternion_base).astype(np.float32)))
        self.get_logger().info("Planning collision-aware start-pose motion with cuRobo plan_single")
        result = mg.plan_single(start, goal, MotionGenPlanConfig(max_attempts=10, enable_graph=True))
        if not bool(result.success.item()):
            detail = ""
            if "SELF_COLLISION" in str(result.status):
                from nbv_start_pose_test.collision import self_collision_details
                detail = "; " + self_collision_details(self.robot, joints)
            raise RuntimeError(f"cuRobo failed: {result.status}{detail}; no motion sent, no unplanned home fallback")
        plan = result.get_interpolated_plan()
        trajectory = build_trajectory(plan, self.robot.arm_joint_names,
                                      float(result.interpolation_dt), self.param("speed_scale"))
        self.get_logger().info(f"Plan ready: {len(trajectory.points)} samples, speed scale={self.param('speed_scale')}")
        if self.param("plan_only"):
            self.get_logger().info("PLAN ONLY successful; no controller goal sent")
            return
        current = self.wait_still()
        if np.max(np.abs(current - np.array(trajectory.points[0].positions))) > 0.02:
            raise RuntimeError("Arm changed during planning; refusing a stale trajectory. Run again when stationary")
        new_t, new_q = self.robot.base_pose()
        if np.linalg.norm(new_t - base_t) > 0.005 or abs(np.dot(new_q, base_q)) < 0.9999:
            raise RuntimeError("Mobile base moved during planning; restart from the new base pose")
        self.execute(trajectory)
        self.wait_still()
        # TF and joint states use independent subscriptions; allow fresh TF after the result.
        deadline = time.monotonic() + 5.0
        while True:
            pe, re = pose_errors(self.robot.camera_world_transform(), target_t, target_q)
            if pe <= self.param("position_tolerance") and re <= self.param("rotation_tolerance"):
                self.get_logger().info(f"START POSE REACHED: camera error {pe * 1000:.1f} mm / {np.degrees(re):.2f} deg")
                return
            if time.monotonic() > deadline:
                raise RuntimeError(f"Controller finished but camera missed target: {pe * 1000:.1f} mm / {np.degrees(re):.2f} deg")
            self.alive()
            time.sleep(0.1)


def main(args=None):
    rclpy.init(args=args)
    node = StartPoseNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    worker = threading.Thread(target=node.run, daemon=True)
    worker.start()
    try:
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
            if node.done.is_set() and (node.failed or not node.param("keep_alive")):
                break
    except KeyboardInterrupt:
        pass
    finally:
        node.stop.set()
        if node.goal_handle is not None:
            future = node.goal_handle.cancel_goal_async()
            deadline = time.monotonic() + 3.0
            while not future.done() and time.monotonic() < deadline and rclpy.ok():
                executor.spin_once(timeout_sec=0.1)
        executor.shutdown(timeout_sec=3.0)
        node.destroy_node()
        rclpy.try_shutdown()
    if node.failed:
        raise SystemExit(1)
