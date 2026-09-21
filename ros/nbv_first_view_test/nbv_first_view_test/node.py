"""Stage 1 of the inspection pipeline on its own: reach the start pose, capture one frame, show it.

Drives the same RosRobot, the same _reach_start_pose and the same NBVVisualizer the inspection
node uses, so a pass here means the production path works up to the first image. It loads no CAD
mesh, samples no surface, and runs no NBV loop.
"""

import os
import threading
import traceback

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter

# Stage 1 of the pipeline, imported rather than reimplemented. A copy here would pass while the
# path the inspection node actually takes was broken, which is the failure this package exists
# to catch.
from nbv_planner.pipeline import _reach_start_pose
from nbv_planner.viz import NBVVisualizer
from nbv_planner_ros.robot_model import detect_joint_prefix, urdf_names, wait_for_robot_description
from nbv_planner_ros.ros_robot import RosRobot

CONFIG_DIR = os.path.join(get_package_share_directory("nbv_planner_ros"), "config")
CONFIG_FOR_TARGET = {
    "sim": os.path.join(CONFIG_DIR, "inspection_config.yaml"),
    "real": os.path.join(CONFIG_DIR, "inspection_config_real.yaml"),
}

# Prefix the live URDF uses, per target. "ur5" is Gazebo and the RViz mock, both harmless to
# drive. "ur5e" is the physical arm. Declaring one and finding the other means the graph is not
# what the operator thinks it is, which is the moment to stop rather than to guess.
PREFIX_FOR_TARGET = {"sim": "ur5", "real": "ur5e"}

MAX_FK_POSITION_ERROR_M = 0.01
MAX_FK_ROTATION_ERROR_RAD = 0.05

DEPTH_PATH = "world/start_pose/pinhole/depth"


def send_blueprint() -> None:
    """Replace the NBV layout with one built to show a single frame.

    NBVVisualizer lays out Reconstruction, Coverage and Status HUD panels that stay empty here,
    and its 3D view explicitly drops start_pose/pinhole/rgb, so the captured frame is logged and
    then hidden. This package exists to show that frame, so it gets its own panels.
    """
    import rerun as rr
    import rerun.blueprint as rrb

    rr.send_blueprint(rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                name="World and robot",
                origin="world",
                contents=["+ $origin/**", f"- {DEPTH_PATH}"],
            ),
            rrb.Vertical(
                rrb.Spatial2DView(name="First frame (RGB)",
                                  origin="world/start_pose/pinhole",
                                  contents=["+ $origin/rgb"]),
                rrb.Spatial2DView(name="First frame (depth)",
                                  origin="world/start_pose/pinhole",
                                  contents=["+ $origin/depth"]),
            ),
            column_shares=[1.6, 1.0],
        ),
    ))


def latched(depth: int = 1):
    """QoS the UR status topics publish with.

    robot_mode and safety_mode are latched. A default Volatile subscription receives nothing at
    all and reads identically to a robot that is powered down, which would turn a missing
    subscription into a false all-clear.
    """
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

    return QoSProfile(depth=depth, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                      reliability=ReliabilityPolicy.RELIABLE)


def log_depth(observation) -> None:
    """Depth beside the RGB, in metres, sharing the start pose pinhole."""
    import rerun as rr

    rr.log(DEPTH_PATH, rr.DepthImage(observation.depth_m.astype("float32"), meter=1.0))


class FirstViewNode(Node):
    def __init__(self):
        super().__init__("nbv_first_view_test")
        self.declare_parameter("target", "sim")
        self.declare_parameter("config_file", "")
        self.declare_parameter("object_name", "mustard_bottle")
        self.declare_parameter("viz", True)
        self.declare_parameter("keep_alive", True)
        self.declare_parameter("confirm", "")

        self.target = self.param("target")
        if self.target not in CONFIG_FOR_TARGET:
            raise RuntimeError(f"target must be one of {sorted(CONFIG_FOR_TARGET)}, got {self.target!r}")

        # Only the simulator publishes /clock. rclpy's TimeSource declares use_sim_time False
        # before this constructor runs, so nothing sets it for us, and leaving it False in Gazebo
        # stamps every controller goal in wall time and schedules it decades out. Setting it here
        # re-runs TimeSource's parameter callback, which swaps the clock. On the real robot there
        # is no /clock, so wall time is correct and it stays False.
        if self.target == "sim" and not self.get_parameter("use_sim_time").value:
            self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])

        self.failed = False
        self.done = threading.Event()
        self.cfg = self.load_config()
        self.worker = threading.Thread(target=self.run, daemon=True)
        self.worker.start()

    def param(self, name):
        return self.get_parameter(name).value

    def load_config(self) -> dict:
        path = self.param("config_file") or CONFIG_FOR_TARGET[self.target]
        if not path or not os.path.isfile(path):
            raise RuntimeError(f"config_file not found: {path or '(empty)'}")
        with open(path) as stream:
            config = yaml.safe_load(stream).get("inspection", {}) or {}
        for key in ("start_camera_position_base", "start_look_at_base"):
            if key not in config:
                raise RuntimeError(f"{path} is missing inspection.{key}")
        config["object_name"] = self.param("object_name")
        self.get_logger().info(f"Target: {self.target} | config: {path}")
        return config

    def check_target_matches_graph(self) -> None:
        """Refuse to run when the declared target is not the robot on the graph.

        The target is declared rather than detected because detection failing towards "sim" only
        wastes a run, while detection failing towards "real" would drive physical hardware with
        the safety gate skipped. Declaring it and then checking the declaration keeps the intent
        explicit and still catches the mistake.

        Runs before anything is built, so a wrong target fails on the mismatch rather than on
        whatever the wrong config happens to reference first.
        """
        expected = PREFIX_FOR_TARGET[self.target]
        actual = detect_joint_prefix(urdf_names(wait_for_robot_description(self))[1])
        if actual != expected:
            other = next((t for t, p in PREFIX_FOR_TARGET.items() if p == actual), None)
            raise RuntimeError(
                f"target:={self.target} expects joints prefixed {expected!r} but the live robot "
                f"publishes {actual!r}." + (f" This looks like target:={other}." if other else ""))
        self.get_logger().info(f"Graph matches target {self.target}: joint prefix {actual!r}")

    def check_safety(self) -> None:
        """Hold until the arm reports it is powered, clear and commandable."""
        import time

        from ur_dashboard_msgs.msg import RobotMode, SafetyMode

        state = {}
        self.create_subscription(RobotMode, "/io_and_status_controller/robot_mode",
                                 lambda m: state.__setitem__("mode", m.mode), latched())
        self.create_subscription(SafetyMode, "/io_and_status_controller/safety_mode",
                                 lambda m: state.__setitem__("safety", m.mode), latched())
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline and len(state) < 2:
            time.sleep(0.1)
        missing = [k for k in ("mode", "safety") if k not in state]
        if missing:
            raise RuntimeError(
                f"No {missing} from /io_and_status_controller after 15 s. Is the arm bringup "
                f"running and ROS_DOMAIN_ID correct? Silence here is not an all-clear.")
        if state["mode"] != RobotMode.RUNNING:
            raise RuntimeError(f"robot_mode is {state['mode']}, need {RobotMode.RUNNING} (RUNNING). "
                               f"Power the arm and clear the safety chain.")
        if state["safety"] != SafetyMode.NORMAL:
            raise RuntimeError(f"safety_mode is {state['safety']}, need {SafetyMode.NORMAL} (NORMAL).")
        if self.param("confirm") != "go":
            raise RuntimeError(
                "This will move the real arm. Re-run with confirm:=go once the cell is clear and "
                "an External Control program is running on the pendant.")
        self.get_logger().warning("Safety checks passed and confirmed; the real arm will move")

    def verify_kinematics(self, robot) -> None:
        """Compare cuRobo forward kinematics against TF, once the arm has stopped moving.

        Both readings are sampled independently, so a moving arm reports a mismatch that is only
        the delay between them.
        """
        robot.wait_until_still()
        position_error, rotation_error = robot.camera_kinematics_error()
        message = (f"Camera kinematics: {position_error * 1000:.1f} mm, "
                   f"{np.degrees(rotation_error):.2f} deg against TF")
        if position_error > MAX_FK_POSITION_ERROR_M or rotation_error > MAX_FK_ROTATION_ERROR_RAD:
            raise RuntimeError(f"{message}: the planning model does not match the live robot")
        self.get_logger().info(message)

    def report(self, observation, position, look_at_world) -> None:
        depth = observation.depth_m
        valid = np.isfinite(depth) & (depth > 0)
        intrinsics = observation.intrinsics
        print(
            f"      First frame: {intrinsics.width}x{intrinsics.height}, "
            f"{100.0 * valid.mean():.1f}% valid depth, "
            f"range {depth[valid].min():.2f} to {depth[valid].max():.2f} m",
            flush=True,
        )
        print(
            f"      Camera at ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}) "
            f"looking at ({look_at_world[0]:.3f}, {look_at_world[1]:.3f}, {look_at_world[2]:.3f})",
            flush=True,
        )

    def run(self) -> None:
        try:
            self.run_once()
        except Exception:
            self.failed = True
            self.get_logger().error(traceback.format_exc())
        finally:
            self.done.set()

    def run_once(self) -> None:
        self.get_logger().info("Checking the live robot description...")
        self.check_target_matches_graph()
        if self.target == "real":
            self.check_safety()
        robot = RosRobot(self, self.cfg)
        robot.wait_for_inputs()
        self.verify_kinematics(robot)

        show = bool(self.param("viz"))
        visualizer = NBVVisualizer(self.cfg["object_name"], enabled=show, mode="cad")
        visualizer.init_scene(robot)
        if show:
            send_blueprint()

        print("=== First view test: start pose and one frame ===", flush=True)
        print("[1/2] Moving arm to start pose with cuRobo...", flush=True)
        position, quaternion, look_at_world, observation = _reach_start_pose(
            robot,
            np.asarray(self.cfg["start_camera_position_base"], dtype=float),
            np.asarray(self.cfg["start_look_at_base"], dtype=float),
            visualizer,
        )

        print("[2/2] Capturing and logging the first frame...", flush=True)
        visualizer.log_start_pose(observation, look_at_world)
        if show:
            log_depth(observation)
        visualizer.update_robot_pose(robot)
        self.report(observation, position, look_at_world)
        print("FIRST VIEW OK", flush=True)


def main(args=None):
    rclpy.init(args=args)
    try:
        node = FirstViewNode()
    except Exception as error:
        print(f"nbv_first_view_test: {error}", flush=True)
        rclpy.try_shutdown()
        raise SystemExit(1)

    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
            if node.done.is_set() and (node.failed or not node.param("keep_alive")):
                break
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown(timeout_sec=3.0)
        node.destroy_node()
        rclpy.try_shutdown()
    if node.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
