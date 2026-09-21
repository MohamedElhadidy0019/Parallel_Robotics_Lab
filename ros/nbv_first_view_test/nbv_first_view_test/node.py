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
from nbv_planner_ros.ros_robot import RosRobot

DEFAULT_CONFIG_PATH = os.path.join(
    get_package_share_directory("nbv_planner_ros"), "config", "inspection_config.yaml")

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


def log_depth(observation) -> None:
    """Depth beside the RGB, in metres, sharing the start pose pinhole."""
    import rerun as rr

    rr.log(DEPTH_PATH, rr.DepthImage(observation.depth_m.astype("float32"), meter=1.0))


class FirstViewNode(Node):
    def __init__(self):
        # Gazebo publishes /clock. rclpy's TimeSource declares use_sim_time (default False) before
        # this constructor runs, so it has to be overridden here or every controller goal is
        # stamped in wall time and scheduled decades into the simulator's future.
        super().__init__("nbv_first_view_test",
                         parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.declare_parameter("config_file", DEFAULT_CONFIG_PATH)
        self.declare_parameter("object_name", "mustard_bottle")
        self.declare_parameter("viz", True)
        self.declare_parameter("keep_alive", True)

        self.failed = False
        self.done = threading.Event()
        self.cfg = self.load_config()
        self.worker = threading.Thread(target=self.run, daemon=True)
        self.worker.start()

    def param(self, name):
        return self.get_parameter(name).value

    def load_config(self) -> dict:
        path = self.param("config_file")
        if not path or not os.path.isfile(path):
            raise RuntimeError(f"config_file not found: {path or '(empty)'}")
        with open(path) as stream:
            config = yaml.safe_load(stream).get("inspection", {}) or {}
        for key in ("start_camera_position_base", "start_look_at_base"):
            if key not in config:
                raise RuntimeError(f"{path} is missing inspection.{key}")
        config["object_name"] = self.param("object_name")
        self.get_logger().info(f"Config: {path}")
        return config

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
        self.get_logger().info("Adopting the live robot description...")
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
