"""Run the nbv_planner inspection pipeline on the live ROS 2 robot."""

import os
import threading
import warnings

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter

warnings.filterwarnings("ignore", category=UserWarning)

from nbv_planner.pipeline import run_inspection
from nbv_planner.viz import NBVVisualizer
from nbv_planner_ros.ros_robot import RosRobot

DEFAULT_CONFIG_PATH = os.path.join(
    get_package_share_directory("nbv_planner_ros"), "config", "inspection_config.yaml")

MAX_FK_POSITION_ERROR_M = 0.01
MAX_FK_ROTATION_ERROR_RAD = 0.05


class InspectionNode(Node):
    def __init__(self):
        # rclpy's TimeSource declares use_sim_time (default False) before this constructor runs, so
        # the usual has_parameter guard never fires and the node stamps goals in wall time while
        # Gazebo runs on /clock. The controller then schedules them decades out and nothing moves.
        super().__init__("nbv_inspection_node",
                         parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.declare_parameter("config_file", DEFAULT_CONFIG_PATH)
        self.declare_parameter("object_name", "mustard_bottle")
        self.declare_parameter("mode", "cad")
        self.declare_parameter("max_views", 8)
        self.declare_parameter("target_coverage", 0.95)
        self.declare_parameter("scan_views", 8)
        self.declare_parameter("segmenter", "sam")
        self.declare_parameter("viz", True)

        self.cfg = self._load_config()
        self.cfg["object_name"] = self.get_parameter("object_name").get_parameter_value().string_value
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def _load_config(self) -> dict:
        """The inspection config, or a hard failure.

        Running without it used to fall back to literals aimed at the robot's +y side, where the
        Gazebo table is not, and to a home posture the arm does not spawn in.
        """
        path = self.get_parameter("config_file").get_parameter_value().string_value
        if not path:
            raise RuntimeError("config_file is empty; pass an inspection YAML")
        if not os.path.isfile(path):
            raise RuntimeError(f"config_file not found: {path}")
        with open(path, "r") as f:
            config = yaml.safe_load(f).get("inspection", {}) or {}
        for key in ("start_camera_position_base", "start_look_at_base"):
            if key not in config:
                raise RuntimeError(f"{path} is missing inspection.{key}")
        self.get_logger().info(f"Loaded inspection config: {path}")
        return config

    def _verify_camera_kinematics(self, robot: RosRobot) -> None:
        position_error, rotation_error = robot.camera_kinematics_error()
        message = (f"Camera kinematics check: {position_error * 1000:.1f} mm, "
                   f"{np.degrees(rotation_error):.2f} deg against TF")
        if position_error > MAX_FK_POSITION_ERROR_M or rotation_error > MAX_FK_ROTATION_ERROR_RAD:
            raise RuntimeError(f"{message}: the planning model does not match the live robot")
        self.get_logger().info(message)

    def _run(self) -> None:
        try:
            parameter = lambda name: self.get_parameter(name).get_parameter_value()
            self.get_logger().info("Adopting the live robot description...")
            robot = RosRobot(self, self.cfg)
            robot.wait_for_inputs()
            self._verify_camera_kinematics(robot)

            object_name = self.cfg["object_name"]
            result = run_inspection(
                robot,
                NBVVisualizer(object_name, enabled=parameter("viz").bool_value, mode=parameter("mode").string_value),
                object_name=object_name,
                mode=parameter("mode").string_value,
                max_views=parameter("max_views").integer_value,
                target_coverage=parameter("target_coverage").double_value,
                start_position_base=np.asarray(self.cfg["start_camera_position_base"], dtype=float),
                look_at_base=np.asarray(self.cfg["start_look_at_base"], dtype=float),
                segmenter_name=parameter("segmenter").string_value,
                scan_views=parameter("scan_views").integer_value,
            )
            self.get_logger().info(f"Inspection result: {result}")
        except Exception as error:
            import traceback
            self.get_logger().error(f"Inspection failed: {error}\n{traceback.format_exc()}")
        finally:
            rclpy.try_shutdown()


def main(args=None):
    rclpy.init(args=args)
    try:
        node = InspectionNode()
    except Exception as error:
        print(f"nbv_inspection_node: {error}", flush=True)
        rclpy.try_shutdown()
        raise SystemExit(1)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
