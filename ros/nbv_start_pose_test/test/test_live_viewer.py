"""Opt-in live ROS smoke test. Reads the robot; never creates a motion goal."""

import os
import threading
import time

import pytest


@pytest.mark.skipif(os.environ.get("NBV_TEST_LIVE") != "1", reason="requires running Steve simulation")
def test_live_urdf_recording(tmp_path, monkeypatch):
    import rclpy
    import rerun as rr
    import yaml
    from ament_index_python.packages import get_package_share_directory
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from nbv_planner_ros.ros_robot import RosRobot
    from nbv_start_pose_test.viewer import RobotViewer

    rclpy.init()
    node = Node("nbv_start_pose_viewer_smoke", parameter_overrides=[Parameter("use_sim_time", value=True)])
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        path = os.path.join(get_package_share_directory("nbv_planner_ros"), "config", "inspection_config.yaml")
        with open(path) as stream:
            robot = RosRobot(node, yaml.safe_load(stream)["inspection"])
        deadline = time.monotonic() + 30.0
        while not all(name in robot.joint_positions for name in robot.arm_joint_names):
            assert time.monotonic() < deadline, "No live arm joint messages"
            time.sleep(0.1)
        recording = tmp_path / "live_robot.rrd"
        # Exercise real URDF loading and Rerun logging, without opening a GUI/network connection.
        monkeypatch.setattr(rr, "connect_grpc", lambda _: rr.save(str(recording)))
        viewer = RobotViewer(robot, "record-for-test")
        viewer.update()
        rr.get_global_data_recording().flush()
        assert len(list(viewer.tree.joints())) >= 6
        assert recording.stat().st_size > 1000
        if os.environ.get("NBV_TEST_COLLISIONS") == "1":
            from nbv_start_pose_test.collision import self_collision_details
            print("Live joint positions:", robot.current_arm_joints())
            print("Collision diagnostics:", self_collision_details(robot, robot.current_arm_joints()))
    finally:
        executor.shutdown()
        thread.join(timeout=5.0)
        node.destroy_node()
        rclpy.try_shutdown()
