"""Launch pieces shared by every launch file that can target the real robot."""

import yaml
from launch.actions import EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from nbv_planner_ros.targets import CONFIG_FOR_TARGET


def table_static_tf(context, *args, **kwargs):
    """Publish table_frame on the real robot, where nothing else does.

    steve_sim_prep provides it in Gazebo. The real bringup has no notion of the inspection table,
    but RosRobot.table_aabb needs the frame to build the collision box the arm plans around.
    """
    if context.perform_substitution(LaunchConfiguration("target")) != "real":
        return []

    path = context.perform_substitution(LaunchConfiguration("config_file")) or CONFIG_FOR_TARGET["real"]
    with open(path) as stream:
        cfg = yaml.safe_load(stream)["inspection"]
    xyz = context.perform_substitution(LaunchConfiguration("table_xyz")).split()
    return [Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="nbv_table_frame",
        output="screen",
        arguments=["--x", xyz[0], "--y", xyz[1], "--z", xyz[2],
                   "--yaw", context.perform_substitution(LaunchConfiguration("table_yaw")),
                   "--frame-id", cfg.get("world_frame", "base_link"),
                   "--child-frame-id", cfg.get("table_frame", "table_frame")],
    )]


def shutdown_with(node: Node) -> RegisterEventHandler:
    """End the launch when the node exits, instead of leaving the table TF publisher running."""
    return RegisterEventHandler(OnProcessExit(target_action=node, on_exit=[EmitEvent(event=Shutdown())]))
