import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import yaml


def table_static_tf(context, *args, **kwargs):
    """Publish table_frame on the real robot, where nothing else does.

    steve_sim_prep provides it in Gazebo. The real bringup has no notion of the inspection table,
    but RosRobot.table_aabb needs the frame to build the collision box the arm plans around.
    """
    if context.perform_substitution(LaunchConfiguration("target")) != "real":
        return []

    path = context.perform_substitution(LaunchConfiguration("config_file")) or os.path.join(
        get_package_share_directory("nbv_planner_ros"), "config", "inspection_config_real.yaml")
    with open(path) as stream:
        cfg = yaml.safe_load(stream)["inspection"]
    xyz = [str(v) for v in context.perform_substitution(LaunchConfiguration("table_xyz")).split()]
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


def generate_launch_description():
    ycb_assets = os.path.join(
        get_package_share_directory("steve_sim_prep"), "models", "ycb_objects")

    return LaunchDescription([
        DeclareLaunchArgument("target", default_value="sim",
                              description="sim (Gazebo) or real (Steve). Selects the config, the "
                                          "clock source, and whether the safety gate runs."),
        DeclareLaunchArgument("config_file", default_value="",
                              description="Override the config the target selects."),
        DeclareLaunchArgument("object_name", default_value="mustard_bottle"),
        DeclareLaunchArgument("viz", default_value="true"),
        DeclareLaunchArgument("keep_alive", default_value="true"),
        DeclareLaunchArgument("confirm", default_value="",
                              description="Must be 'go' to move the real arm. Ignored for sim."),
        DeclareLaunchArgument("table_xyz", default_value="0.0 -0.60 0.0",
                              description="table_frame origin in the world frame, real target only."),
        DeclareLaunchArgument("table_yaw", default_value="1.5708",
                              description="table_frame yaw in radians, real target only."),
        SetEnvironmentVariable("NBV_YCB_ROOT", ycb_assets),
        OpaqueFunction(function=table_static_tf),
        Node(
            package="nbv_first_view_test",
            executable="first_view_node",
            name="nbv_first_view_test",
            output="screen",
            parameters=[{
                "target": LaunchConfiguration("target"),
                "config_file": LaunchConfiguration("config_file"),
                "object_name": LaunchConfiguration("object_name"),
                "viz": LaunchConfiguration("viz"),
                "keep_alive": LaunchConfiguration("keep_alive"),
                "confirm": LaunchConfiguration("confirm"),
            }],
        ),
    ])
