import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

from nbv_planner_ros.launch_support import shutdown_with, table_static_tf


def generate_launch_description():
    ycb_assets = os.path.join(
        get_package_share_directory("steve_sim_prep"), "models", "ycb_objects")

    node = Node(
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
    )

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
                              description="'go' moves the real arm. Anything else on target:=real plans "
                                          "and stops before the first motion. Ignored for sim."),
        DeclareLaunchArgument("table_xyz", default_value="0.0 -0.60 0.0",
                              description="table_frame origin in the world frame, real target only."),
        DeclareLaunchArgument("table_yaw", default_value="1.5708",
                              description="table_frame yaw in radians, real target only."),
        SetEnvironmentVariable("NBV_YCB_ROOT", ycb_assets),
        OpaqueFunction(function=table_static_tf),
        node,
        shutdown_with(node),
    ])
