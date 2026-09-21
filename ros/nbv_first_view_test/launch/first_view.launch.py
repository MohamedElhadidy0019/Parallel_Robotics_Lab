import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    default_config = os.path.join(
        get_package_share_directory("nbv_planner_ros"), "config", "inspection_config.yaml")
    ycb_assets = os.path.join(
        get_package_share_directory("steve_sim_prep"), "models", "ycb_objects")

    return LaunchDescription([
        DeclareLaunchArgument("config_file", default_value=default_config),
        DeclareLaunchArgument("object_name", default_value="mustard_bottle"),
        DeclareLaunchArgument("viz", default_value="true"),
        DeclareLaunchArgument("keep_alive", default_value="true"),
        SetEnvironmentVariable("NBV_YCB_ROOT", ycb_assets),
        Node(
            package="nbv_first_view_test",
            executable="first_view_node",
            name="nbv_first_view_test",
            output="screen",
            parameters=[{
                "config_file": LaunchConfiguration("config_file"),
                "object_name": LaunchConfiguration("object_name"),
                "viz": LaunchConfiguration("viz"),
                "keep_alive": LaunchConfiguration("keep_alive"),
            }],
        ),
    ])
