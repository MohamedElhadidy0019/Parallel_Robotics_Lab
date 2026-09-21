from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    defaults = {
        "config_file": "",
        "use_sim_time": "true",
        "viz": "true",
        "rerun_url": "",
        "plan_only": "false",
        "speed_scale": "0.25",
        "keep_alive": "true",
    }
    return LaunchDescription([
        *[DeclareLaunchArgument(key, default_value=value) for key, value in defaults.items()],
        Node(package="nbv_start_pose_test", executable="start_pose_node", output="screen",
             parameters=[{key: LaunchConfiguration(key) for key in defaults}]),
    ])
