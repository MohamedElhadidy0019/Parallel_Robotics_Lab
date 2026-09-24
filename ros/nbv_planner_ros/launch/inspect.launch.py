import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

from nbv_planner_ros.launch_support import shutdown_with, table_static_tf


def generate_launch_description():
    ycb_assets = os.path.join(get_package_share_directory("steve_sim_prep"), "models", "ycb_objects")
    rerun_url = os.environ.get("RERUN_URL", os.environ.get("RERUN_ADDR", os.environ.get(
        "RERUN_SERVER", "rerun+http://127.0.0.1:9876/proxy")))

    node = Node(
        package="nbv_planner_ros",
        executable="inspection_node",
        name="nbv_inspection_node",
        output="screen",
        parameters=[{
            "target": LaunchConfiguration("target"),
            "confirm": LaunchConfiguration("confirm"),
            "config_file": LaunchConfiguration("config_file"),
            "object_name": LaunchConfiguration("object_name"),
            "mode": LaunchConfiguration("mode"),
            "viz": LaunchConfiguration("viz"),
            "max_views": LaunchConfiguration("max_views"),
            "target_coverage": LaunchConfiguration("target_coverage"),
            "scan_views": LaunchConfiguration("scan_views"),
            "segmenter": LaunchConfiguration("segmenter"),
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument("target", default_value="sim",
                              description="sim (Gazebo) or real (Steve). Selects the config, the "
                                          "clock source, and whether the safety gate runs."),
        DeclareLaunchArgument("confirm", default_value="",
                              description="'go' moves the real arm. Anything else on target:=real "
                                          "plans and stops before the first motion. Ignored for sim."),
        DeclareLaunchArgument("config_file", default_value="",
                              description="Override the config the target selects."),
        DeclareLaunchArgument("object_name", default_value="mustard_bottle",
                              description="YCB object name in sim; a label for outputs on the robot."),
        DeclareLaunchArgument("mode", default_value="cad",
                              description="cad, scan, or both. Only scan runs on target:=real."),
        DeclareLaunchArgument("viz", default_value="true", description="Stream to the Rerun viewer"),
        DeclareLaunchArgument("max_views", default_value="8"),
        DeclareLaunchArgument("target_coverage", default_value="0.95"),
        DeclareLaunchArgument("scan_views", default_value="8"),
        DeclareLaunchArgument("segmenter", default_value="sam", description="sam or depth"),
        DeclareLaunchArgument("table_xyz", default_value="0.0 -0.60 0.0",
                              description="table_frame origin in the world frame, real target only."),
        DeclareLaunchArgument("table_yaw", default_value="1.5708",
                              description="table_frame yaw in radians, real target only."),
        DeclareLaunchArgument("rerun_url", default_value=rerun_url),
        SetEnvironmentVariable("RERUN_URL", LaunchConfiguration("rerun_url")),
        SetEnvironmentVariable("NBV_YCB_ROOT", ycb_assets),
        OpaqueFunction(function=table_static_tf),
        node,
        shutdown_with(node),
    ])
