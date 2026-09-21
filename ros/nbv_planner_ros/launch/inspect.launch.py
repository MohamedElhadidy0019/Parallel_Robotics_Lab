import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory("nbv_planner_ros")
    default_config = os.path.join(pkg_share, "config", "inspection_config.yaml")

    declare_config_file = DeclareLaunchArgument(
        "config_file",
        default_value=default_config,
        description="Path to inspection configuration YAML file",
    )

    declare_object_name = DeclareLaunchArgument(
        "object_name",
        default_value="mustard_bottle",
        description="Target YCB object name (e.g. mustard_bottle, cracker_box)",
    )

    declare_mode = DeclareLaunchArgument(
        "mode",
        default_value="cad",
        description="cad, scan, or both",
    )

    declare_viz = DeclareLaunchArgument("viz", default_value="true", description="Stream to the Rerun viewer")

    declare_max_views = DeclareLaunchArgument(
        "max_views",
        default_value="8",
        description="Maximum number of inspection views to execute",
    )

    declare_target_coverage = DeclareLaunchArgument(
        "target_coverage",
        default_value="0.95",
        description="Target surface coverage fraction (0.0 - 1.0)",
    )

    declare_use_sim_time = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation clock if true",
    )

    ycb_assets = os.path.join(get_package_share_directory("steve_sim_prep"), "models", "ycb_objects")

    inspection_node = Node(
        package="nbv_planner_ros",
        executable="inspection_node",
        name="nbv_inspection_node",
        output="screen",
        parameters=[{
            "config_file": LaunchConfiguration("config_file"),
            "object_name": LaunchConfiguration("object_name"),
            "mode": LaunchConfiguration("mode"),
            "viz": LaunchConfiguration("viz"),
            "max_views": LaunchConfiguration("max_views"),
            "target_coverage": LaunchConfiguration("target_coverage"),
            "use_sim_time": LaunchConfiguration("use_sim_time"),
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument("rerun_url", default_value=os.environ.get("RERUN_URL", os.environ.get("RERUN_ADDR", os.environ.get("RERUN_SERVER", "rerun+http://127.0.0.1:9876/proxy")))),
        SetEnvironmentVariable("RERUN_URL", LaunchConfiguration("rerun_url")),
        declare_config_file,
        declare_object_name,
        declare_mode,
        declare_viz,
        declare_max_views,
        declare_target_coverage,
        declare_use_sim_time,
        SetEnvironmentVariable("NBV_YCB_ROOT", ycb_assets),
        inspection_node,
    ])
