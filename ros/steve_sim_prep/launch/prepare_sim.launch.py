import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    try:
        pkg_share = get_package_share_directory("steve_sim_prep")
        default_config = os.path.join(pkg_share, "config", "table_config.yaml")
    except Exception:
        default_config = ""

    declare_config_file = DeclareLaunchArgument(
        "config_file",
        default_value=default_config,
        description="Path to table configuration YAML file"
    )

    declare_obj = DeclareLaunchArgument(
        "object_name",
        default_value="mustard_bottle",
        description="Target object name (e.g. mustard_bottle, cracker_box)"
    )

    # Optional table pose overrides (leave empty/nan to use config values)
    declare_table_x = DeclareLaunchArgument("table_x", default_value="nan")
    declare_table_y = DeclareLaunchArgument("table_y", default_value="nan")
    declare_table_z = DeclareLaunchArgument("table_z", default_value="nan")
    declare_table_yaw = DeclareLaunchArgument("table_yaw", default_value="nan", description="Table Yaw in degrees (e.g. 90.0)")

    # Optional table shape overrides
    declare_table_shape = DeclareLaunchArgument("table_shape", default_value="")
    declare_table_height = DeclareLaunchArgument("table_height", default_value="-1.0")
    declare_table_thickness = DeclareLaunchArgument("table_thickness", default_value="-1.0")

    # Optional target object pose overrides (leave nan to auto-center on table top)
    declare_obj_x = DeclareLaunchArgument("object_x", default_value="nan")
    declare_obj_y = DeclareLaunchArgument("object_y", default_value="nan")
    declare_obj_z = DeclareLaunchArgument("object_z", default_value="nan")

    preparer_node = Node(
        package="steve_sim_prep",
        executable="preparer_node",
        name="steve_sim_preparer",
        output="screen",
        parameters=[{
            "config_file": LaunchConfiguration("config_file"),
            "object_name": LaunchConfiguration("object_name"),
            "table_x": LaunchConfiguration("table_x"),
            "table_y": LaunchConfiguration("table_y"),
            "table_z": LaunchConfiguration("table_z"),
            "table_yaw": LaunchConfiguration("table_yaw"),
            "table_shape": LaunchConfiguration("table_shape"),
            "table_height": LaunchConfiguration("table_height"),
            "table_thickness": LaunchConfiguration("table_thickness"),
            "object_x": LaunchConfiguration("object_x"),
            "object_y": LaunchConfiguration("object_y"),
            "object_z": LaunchConfiguration("object_z"),
        }]
    )

    return LaunchDescription([
        declare_config_file,
        declare_obj,
        declare_table_x,
        declare_table_y,
        declare_table_z,
        declare_table_yaw,
        declare_table_shape,
        declare_table_height,
        declare_table_thickness,
        declare_obj_x,
        declare_obj_y,
        declare_obj_z,
        preparer_node,
    ])
