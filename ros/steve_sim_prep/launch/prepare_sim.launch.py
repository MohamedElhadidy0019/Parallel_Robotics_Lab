from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    declare_obj = DeclareLaunchArgument(
        "object_name",
        default_value="mustard_bottle",
        description="Target object to spawn (e.g. mustard_bottle, cracker_box, or file path)"
    )
    declare_table_x = DeclareLaunchArgument("table_x", default_value="0.85")
    declare_table_y = DeclareLaunchArgument("table_y", default_value="0.0")
    declare_table_z = DeclareLaunchArgument("table_z", default_value="0.0")
    declare_obj_x = DeclareLaunchArgument("object_x", default_value="0.65")
    declare_obj_y = DeclareLaunchArgument("object_y", default_value="0.0")
    declare_obj_z = DeclareLaunchArgument("object_z", default_value="0.45")

    preparer_node = Node(
        package="steve_sim_prep",
        executable="preparer_node",
        name="steve_sim_preparer",
        output="screen",
        parameters=[{
            "object_name": LaunchConfiguration("object_name"),
            "table_x": LaunchConfiguration("table_x"),
            "table_y": LaunchConfiguration("table_y"),
            "table_z": LaunchConfiguration("table_z"),
            "object_x": LaunchConfiguration("object_x"),
            "object_y": LaunchConfiguration("object_y"),
            "object_z": LaunchConfiguration("object_z"),
        }]
    )

    return LaunchDescription([
        declare_obj,
        declare_table_x,
        declare_table_y,
        declare_table_z,
        declare_obj_x,
        declare_obj_y,
        declare_obj_z,
        preparer_node,
    ])
