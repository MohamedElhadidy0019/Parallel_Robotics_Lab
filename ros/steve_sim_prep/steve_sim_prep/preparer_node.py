#!/usr/bin/env python3
import os
import sys
import time
import math
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity
from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from ament_index_python.packages import get_package_share_directory
from steve_sim_prep.asset_manager import AssetManager

def quaternion_from_euler(ai, aj, ak):
    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci * ck
    cs = ci * sk
    sc = si * ck
    ss = si * sk
    return [
        cj * sc - sj * cs,
        cj * ss + sj * cc,
        cj * cs - sj * sc,
        cj * cc + sj * ss
    ]

class SteveSimPreparer(Node):
    def __init__(self):
        super().__init__("steve_sim_preparer")

        self.declare_parameter("object_name", "mustard_bottle")
        self.declare_parameter("table_x", 0.85)
        self.declare_parameter("table_y", 0.0)
        self.declare_parameter("table_z", 0.0)
        self.declare_parameter("object_x", 0.65)
        self.declare_parameter("object_y", 0.0)
        self.declare_parameter("object_z", 0.45)

        self.spawn_cli = self.create_client(SpawnEntity, "/spawn_entity")
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        self.asset_mgr = AssetManager(logger=self.get_logger())

    def wait_for_gazebo(self, timeout_sec=60.0):
        self.get_logger().info("Waiting for Gazebo service /spawn_entity...")
        start_time = time.time()
        while not self.spawn_cli.wait_for_service(timeout_sec=2.0):
            if time.time() - start_time > timeout_sec:
                self.get_logger().error("Timeout waiting for Gazebo /spawn_entity")
                return False
            self.get_logger().info("Still waiting for /spawn_entity...")
        self.get_logger().info("Gazebo is ready!")
        return True

    def spawn(self, name, xml, x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0):
        req = SpawnEntity.Request()
        req.name = name
        req.xml = xml
        req.initial_pose.position.x = float(x)
        req.initial_pose.position.y = float(y)
        req.initial_pose.position.z = float(z)

        q = quaternion_from_euler(roll, pitch, yaw)
        req.initial_pose.orientation.x = float(q[0])
        req.initial_pose.orientation.y = float(q[1])
        req.initial_pose.orientation.z = float(q[2])
        req.initial_pose.orientation.w = float(q[3])

        self.get_logger().info(f"Spawning entity [{name}] at ({x:.2f}, {y:.2f}, {z:.2f})...")
        future = self.spawn_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        res = future.result()
        if res and res.success:
            self.get_logger().info(f"Successfully spawned [{name}]: {res.status_message}")
            return True
        else:
            msg = res.status_message if res else "No response"
            self.get_logger().warn(f"Spawn [{name}] returned: {msg}")
            return False

    def publish_tf(self, parent, child, x, y, z, roll=0.0, pitch=0.0, yaw=0.0):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent
        t.child_frame_id = child
        t.transform.translation.x = float(x)
        t.transform.translation.y = float(y)
        t.transform.translation.z = float(z)

        q = quaternion_from_euler(roll, pitch, yaw)
        t.transform.rotation.x = float(q[0])
        t.transform.rotation.y = float(q[1])
        t.transform.rotation.z = float(q[2])
        t.transform.rotation.w = float(q[3])

        self.tf_broadcaster.sendTransform(t)
        self.get_logger().info(f"Broadcasted TF: {parent} -> {child}")

def main():
    rclpy.init()
    node = SteveSimPreparer()

    if not node.wait_for_gazebo():
        sys.exit(1)

    obj_name_param = node.get_parameter("object_name").get_parameter_value().string_value
    table_x = node.get_parameter("table_x").get_parameter_value().double_value
    table_y = node.get_parameter("table_y").get_parameter_value().double_value
    table_z = node.get_parameter("table_z").get_parameter_value().double_value

    obj_x = node.get_parameter("object_x").get_parameter_value().double_value
    obj_y = node.get_parameter("object_y").get_parameter_value().double_value
    obj_z = node.get_parameter("object_z").get_parameter_value().double_value

    # 1. Spawn Inspection Table
    node.get_logger().info("=== STEP 1: Spawning Inspection Table ===")
    try:
        pkg_share = get_package_share_directory("steve_sim_prep")
    except Exception:
        pkg_share = "/home/ws/src/steve_sim_prep"
    table_path = os.path.join(pkg_share, "models", "table", "table.urdf")
    with open(table_path, "r") as f:
        table_urdf = f.read()
    node.spawn("inspection_table", table_urdf, x=table_x, y=table_y, z=table_z)
    time.sleep(0.5)

    # 2. Resolve & Spawn Target Object
    node.get_logger().info(f"=== STEP 2: Resolving and Spawning Target Object [{obj_name_param}] ===")
    entity_name, obj_urdf, z_offset = node.asset_mgr.resolve_object(obj_name_param)
    node.spawn(entity_name, obj_urdf, x=obj_x, y=obj_y, z=obj_z)
    node.publish_tf("world", "object_frame", obj_x, obj_y, obj_z)

    node.get_logger().info("=== Scenario Preparation Complete! Spinning for TF broadcast ===")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
