#!/usr/bin/env python3
import os
import sys
import time
import math
import yaml
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity, DeleteEntity
from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros.transform_broadcaster import TransformBroadcaster
from tf2_ros import Buffer, TransformListener
from ament_index_python.packages import get_package_share_directory
from steve_sim_prep.asset_manager import AssetManager
from steve_sim_prep.table_generator import generate_table_urdf

def euler_to_quaternion(roll, pitch, yaw):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    return [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy
    ]

class SteveSimPreparer(Node):
    def __init__(self):
        super().__init__("steve_sim_preparer")

        # 1. Config file parameter
        default_config = ""
        try:
            pkg_share = get_package_share_directory("steve_sim_prep")
            default_config = os.path.join(pkg_share, "config", "table_config.yaml")
        except Exception:
            pass

        self.declare_parameter("config_file", default_config)

        # 2. Table geometry overrides (empty / sentinel default means use config)
        self.declare_parameter("table_shape", "")
        self.declare_parameter("table_height", -1.0)
        self.declare_parameter("table_thickness", -1.0)
        self.declare_parameter("table_depth", -1.0)
        self.declare_parameter("table_width", -1.0)
        self.declare_parameter("table_size", -1.0)
        self.declare_parameter("table_radius", -1.0)

        # 3. Table pose overrides (NaN sentinel means use config)
        self.declare_parameter("table_x", float("nan"))
        self.declare_parameter("table_y", float("nan"))
        self.declare_parameter("table_z", float("nan"))
        self.declare_parameter("table_yaw", float("nan"))

        # 4. Target object & asset parameters
        self.declare_parameter("object_name", "mustard_bottle")
        self.declare_parameter("assets_dir", "")
        self.declare_parameter("object_x", float("nan"))
        self.declare_parameter("object_y", float("nan"))
        self.declare_parameter("object_z", float("nan"))

        # Service clients
        self.spawn_cli = self.create_client(SpawnEntity, "/spawn_entity")
        self.del_cli = self.create_client(DeleteEntity, "/delete_entity")
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        self.dynamic_tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.active_transforms = []
        self.tf_timer = None
        
        assets_dir = self.get_parameter("assets_dir").get_parameter_value().string_value
        self.asset_mgr = AssetManager(assets_dir=assets_dir or None, logger=self.get_logger())

    def load_config(self):
        config_path = self.get_parameter("config_file").get_parameter_value().string_value
        cfg = {}
        if config_path and os.path.exists(config_path):
            self.get_logger().info(f"Loading table configuration from: {config_path}")
            try:
                with open(config_path, "r") as f:
                    cfg = yaml.safe_load(f).get("table", {})
            except Exception as e:
                self.get_logger().warn(f"Failed to parse config file: {e}")
        else:
            self.get_logger().info("No external config file found, using built-in defaults.")

        # Baseline defaults
        table_shape = cfg.get("shape", "rectangle")
        table_height = float(cfg.get("height", 0.40))
        table_thickness = float(cfg.get("thickness", 0.04))
        table_depth = float(cfg.get("depth", 0.50))
        table_width = float(cfg.get("width", 0.80))
        table_size = float(cfg.get("size", 0.60))
        table_radius = float(cfg.get("radius", 0.30))
        table_mat = cfg.get("material", "Gazebo/WoodFloor")

        d_pose = cfg.get("default_pose", {})
        table_x = float(d_pose.get("x", 0.80))
        table_y = float(d_pose.get("y", 0.00))
        table_z = float(d_pose.get("z", 0.00))
        table_yaw = float(d_pose.get("yaw", 0.00))

        # Check if robot is near origin (0, 0) and CLI did not pass table_x/y
        p_tx = self.get_parameter("table_x").get_parameter_value().double_value
        p_ty = self.get_parameter("table_y").get_parameter_value().double_value
        if math.isnan(p_tx) and math.isnan(p_ty):
            for _ in range(15):
                rclpy.spin_once(self, timeout_sec=0.05)
                try:
                    tf = self.tf_buffer.lookup_transform("odom", "base_link", rclpy.time.Time())
                    rx = tf.transform.translation.x
                    ry = tf.transform.translation.y
                    if abs(rx) < 0.5 and abs(ry) < 0.5:
                        table_x = rx
                        table_y = ry + 0.60
                        table_yaw = 0.0
                        self.get_logger().info(
                            f"Auto-aligning table to robot at origin: ({table_x:.2f}, {table_y:.2f})"
                        )
                    break
                except Exception:
                    pass

        # Apply CLI parameter overrides if passed
        p_shape = self.get_parameter("table_shape").get_parameter_value().string_value
        if p_shape: table_shape = p_shape

        p_h = self.get_parameter("table_height").get_parameter_value().double_value
        if p_h > 0: table_height = p_h

        p_th = self.get_parameter("table_thickness").get_parameter_value().double_value
        if p_th > 0: table_thickness = p_th

        p_d = self.get_parameter("table_depth").get_parameter_value().double_value
        if p_d > 0: table_depth = p_d

        p_w = self.get_parameter("table_width").get_parameter_value().double_value
        if p_w > 0: table_width = p_w

        p_sz = self.get_parameter("table_size").get_parameter_value().double_value
        if p_sz > 0: table_size = p_sz

        p_r = self.get_parameter("table_radius").get_parameter_value().double_value
        if p_r > 0: table_radius = p_r

        p_tx = self.get_parameter("table_x").get_parameter_value().double_value
        if not math.isnan(p_tx): table_x = p_tx

        p_ty = self.get_parameter("table_y").get_parameter_value().double_value
        if not math.isnan(p_ty): table_y = p_ty

        p_tz = self.get_parameter("table_z").get_parameter_value().double_value
        if not math.isnan(p_tz): table_z = p_tz

        p_tyaw = self.get_parameter("table_yaw").get_parameter_value().double_value
        if not math.isnan(p_tyaw): table_yaw = p_tyaw

        return {
            "shape": table_shape,
            "height": table_height,
            "thickness": table_thickness,
            "depth": table_depth,
            "width": table_width,
            "size": table_size,
            "radius": table_radius,
            "material": table_mat,
            "x": table_x,
            "y": table_y,
            "z": table_z,
            "yaw": table_yaw,
        }

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

    def delete_entity(self, name):
        if not self.del_cli.service_is_ready():
            self.del_cli.wait_for_service(timeout_sec=2.0)
        req = DeleteEntity.Request()
        req.name = name
        future = self.del_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        res = future.result()
        if res and res.success:
            self.get_logger().info(f"Cleaned previous entity [{name}]")
            return True
        return False

    def spawn(self, name, xml, x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0):
        req = SpawnEntity.Request()
        req.name = name
        req.xml = xml
        req.initial_pose.position.x = float(x)
        req.initial_pose.position.y = float(y)
        req.initial_pose.position.z = float(z)

        q = euler_to_quaternion(roll, pitch, yaw)
        req.initial_pose.orientation.x = float(q[0])
        req.initial_pose.orientation.y = float(q[1])
        req.initial_pose.orientation.z = float(q[2])
        req.initial_pose.orientation.w = float(q[3])

        self.get_logger().info(f"Spawning entity [{name}] at (x={x:.2f}, y={y:.2f}, z={z:.2f}, yaw={yaw:.2f})...")
        future = self.spawn_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        res = future.result()
        if res and res.success:
            self.get_logger().info(f"Successfully spawned [{name}]")
            return True
        else:
            msg = res.status_message if res else "timeout / error"
            self.get_logger().warn(f"Spawn [{name}] failed: {msg}")
            return False

    def add_tf(self, parent, child, x, y, z, roll=0.0, pitch=0.0, yaw=0.0):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent
        t.child_frame_id = child
        t.transform.translation.x = float(x)
        t.transform.translation.y = float(y)
        t.transform.translation.z = float(z)

        q = euler_to_quaternion(roll, pitch, yaw)
        t.transform.rotation.x = float(q[0])
        t.transform.rotation.y = float(q[1])
        t.transform.rotation.z = float(q[2])
        t.transform.rotation.w = float(q[3])

        self.active_transforms.append(t)
        self.get_logger().info(f"Registered TF: {parent} -> {child}")

    def model_pose(self, name):
        """Where Gazebo actually has the model right now, as (x, y, z, roll, pitch, yaw)."""
        import subprocess

        try:
            output = subprocess.run(["gz", "model", "-m", name, "-p"], capture_output=True, text=True, timeout=5.0)
            values = [float(v) for v in output.stdout.split()]
        except Exception:
            return None
        return values if len(values) == 6 else None

    def track_model_frame(self, name, child):
        """Keep a TF frame on the settled pose of a spawned model, which physics may have nudged."""
        pose = self.model_pose(name)
        if pose is None:
            self.get_logger().warn(f"Could not read the pose of {name}; {child} stays at its spawn pose")
            return
        for transform in self.active_transforms:
            if transform.child_frame_id == child:
                x, y, z, roll, pitch, yaw = pose
                q = euler_to_quaternion(roll, pitch, yaw)
                transform.transform.translation.x = x
                transform.transform.translation.y = y
                transform.transform.translation.z = z
                transform.transform.rotation.x, transform.transform.rotation.y = q[0], q[1]
                transform.transform.rotation.z, transform.transform.rotation.w = q[2], q[3]
                self.get_logger().info(
                    f"{child} tracks {name} at ({x:.3f}, {y:.3f}, {z:.3f}), yaw {math.degrees(yaw):.1f} deg"
                )
                return

    def broadcast_all_tf(self):
        now = self.get_clock().now().to_msg()
        for t in self.active_transforms:
            t.header.stamp = now
        if self.active_transforms:
            self.tf_broadcaster.sendTransform(self.active_transforms)
            self.dynamic_tf_broadcaster.sendTransform(self.active_transforms)

    def publish_tf(self, parent, child, x, y, z, roll=0.0, pitch=0.0, yaw=0.0):
        self.add_tf(parent, child, x, y, z, roll, pitch, yaw)
        self.broadcast_all_tf()

def main():
    rclpy.init()
    node = SteveSimPreparer()

    obj_name_param = node.get_parameter("object_name").get_parameter_value().string_value.strip()

    # Handle object list query
    if obj_name_param.lower() in ["list", "help", "--help"]:
        print("\n=======================================================")
        print("          STEVE SIM PREP - AVAILABLE OBJECTS           ")
        print("=======================================================")
        for obj in node.asset_mgr.list_available_objects():
            print(f"  * {obj}")
        print("\nUsage example:")
        print("  ros2 launch steve_sim_prep prepare_sim.launch.py object_name:=chips_can")
        print("  ros2 launch steve_sim_prep prepare_sim.launch.py object_name:=cracker_box")
        print("=======================================================\n")
        node.destroy_node()
        rclpy.shutdown()
        return

    if not node.wait_for_gazebo():
        sys.exit(1)

    # 1. Resolve table parameters from config + CLI overrides
    t_cfg = node.load_config()
    node.get_logger().info(
        f"Table specs: shape={t_cfg['shape']}, height={t_cfg['height']}m, "
        f"thickness={t_cfg['thickness']}m, pose=({t_cfg['x']:.2f}, {t_cfg['y']:.2f}, {t_cfg['z']:.2f}, yaw={t_cfg['yaw']:.1f} deg)"
    )

    # 2. Resolve target object
    p_ox = node.get_parameter("object_x").get_parameter_value().double_value
    p_oy = node.get_parameter("object_y").get_parameter_value().double_value
    p_oz = node.get_parameter("object_z").get_parameter_value().double_value

    entity_name, obj_urdf, z_offset = node.asset_mgr.resolve_object(obj_name_param)

    # Auto-align target on table top with 5mm landing clearance
    obj_x = p_ox if not math.isnan(p_ox) else t_cfg["x"]
    obj_y = p_oy if not math.isnan(p_oy) else t_cfg["y"]
    obj_z = p_oz if not math.isnan(p_oz) else (t_cfg["z"] + t_cfg["height"] + z_offset + 0.005)

    # 3. Clean-first: delete existing entities so re-running is always seamless
    node.get_logger().info("=== Cleaning any previous inspection entities ===")
    cleanup_entities = [
        "inspection_table",
        "target_object",
        entity_name,
        "mustard_bottle",
        "YcbMustardBottle",
        "YcbChipsCan",
        "YcbCrackerBox",
        "YcbGelatinBox",
        "YcbMasterChefCan",
        "YcbOrionPie",
        "YcbPottedMeatCan",
        "YcbTomatoSoupCan",
        "YcbBleachCleanser",
        "Ycbsuger1",
        "Ycbsuger2",
        "Ycbsuger3",
    ]
    for ent in list(dict.fromkeys(cleanup_entities)):
        node.delete_entity(ent)
    time.sleep(0.3)

    # 4. Generate & Spawn Table
    node.get_logger().info("=== Generating and Spawning Inspection Table ===")
    table_urdf = generate_table_urdf(
        shape=t_cfg["shape"],
        height=t_cfg["height"],
        thickness=t_cfg["thickness"],
        depth=t_cfg["depth"],
        width=t_cfg["width"],
        size=t_cfg["size"],
        radius=t_cfg["radius"],
        material=t_cfg["material"]
    )
    yaw_rad = math.radians(t_cfg["yaw"])
    node.spawn("inspection_table", table_urdf, x=t_cfg["x"], y=t_cfg["y"], z=t_cfg["z"], yaw=yaw_rad)
    time.sleep(0.3)

    # 5. Spawn Target Object
    node.get_logger().info(f"=== Spawning Target Object [{entity_name}] on Table Top ===")
    node.spawn(entity_name, obj_urdf, x=obj_x, y=obj_y, z=obj_z)

    # 6. Broadcast Unified TF Tree
    # Connect Gazebo world to robot odom so camera and inspection points share a complete TF tree
    node.add_tf("world", "odom", 0.0, 0.0, 0.0)
    node.add_tf("world", "table_frame", t_cfg["x"], t_cfg["y"], t_cfg["z"], yaw=yaw_rad)
    node.add_tf("world", "object_frame", obj_x, obj_y, obj_z)
    node.broadcast_all_tf()

    # The object settles after spawning, so publish where it came to rest, not where it was dropped.
    # Read it once: polling Gazebo's transport in a loop destabilises gzserver.
    time.sleep(3.0)
    node.track_model_frame(entity_name, "object_frame")
    node.broadcast_all_tf()
    node.tf_timer = node.create_timer(0.1, node.broadcast_all_tf)

    node.get_logger().info("=== Scene Preparation Complete! ===")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
