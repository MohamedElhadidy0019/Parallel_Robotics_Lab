"""The robot interface the inspection pipeline expects, served from ROS 2 topics, TF and a controller action."""

import time

import numpy as np
import rclpy
from rclpy.duration import Duration
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image, JointState
from tf2_ros import Buffer, TransformListener

from nbv_planner.camera import CameraIntrinsics
from nbv_planner.config import ROBOT_CONFIG_PATH
from nbv_planner.coverage import load_ycb_mesh, transform_mesh
from nbv_planner.motion_planning import last_plan_interpolation_dt, set_robot_model
from nbv_planner.observations import Observation
from nbv_planner.viewer_urdf import viewer_urdf
from nbv_planner_ros.robot_model import (
    REFIT_COLLISION_LINKS,
    REFIT_SPHERE_MAX_RADIUS_M,
    adapt_robot_config,
    camera_link_from_handeye,
    camera_pose_errors,
    load_packaged_config,
    wait_for_robot_description,
    write_urdf,
)
from nbv_planner_ros.trajectory_client import TrajectoryClient

ENCODING_DTYPES = {"32FC1": np.float32, "16UC1": np.uint16, "rgb8": np.uint8, "bgr8": np.uint8}


def image_array(msg: Image) -> np.ndarray:
    dtype = ENCODING_DTYPES.get(msg.encoding)
    if dtype is None:
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")
    channels = 3 if msg.encoding in ("rgb8", "bgr8") else 1
    array = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width, channels)
    if msg.encoding == "bgr8":
        array = array[..., ::-1]
    return array if channels == 3 else array[..., 0]


class RosRobot:
    """Adapts a ROS 2 robot to the interface nbv_planner.pipeline drives."""

    def __init__(self, node, config: dict) -> None:
        self.node = node
        self.config = config
        self.max_reach_m = float(config.get("max_reach", 0.85))
        self.camera_frame = config.get("camera_optical_frame", "camera_depth_optical_frame")
        self.world_frame = config.get("world_frame", "world")
        self.table_frame = config.get("table_frame", "table_frame")
        self.object_frame = config.get("object_frame", "object_frame")

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, node)

        self.urdf_xml = wait_for_robot_description(node)
        # A camera published only as a static TF is invisible to cuRobo, which reads kinematics
        # from the URDF alone. Give it a link so the planner has an end effector to plan to.
        handeye = self.config.get("handeye_file")
        if handeye:
            self.urdf_xml = camera_link_from_handeye(
                self.urdf_xml, handeye,
                self.config["camera_mount_link"], self.camera_frame)
            node.get_logger().info(
                f"Injected {self.camera_frame} on {self.config['camera_mount_link']} from {handeye}")
        urdf_xml = self.urdf_xml
        self.urdf_path = write_urdf(urdf_xml)
        self.robot_config, self.base_link, self.arm_joint_names, dropped, refitted = adapt_robot_config(
            load_packaged_config(ROBOT_CONFIG_PATH), urdf_xml, self.camera_frame,
            refit_links=self.config.get("refit_collision_links", REFIT_COLLISION_LINKS),
            refit_max_radius=float(self.config.get("refit_sphere_max_radius", REFIT_SPHERE_MAX_RADIUS_M)),
        )
        self.ee_link = self.camera_frame
        set_robot_model(self.urdf_path, self.robot_config)
        if dropped:
            node.get_logger().warn(f"Collision links absent from the live robot, not avoided: {dropped}")
        for name in refitted:
            count = len(self.robot_config["robot_cfg"]["kinematics"]["collision_spheres"][name])
            node.get_logger().info(f"Refitted {name} collision spheres from the live mesh: {count} spheres")

        self.joint_positions: dict[str, float] = {}
        self.latest_depth: Image | None = None
        self.latest_color: Image | None = None
        self.camera_info: CameraInfo | None = None
        node.create_subscription(JointState, config.get("joint_states_topic", "/joint_states"),
                                 lambda msg: self.joint_positions.update(dict(zip(msg.name, msg.position))), 10)
        node.create_subscription(Image, config.get("depth_topic", "/camera/depth/image_rect_raw"),
                                 lambda msg: setattr(self, "latest_depth", msg), 10)
        node.create_subscription(Image, config.get("color_topic", "/camera/color/image_raw"),
                                 lambda msg: setattr(self, "latest_color", msg), 10)
        node.create_subscription(CameraInfo, config.get("camera_info_topic", "/camera/depth/camera_info"),
                                 lambda msg: setattr(self, "camera_info", msg), 10)

        self.trajectory_client = TrajectoryClient(
            node,
            action_name=config.get("controller_action", "/joint_trajectory_controller/follow_joint_trajectory"),
            joint_names=tuple(self.arm_joint_names),
        )

    def wait_for_inputs(self, timeout_sec: float = 30.0) -> None:
        """Block until joint states, camera info and a depth frame have arrived."""
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline and rclpy.ok():
            if self.joint_positions and self.camera_info is not None and self.latest_depth is not None:
                return
            time.sleep(0.1)
        missing = [name for name, value in (("joint_states", self.joint_positions),
                                            ("camera_info", self.camera_info),
                                            ("depth", self.latest_depth)) if not value]
        raise RuntimeError(f"Robot inputs missing after {timeout_sec:.0f} s: {missing}")

    @property
    def intrinsics(self) -> CameraIntrinsics:
        info = self.camera_info
        fy = float(info.k[4])
        return CameraIntrinsics(
            width=info.width,
            height=info.height,
            fov=float(np.degrees(2.0 * np.arctan(info.height / (2.0 * fy)))),
            near=float(self.config.get("camera_near", 0.1)),
            far=float(self.config.get("camera_far", 2.5)),
        )

    def transform(self, target: str, source: str, timeout_sec: float = 5.0, stamp=None) -> np.ndarray:
        """Pose of source in target, at a given stamp when one is given, else the latest."""
        deadline = time.monotonic() + timeout_sec
        while True:
            try:
                when = rclpy.time.Time.from_msg(stamp) if stamp is not None else rclpy.time.Time()
                tf = self.tf_buffer.lookup_transform(target, source, when,
                                                     timeout=Duration(seconds=0.5)).transform
                break
            except Exception:
                if time.monotonic() > deadline:
                    raise RuntimeError(f"No transform {target} -> {source}")
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat([tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w]).as_matrix()
        pose[:3, 3] = [tf.translation.x, tf.translation.y, tf.translation.z]
        return pose

    def base_pose(self) -> tuple[np.ndarray, np.ndarray]:
        pose = self.transform(self.world_frame, self.base_link)
        return pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_quat()

    def arm_base_pos(self) -> np.ndarray:
        for link in (f"{self.arm_joint_names[0][:-len('shoulder_pan_joint')]}base_link", self.base_link):
            try:
                return self.transform(self.world_frame, link, timeout_sec=1.0)[:3, 3]
            except RuntimeError:
                continue
        raise RuntimeError("No arm base transform")

    def max_reach(self) -> float:
        return self.max_reach_m

    def current_arm_joints(self) -> np.ndarray:
        missing = [name for name in self.arm_joint_names if name not in self.joint_positions]
        if missing:
            raise RuntimeError(f"Joint states missing: {missing}")
        return np.array([self.joint_positions[name] for name in self.arm_joint_names], dtype=np.float32)

    def camera_world_transform(self) -> np.ndarray:
        return self.transform(self.world_frame, self.camera_frame)

    @property
    def table_surface_z(self) -> float:
        return float(self.transform(self.world_frame, self.table_frame)[2, 3]) + float(self.config.get("table_height", 0.70))

    @property
    def table_aabb(self) -> tuple[np.ndarray, np.ndarray]:
        """Whole table in world coordinates, floor to top surface.

        The spawned table is a top slab on a pedestal, not a floating slab. Modelling only the
        slab leaves the pedestal invisible to the planner, which then drives the arm through it.
        The box runs from the ground to the top surface at the tabletop's full footprint, which
        also covers the narrower pedestal.
        """
        pose = self.transform(self.world_frame, self.table_frame)
        footprint = np.array([float(self.config.get("table_depth", 0.5)),
                              float(self.config.get("table_width", 0.8)), 0.0])
        half = np.abs(pose[:3, :3] @ (footprint / 2.0))
        ground_z = float(pose[2, 3])
        lo = np.array([pose[0, 3] - half[0], pose[1, 3] - half[1], ground_z])
        hi = np.array([pose[0, 3] + half[0], pose[1, 3] + half[1], self.table_surface_z])
        return lo, hi

    def object_mesh_world(self):
        """CAD mesh of the target placed at the object frame the scene publishes."""
        pose = self.transform(self.world_frame, self.object_frame)
        name = self.config["object_name"]
        candidates = [name, "Ycb" + "".join(part.capitalize() for part in name.split("_"))]
        for candidate in candidates:
            try:
                mesh = load_ycb_mesh(candidate)
            except Exception:
                continue
            return transform_mesh(mesh, pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_quat(),
                                  obj_name=candidate, is_inertial_frame=False)
        raise FileNotFoundError(f"No CAD mesh for {candidates}")

    def capture_observation(self) -> Observation:
        """Freshest depth and colour frame, taken once the arm has stopped, with the camera pose of that frame."""
        self.wait_until_still()
        self.latest_depth = None
        deadline = time.monotonic() + float(self.config.get("frame_timeout", 30.0))
        while self.latest_depth is None and time.monotonic() < deadline and rclpy.ok():
            time.sleep(0.02)
        if self.latest_depth is None:
            raise RuntimeError("No depth frame received; is the camera publishing and the simulator stepping?")

        frame = self.latest_depth
        depth = image_array(frame).astype(np.float32)
        if frame.encoding == "16UC1":
            depth = depth / 1000.0
        color = image_array(self.latest_color) if self.latest_color is not None else np.zeros((*depth.shape, 3), np.uint8)
        if color.shape[:2] != depth.shape:
            color = np.zeros((*depth.shape, 3), np.uint8)

        # The camera pose must be the one that took this frame, not wherever the arm has moved since.
        try:
            pose = self.transform(self.world_frame, self.camera_frame, stamp=frame.header.stamp)
        except RuntimeError:
            pose = self.camera_world_transform()
        return Observation(color, depth, self.intrinsics, pose, time.monotonic())

    def wait_until_still(self, tolerance: float = 1e-3, timeout_sec: float = 3.0) -> None:
        """Wait for the joints to stop moving, the way the simulator settles before a shot."""
        deadline = time.monotonic() + timeout_sec
        previous = self.current_arm_joints()
        while time.monotonic() < deadline and rclpy.ok():
            time.sleep(0.1)
            current = self.current_arm_joints()
            if np.max(np.abs(current - previous)) < tolerance:
                return
            previous = current

    def execute_trajectory(self, trajectory, visualizer=None) -> tuple[np.ndarray, float]:
        """Drive the arm through the controller, streaming the robot pose to the viewer while it moves."""
        started = time.perf_counter()
        dt = last_plan_interpolation_dt()
        if dt is None:
            raise RuntimeError("No planned sample interval; execute_trajectory follows a cuRobo plan")
        ok, message, _ = self.trajectory_client.execute(
            trajectory,
            dt=dt,
            speed_scale=float(self.config.get("speed_scale", 0.5)),
            settle_sec=float(self.config.get("settle_time_sec", 0.5)),
            on_tick=(lambda: visualizer.update_robot_pose(self)) if visualizer is not None else None,
        )
        if not ok:
            self.node.get_logger().warn(f"Arm motion issue: {message}")
        if visualizer is not None:
            visualizer.update_robot_pose(self)
        return self.camera_world_transform()[:3, 3], (time.perf_counter() - started) * 1000.0

    @property
    def viewer_urdf_path(self) -> str:
        if not getattr(self, "_viewer_urdf_path", None):
            self._viewer_urdf_path = viewer_urdf(self.urdf_xml, name="viewer_robot.urdf")
        return self._viewer_urdf_path

    def link_pose(self, link_name: str) -> tuple[np.ndarray, np.ndarray]:
        pose = self.transform(self.world_frame, link_name)
        return pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_quat()

    def joint_values(self) -> dict[str, float]:
        return dict(self.joint_positions)

    def camera_kinematics_error(self) -> tuple[float, float]:
        """How far cuRobo's camera forward kinematics sits from the camera pose TF reports."""
        pose = self.transform(self.base_link, self.camera_frame)
        return camera_pose_errors(self.urdf_path, self.robot_config, self.base_link, self.camera_frame,
                                  self.arm_joint_names, self.current_arm_joints(),
                                  pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_quat())
