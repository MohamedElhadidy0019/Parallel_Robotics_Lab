"""Build the cuRobo robot model from the robot the simulator or driver is actually running."""

import os
import tempfile
import time
import xml.etree.ElementTree as ET
from typing import Sequence

import numpy as np
import rclpy
import yaml
from scipy.spatial.transform import Rotation
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

# Packaged model name -> name on a Gazebo/robotiq_description robot. Several pads map onto one live link.
DEFAULT_LINK_ALIASES = {
    "robotiq_arg2f_base_link": "robotiq_85_base_link",
    "left_outer_knuckle": "robotiq_85_left_knuckle_link",
    "left_inner_knuckle": "robotiq_85_left_inner_knuckle_link",
    "left_outer_finger": "robotiq_85_left_finger_link",
    "left_inner_finger": "robotiq_85_left_finger_tip_link",
    "left_inner_finger_pad": "robotiq_85_left_finger_tip_link",
    "right_outer_knuckle": "robotiq_85_right_knuckle_link",
    "right_inner_knuckle": "robotiq_85_right_inner_knuckle_link",
    "right_outer_finger": "robotiq_85_right_finger_link",
    "right_inner_finger": "robotiq_85_right_finger_tip_link",
    "right_inner_finger_pad": "robotiq_85_right_finger_tip_link",
    "dummy_camera_link": "camera_link",
}

# The packaged spheres approximate a box base with a handful of balls, which leaves its corners and
# top uncovered. Refitting them from the live collision mesh closes that gap but inflates the base
# enough to put the configured start pose out of reach, so it stays opt-in until the poses are retuned.
REFIT_COLLISION_LINKS: tuple[str, ...] = ()
REFIT_SPHERE_MAX_RADIUS_M = 0.12

ARM_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


def wait_for_robot_description(node, topic: str = "/robot_description", timeout_sec: float = 30.0) -> str:
    """Latched robot URDF published by robot_state_publisher.

    Spins the node only when nothing else does, so callbacks are not stolen from a running executor.
    """
    received = []
    qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
    subscription = node.create_subscription(String, topic, lambda msg: received.append(msg.data), qos)
    try:
        deadline = time.monotonic() + timeout_sec
        while not received and rclpy.ok() and time.monotonic() < deadline:
            if node.executor is None:
                rclpy.spin_once(node, timeout_sec=0.1)
            else:
                time.sleep(0.1)
    finally:
        node.destroy_subscription(subscription)
    if not received:
        raise RuntimeError(f"No URDF received on {topic}; is robot_state_publisher running?")
    return received[0]


DEFAULT_JOINT_LIMIT = {"lower": "-3.14159", "upper": "3.14159", "effort": "100.0", "velocity": "1.0"}


def add_missing_joint_limits(urdf_xml: str) -> str:
    """cuRobo needs a limit on every movable joint; the continuous mimic joints of the gripper have none."""
    root = ET.fromstring(urdf_xml)
    for joint in root.findall("joint"):
        if joint.get("type") in ("revolute", "prismatic", "continuous") and joint.find("limit") is None:
            ET.SubElement(joint, "limit", DEFAULT_JOINT_LIMIT)
    return ET.tostring(root, encoding="unicode")


def inject_camera_link(urdf_xml: str, parent: str, child: str, translation, quaternion_xyzw) -> str:
    """Add a fixed joint placing a calibrated camera on the robot, as a real link.

    A camera published only as a static TF is invisible to cuRobo, whose kinematics come from the
    URDF alone. The planner's end effector is the camera, so without this the model has nothing to
    plan to. Only the temporary copy handed to cuRobo is changed; the robot's own URDF is not.
    """
    root = ET.fromstring(urdf_xml)
    names = {link.get("name") for link in root.findall("link")}
    if parent not in names:
        raise RuntimeError(f"Camera mount link {parent} is not a link of the live robot")
    if child in names:
        return urdf_xml

    ET.SubElement(root, "link", {"name": child})
    joint = ET.SubElement(root, "joint", {"name": f"{child}_fixed_joint", "type": "fixed"})
    ET.SubElement(joint, "parent", {"link": parent})
    ET.SubElement(joint, "child", {"link": child})
    rpy = Rotation.from_quat(np.asarray(quaternion_xyzw, dtype=float)).as_euler("xyz")
    ET.SubElement(joint, "origin", {
        "xyz": " ".join(f"{v:.9g}" for v in np.asarray(translation, dtype=float)),
        "rpy": " ".join(f"{v:.9g}" for v in rpy),
    })
    return ET.tostring(root, encoding="unicode")


def camera_link_from_handeye(urdf_xml: str, handeye_path: str, parent: str, child: str) -> str:
    """Inject the camera described by a hand-eye calibration JSON.

    Expects the format handeye_solve.py writes: translation_m and quaternion_xyzw, giving the
    camera's pose expressed in the mount link's frame.
    """
    import json

    with open(handeye_path) as stream:
        result = json.load(stream)
    return inject_camera_link(urdf_xml, parent, child,
                              result["translation_m"], result["quaternion_xyzw"])


def write_urdf(urdf_xml: str, directory: str | None = None) -> str:
    """Write a cuRobo-readable copy of the live URDF."""
    directory = directory or tempfile.mkdtemp(prefix="nbv_robot_")
    path = os.path.join(directory, "robot.urdf")
    with open(path, "w") as f:
        f.write(add_missing_joint_limits(urdf_xml))
    return path


def urdf_names(urdf_xml: str) -> tuple[set[str], set[str]]:
    root = ET.fromstring(urdf_xml)
    return ({link.get("name") for link in root.findall("link")},
            {joint.get("name") for joint in root.findall("joint")})


def movable_joints(urdf_xml: str) -> list[str]:
    """Independently driven joints. Mimic joints follow another joint and cannot be locked on their own."""
    root = ET.fromstring(urdf_xml)
    return [j.get("name") for j in root.findall("joint")
            if j.get("type") != "fixed" and j.find("mimic") is None]


def detect_joint_prefix(joint_names: set[str]) -> str:
    """Prefix the live robot adds to the UR5 joint names, empty when it adds none."""
    for name in joint_names:
        if name.endswith(ARM_JOINTS[0]):
            return name[: -len(ARM_JOINTS[0])]
    raise RuntimeError(f"No {ARM_JOINTS[0]} found in the robot description")



def resolve_mesh_path(uri: str) -> str | None:
    """Absolute path for a URDF mesh URI, resolving package:// through the ament index."""
    if uri.startswith("file://"):
        uri = uri[len("file://"):]
    if not uri.startswith("package://"):
        return uri if os.path.isfile(uri) else None
    package, _, relative = uri[len("package://"):].partition("/")
    try:
        from ament_index_python.packages import get_package_share_directory

        path = os.path.join(get_package_share_directory(package), relative)
    except Exception:
        return None
    return path if os.path.isfile(path) else None


def link_collision_aabb(urdf_xml: str, link_name: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Axis-aligned bounds of a link's collision geometry, in the link frame.

    Handles box primitives directly and meshes by loading them, so the numbers come from the live
    robot rather than from a hand-written approximation.
    """
    link = ET.fromstring(urdf_xml).find(f".//link[@name='{link_name}']")
    if link is None:
        return None

    lo, hi = None, None
    for collision in link.findall("collision"):
        geometry = collision.find("geometry")
        if geometry is None:
            continue
        origin = collision.find("origin")
        xyz = np.array([float(v) for v in (origin.get("xyz", "0 0 0").split() if origin is not None else "0 0 0".split())])
        rpy = np.array([float(v) for v in (origin.get("rpy", "0 0 0").split() if origin is not None else "0 0 0".split())])
        rotation = Rotation.from_euler("xyz", rpy).as_matrix()

        box = geometry.find("box")
        mesh = geometry.find("mesh")
        if box is not None:
            half = np.array([float(v) for v in box.get("size", "0 0 0").split()]) / 2.0
            corners = np.array([[x, y, z] for x in (-half[0], half[0])
                                for y in (-half[1], half[1]) for z in (-half[2], half[2])])
        elif mesh is not None:
            path = resolve_mesh_path(mesh.get("filename", ""))
            if path is None:
                continue
            import trimesh

            loaded = trimesh.load(path, force="mesh")
            corners = np.asarray(loaded.vertices) * np.array(
                [float(v) for v in mesh.get("scale", "1 1 1").split()]
            )
        else:
            continue

        corners = corners @ rotation.T + xyz
        lo = corners.min(axis=0) if lo is None else np.minimum(lo, corners.min(axis=0))
        hi = corners.max(axis=0) if hi is None else np.maximum(hi, corners.max(axis=0))

    return None if lo is None else (lo, hi)


def fill_box_with_spheres(lo: np.ndarray, hi: np.ndarray, max_radius: float) -> list[dict]:
    """Grid of spheres that fully encloses a box, as cuRobo collision_spheres entries.

    A handful of large spheres cannot fill a box: the corners fall outside every one of them.
    Each cell here is covered by a sphere through its own corners, so the whole volume is inside.
    """
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    extent = np.maximum(hi - lo, 1e-6)
    counts = np.maximum(1, np.ceil(extent / (2.0 * max_radius / np.sqrt(3.0))).astype(int))
    spacing = extent / counts
    radius = float(np.linalg.norm(spacing) / 2.0)
    axes = [lo[i] + spacing[i] * (np.arange(counts[i]) + 0.5) for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

    # Spheres that bulge past the real box are not free padding. The arm mounts on top of this
    # link, so a fill reaching above the true top surface puts the shoulder and upper arm in
    # permanent self-collision and nothing plans. Pull every centre in far enough that the
    # sphere stays inside, wherever the box is thick enough to allow it.
    inner_lo = np.minimum(lo + radius, (lo + hi) / 2.0)
    inner_hi = np.maximum(hi - radius, (lo + hi) / 2.0)
    grid = np.clip(grid, inner_lo, inner_hi)

    unique = {tuple(center.round(4)): center for center in grid}
    return [{"center": list(center), "radius": round(radius, 4)} for center in unique]


def refit_collision_spheres(spheres: dict, urdf_xml: str, link_names, max_radius: float) -> list[str]:
    """Replace hand-authored spheres with a box fill measured from the live robot. Returns the links refitted."""
    refitted = []
    for name in link_names:
        if name not in spheres:
            continue
        bounds = link_collision_aabb(urdf_xml, name)
        if bounds is None:
            continue
        spheres[name] = fill_box_with_spheres(*bounds, max_radius=max_radius)
        refitted.append(name)
    return refitted

def adapt_robot_config(config: dict, urdf_xml: str, ee_link: str,
                       link_aliases: dict | None = None,
                       refit_links: Sequence[str] = REFIT_COLLISION_LINKS,
                       refit_max_radius: float = REFIT_SPHERE_MAX_RADIUS_M
                       ) -> tuple[dict, str, list[str], list[str], list[str]]:
    """Rename the packaged cuRobo config onto the live robot and point it at the real camera frame.

    Returns (config, base link, arm joint names, collision links the live robot does not have,
    links whose spheres were refitted to the live geometry).
    """
    links, joints = urdf_names(urdf_xml)
    aliases = {**DEFAULT_LINK_ALIASES, **(link_aliases or {})}
    prefix = detect_joint_prefix(joints)
    if ee_link not in links:
        raise RuntimeError(f"Camera frame {ee_link} is not a link of the live robot")

    def live_link(name: str) -> str:
        alias = aliases.get(name, name)
        return alias if alias in links else f"{prefix}{alias}"

    arm_joints = [f"{prefix}{name}" if f"{prefix}{name}" in joints else name for name in ARM_JOINTS]
    kinematics = config["robot_cfg"]["kinematics"]
    kinematics["base_link"] = live_link(kinematics["base_link"])
    kinematics["ee_link"] = ee_link

    # Every joint the arm does not drive stays put, or cuRobo would plan the gripper and the wheels too.
    kinematics["lock_joints"] = {name: 0.0 for name in movable_joints(urdf_xml) if name not in arm_joints}

    spheres, merged, dropped = kinematics.get("collision_spheres") or {}, {}, []
    for name, value in spheres.items():
        target = live_link(name)
        if target in links:
            merged.setdefault(target, []).extend(value)
        else:
            dropped.append(name)
    refitted = refit_collision_spheres(merged, urdf_xml, [live_link(n) for n in refit_links], refit_max_radius)
    kinematics["collision_spheres"] = merged
    kinematics["collision_link_names"] = list(merged)

    ignore = {}
    for name, others in (kinematics.get("self_collision_ignore") or {}).items():
        target = live_link(name)
        if target in links:
            ignore.setdefault(target, []).extend(o for o in map(live_link, others) if o in links)
    kinematics["self_collision_ignore"] = {name: sorted(set(others)) for name, others in ignore.items()}
    kinematics["self_collision_buffer"] = {
        live_link(name): value for name, value in (kinematics.get("self_collision_buffer") or {}).items()
        if live_link(name) in links
    }
    # Spheres carry the collision geometry; link meshes would have to be fetched through package:// URIs.
    kinematics["mesh_link_names"] = []

    cspace = kinematics.get("cspace")
    if cspace:
        keep = [i for i, name in enumerate(cspace["joint_names"])
                if (f"{prefix}{name}" in joints or name in joints) and live_joint(name, prefix, joints) in arm_joints]
        cspace["joint_names"] = [live_joint(cspace["joint_names"][i], prefix, joints) for i in keep]
        for field in ("retract_config", "null_space_weight", "cspace_distance_weight"):
            if field in cspace:
                cspace[field] = [cspace[field][i] for i in keep]

    return config, kinematics["base_link"], arm_joints, sorted(dropped), refitted


def live_joint(name: str, prefix: str, joints: set[str]) -> str:
    return name if name in joints else f"{prefix}{name}"


def load_packaged_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def camera_pose_errors(urdf_path: str, robot_config: dict, base_link: str, ee_link: str,
                       joint_names: list[str], joint_values: np.ndarray,
                       t_camera_base: np.ndarray, q_camera_base_xyzw: np.ndarray) -> tuple[float, float]:
    """(position error m, rotation error rad) between cuRobo forward kinematics and the live TF camera pose."""
    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
    from curobo.types.base import TensorDeviceType
    from curobo.types.robot import RobotConfig
    from scipy.spatial.transform import Rotation

    tensor_args = TensorDeviceType()
    config = dict(robot_config)
    config["robot_cfg"]["kinematics"]["urdf_path"] = urdf_path
    config["robot_cfg"]["kinematics"]["asset_root_path"] = os.path.dirname(urdf_path)
    config["robot_cfg"]["kinematics"]["base_link"] = base_link
    config["robot_cfg"]["kinematics"]["ee_link"] = ee_link

    # Ask for the camera link by name: the end effector pose cuRobo reports is not this frame.
    config["robot_cfg"]["kinematics"]["link_names"] = [ee_link]

    model = CudaRobotModel(RobotConfig.from_dict(config, tensor_args=tensor_args).kinematics)
    order = [list(joint_names).index(name) for name in model.joint_names]
    values = np.asarray(joint_values, dtype=np.float32)[order]
    state = model.get_state(tensor_args.to_device(values[None, :]))

    link = model.link_names.index(ee_link)
    position = state.links_position.cpu().numpy()[0][link]
    quaternion_wxyz = state.links_quaternion.cpu().numpy()[0][link]
    if os.environ.get("NBV_FK_DEBUG"):
        print(f"[fk] joints {np.round(values, 3).tolist()} model order {list(model.joint_names)}", flush=True)
        print(f"[fk] cuRobo {np.round(position, 4).tolist()} | TF {np.round(t_camera_base, 4).tolist()}", flush=True)
    rotation = Rotation.from_quat(np.roll(quaternion_wxyz, -1))
    difference = rotation.inv() * Rotation.from_quat(q_camera_base_xyzw)
    return float(np.linalg.norm(position - t_camera_base)), float(difference.magnitude())


def resolve_mesh_path(filename: str) -> str | None:
    """Turn a package:// or file:// URDF mesh reference into a path on disk."""
    from ament_index_python.packages import get_package_share_directory

    if filename.startswith("file://"):
        filename = filename[len("file://"):]
    if filename.startswith("package://"):
        package, _, relative = filename[len("package://"):].partition("/")
        try:
            filename = os.path.join(get_package_share_directory(package), relative)
        except Exception:
            return None
    return filename if os.path.isfile(filename) else None


VIEWER_MESH_FORMATS = (".obj", ".stl", ".glb", ".gltf")
_MESH_CACHE_DIR = None


def viewer_mesh(path: str) -> str | None:
    """A mesh the viewer can read. COLLADA is XML, so it is converted once into a cache directory."""
    global _MESH_CACHE_DIR

    if path.lower().endswith(VIEWER_MESH_FORMATS):
        return path
    stem = os.path.splitext(path)[0]
    for extension in VIEWER_MESH_FORMATS:
        if os.path.isfile(stem + extension):
            return stem + extension

    _MESH_CACHE_DIR = _MESH_CACHE_DIR or tempfile.mkdtemp(prefix="nbv_meshes_")
    target = os.path.join(_MESH_CACHE_DIR, os.path.basename(stem) + ".obj")
    if not os.path.isfile(target):
        import trimesh
        try:
            trimesh.load(path, force="mesh").export(target)
        except Exception:
            return None
    return target


def urdf_visuals(urdf_xml: str) -> list[dict]:
    """Visual meshes and boxes of every link, in the shape the viewer expects."""
    visuals = []
    for link in ET.fromstring(urdf_xml).findall("link"):
        for index, visual in enumerate(link.findall("visual")):
            geometry = visual.find("geometry")
            if geometry is None:
                continue
            origin = visual.find("origin")
            xyz = [float(v) for v in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()]
            rpy = [float(v) for v in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()]
            entry = {
                "name": f"{link.get('name')}_{index}",
                "link": link.get("name"),
                "offset_position": xyz,
                "offset_quaternion": Rotation.from_euler("xyz", rpy).as_quat().tolist(),
                "color": [200, 200, 205, 255],
                "mesh": None,
                "half_sizes": None,
            }
            mesh = geometry.find("mesh")
            box = geometry.find("box")
            if mesh is not None:
                path = resolve_mesh_path(mesh.get("filename", ""))
                path = viewer_mesh(path) if path else None
                if path is None:
                    continue
                entry["mesh"] = path
                scale = [float(v) for v in mesh.get("scale", "1 1 1").split()]
                entry["scale"] = scale
            elif box is not None:
                entry["half_sizes"] = [float(v) / 2.0 for v in box.get("size", "0 0 0").split()]
            else:
                continue
            visuals.append(entry)
    return visuals
