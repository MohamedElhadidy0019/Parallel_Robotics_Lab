"""What differs between driving Gazebo and driving the physical Steve, in one place.

Both nodes that move the arm import from here, so the safety gate cannot drift between them.
"""

import os
import time

from ament_index_python.packages import get_package_share_directory
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from nbv_planner_ros.robot_model import detect_joint_prefix, urdf_names, wait_for_robot_description

CONFIG_DIR = os.path.join(get_package_share_directory("nbv_planner_ros"), "config")
CONFIG_FOR_TARGET = {
    "sim": os.path.join(CONFIG_DIR, "inspection_config.yaml"),
    "real": os.path.join(CONFIG_DIR, "inspection_config_real.yaml"),
}

# Prefix the live URDF uses, per target. "ur5" is Gazebo and the RViz mock, both harmless to
# drive. "ur5e" is the physical arm. Declaring one and finding the other means the graph is not
# what the operator thinks it is, which is the moment to stop rather than to guess.
PREFIX_FOR_TARGET = {"sim": "ur5", "real": "ur5e"}

# Modes that place a CAD mesh at object_frame. Nothing publishes that frame on the real robot and
# the real object has no CAD model, so only scan mode can run there.
MODES_FOR_TARGET = {"sim": ("cad", "scan", "both"), "real": ("scan",)}


class DryRunStop(Exception):
    """Raised instead of sending the first trajectory when the operator has not confirmed motion."""


def validate_target(target: str) -> str:
    if target not in CONFIG_FOR_TARGET:
        raise RuntimeError(f"target must be one of {sorted(CONFIG_FOR_TARGET)}, got {target!r}")
    return target


def validate_mode(target: str, mode: str) -> None:
    allowed = MODES_FOR_TARGET[target]
    if mode not in allowed:
        raise RuntimeError(
            f"mode:={mode} cannot run on target:={target}; use one of {list(allowed)}. "
            f"cad and both need a CAD mesh at object_frame, which the real robot does not publish.")


def apply_clock(node, target: str) -> None:
    """Sim time in Gazebo, wall time on the robot.

    rclpy's TimeSource declares use_sim_time before any node constructor runs, so a has_parameter
    guard never fires. Setting it re-runs TimeSource's callback, which swaps the clock. Leaving it
    False in Gazebo schedules every goal decades out; leaving it True on the robot, which has no
    /clock, freezes the node at time zero.
    """
    wanted = target == "sim"
    if node.get_parameter("use_sim_time").value != wanted:
        node.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, wanted)])


def latched(depth: int = 1) -> QoSProfile:
    """QoS the UR status topics publish with.

    robot_mode and safety_mode are latched. A default Volatile subscription receives nothing at
    all and reads identically to a robot that is powered down, which would turn a missing
    subscription into a false all-clear.
    """
    return QoSProfile(depth=depth, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                      reliability=ReliabilityPolicy.RELIABLE)


def check_target_matches_graph(node, target: str) -> None:
    """Refuse to run when the declared target is not the robot on the graph.

    The target is declared rather than detected because detection failing towards "sim" only
    wastes a run, while detection failing towards "real" would drive physical hardware with the
    safety gate skipped. Declaring it and then checking the declaration keeps the intent explicit
    and still catches the mistake.
    """
    expected = PREFIX_FOR_TARGET[target]
    actual = detect_joint_prefix(urdf_names(wait_for_robot_description(node))[1])
    if actual != expected:
        other = next((t for t, p in PREFIX_FOR_TARGET.items() if p == actual), None)
        raise RuntimeError(
            f"target:={target} expects joints prefixed {expected!r} but the live robot "
            f"publishes {actual!r}." + (f" This looks like target:={other}." if other else ""))
    node.get_logger().info(f"Graph matches target {target}: joint prefix {actual!r}")


def controller_name(action_name: str) -> str:
    """/scaled_joint_trajectory_controller/follow_joint_trajectory -> scaled_joint_trajectory_controller"""
    return action_name.strip("/").split("/")[0]


def wait_future(future, timeout_sec: float):
    deadline = time.monotonic() + timeout_sec
    while not future.done() and time.monotonic() < deadline:
        time.sleep(0.05)
    return future.result() if future.done() else None


def check_controller_active(node, action_name: str, timeout_sec: float = 10.0) -> None:
    """The trajectory controller must be active, which on a UR needs External Control running.

    The action server exists while the controller is inactive, so without this check the first
    goal is simply rejected and the pipeline burns through every fallback start pose.
    """
    from controller_manager_msgs.srv import ListControllers

    wanted = controller_name(action_name)
    client = node.create_client(ListControllers, "/controller_manager/list_controllers")
    try:
        if not client.wait_for_service(timeout_sec=timeout_sec):
            raise RuntimeError("/controller_manager/list_controllers is not answering. "
                               "Is the arm bringup running and ROS_DOMAIN_ID correct?")
        response = wait_future(client.call_async(ListControllers.Request()), timeout_sec)
        if response is None:
            raise RuntimeError("/controller_manager/list_controllers did not reply")
        states = {c.name: c.state for c in response.controller}
    finally:
        node.destroy_client(client)
    if wanted not in states:
        raise RuntimeError(f"Controller {wanted} is not loaded. Loaded: {sorted(states)}")
    if states[wanted] != "active":
        raise RuntimeError(
            f"Controller {wanted} is {states[wanted]}, need active. Start the External Control "
            f"program on the pendant (press Play and leave it running).")
    node.get_logger().info(f"Controller {wanted} is active")


def check_safety(node, action_name: str) -> None:
    """Hold until the arm reports it is powered, clear and commandable."""
    from ur_dashboard_msgs.msg import RobotMode, SafetyMode

    state = {}
    subscriptions = [
        node.create_subscription(RobotMode, "/io_and_status_controller/robot_mode",
                                 lambda m: state.__setitem__("mode", m.mode), latched()),
        node.create_subscription(SafetyMode, "/io_and_status_controller/safety_mode",
                                 lambda m: state.__setitem__("safety", m.mode), latched()),
    ]
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline and len(state) < 2:
        time.sleep(0.1)
    for subscription in subscriptions:
        node.destroy_subscription(subscription)
    missing = [k for k in ("mode", "safety") if k not in state]
    if missing:
        raise RuntimeError(
            f"No {missing} from /io_and_status_controller after 15 s. Is the arm bringup "
            f"running and ROS_DOMAIN_ID correct? Silence here is not an all-clear.")
    if state["mode"] != RobotMode.RUNNING:
        raise RuntimeError(f"robot_mode is {state['mode']}, need {RobotMode.RUNNING} (RUNNING). "
                           f"Power the arm and clear the safety chain.")
    if state["safety"] != SafetyMode.NORMAL:
        raise RuntimeError(f"safety_mode is {state['safety']}, need {SafetyMode.NORMAL} (NORMAL).")
    check_controller_active(node, action_name)


def prepare_real_motion(node, config: dict, confirm: str) -> bool:
    """Run the safety gate and decide between real motion and a dry run.

    Returns True when the arm may move. Without confirm:=go everything still runs up to the first
    plan, so the plan can be inspected before anything moves. A dry run reports a failed safety
    check instead of stopping on it, since planning needs no power on the arm.
    """
    action = config.get("controller_action", "/scaled_joint_trajectory_controller/follow_joint_trajectory")
    if confirm != "go":
        try:
            check_safety(node, action)
        except RuntimeError as error:
            node.get_logger().warning(f"Would refuse to move: {error}")
        config["dry_run"] = True
        node.get_logger().warning(
            "DRY RUN: planning only, the arm will not move. Re-run with confirm:=go once the plan "
            "looks right and the cell is clear.")
        return False
    check_safety(node, action)
    config["dry_run"] = False
    node.get_logger().warning("Safety checks passed and confirmed; the real arm will move")
    return True
