"""Find an unknown object from the current camera pose and refine its box over a ring of views."""

from typing import Callable

import numpy as np
from scipy.spatial.transform import Rotation

from nbv_planner.detection import Detection, Segmenter, TableBox, detect_object
from nbv_planner.motion_planning import build_world_config
from nbv_planner.object_estimate import ObjectEstimate
from nbv_planner.reachability import ik_filter
from nbv_planner.scan_views import ring_viewpoints

MoveCamera = Callable[[np.ndarray, np.ndarray, object], bool]
OnFrame = Callable[[str, object, Detection, ObjectEstimate], None]


def obstacle_world_config(robot, center: np.ndarray, yaw: float, size: np.ndarray):
    lo, hi = robot.table_aabb
    base_position, base_quaternion = robot.base_pose()
    return build_world_config(
        base_position, base_quaternion, table_lo=lo, table_hi=hi, t_obj_world=center,
        q_obj_world_xyzw=Rotation.from_euler("z", yaw).as_quat(), obj_dims=size,
    )


def box_world_config(robot, box: TableBox, padding: float = 0.02):
    """The unseen far side makes the short side unreliable, so the obstacle uses the long side both ways."""
    long_side = float(box.size[:2].max())
    return obstacle_world_config(robot, box.center, box.yaw, np.array([long_side, long_side, box.size[2]]) + padding)


def scan_object(
    robot,
    segmenter: Segmenter | None,
    move: MoveCamera,
    n_views: int = 8,
    candidates_per_view: int = 3,
    on_frame: OnFrame | None = None,
    on_views: Callable[[np.ndarray], None] | None = None,
    log: Callable[[str], None] = print,
) -> ObjectEstimate:
    """Detect from the current pose, then visit a ring of views around the box and merge each detection."""
    estimate = ObjectEstimate()
    observation = robot.capture_observation()
    detection = detect_object(observation, segmenter)
    estimate.add(detection)
    if on_frame:
        on_frame("start", observation, detection, estimate)

    base_position, base_quaternion = robot.base_pose()
    slots, positions, quaternions = ring_viewpoints(
        estimate.box, robot.intrinsics, observation.camera_position, n_views
    )
    # The robot carries its own model. The packaged constants point at the PyBullet URDF, which
    # does not exist beside a live ROS robot and names links that robot does not have.
    reachable, _ = ik_filter(robot.urdf_path, robot.base_link, robot.ee_link,
                             positions, quaternions, base_position, base_quaternion)
    if on_views:
        on_views(positions[reachable])

    for slot in range(1, n_views + 1):
        label = f"view {slot}"
        world_config = box_world_config(robot, estimate.box)
        candidates = np.flatnonzero((slots == slot) & reachable)[:candidates_per_view]
        if not any(move(positions[i], quaternions[i], world_config) for i in candidates):
            log(f"{label}: skipped, no reachable collision-free pose")
            continue

        observation = robot.capture_observation()
        try:
            detection = detect_object(observation, segmenter, prior=estimate.box)
        except ValueError as error:
            log(f"{label}: skipped, {error}")
            continue
        estimate.add(detection)
        if on_frame:
            on_frame(label, observation, detection, estimate)
    return estimate
