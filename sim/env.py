"""Simulation environment for Steve mobile manipulator, inspection table, and YCB objects."""

import os
from typing import Any, Sequence

import contextlib
import numpy as np

def silence_c_output():
    """Route low-level C stdout and stderr to /dev/null while keeping Python sys.stdout / sys.stderr intact."""
    import io
    import sys

    if getattr(sys, "_c_output_silenced", False):
        return
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        real_out = os.dup(1)
        real_err = os.dup(2)
        sys.stdout = io.TextIOWrapper(open(real_out, "wb", buffering=0), encoding="utf-8", write_through=True)
        sys.stderr = io.TextIOWrapper(open(real_err, "wb", buffering=0), encoding="utf-8", write_through=True)
        null_fd = os.open(os.devnull, os.O_RDWR)
        os.dup2(null_fd, 1)
        os.dup2(null_fd, 2)
        os.close(null_fd)
        sys._c_output_silenced = True
    except Exception:
        pass


silence_c_output()


@contextlib.contextmanager
def _suppress_c_output():
    """Silence low-level C stdout and stderr (context manager wrapper)."""
    silence_c_output()
    yield


import pybullet as p
import pybullet_data

from nbv_core.camera import CameraIntrinsics, capture_rgbd
from nbv_core.config import (
    DEFAULT_YCB_OBJECT,
    ORBIT_DEPTH_FRACTION,
    WORLD_UP_Z,
    YCB_ROOT,
)
from sim.config import SimConfig, TableConfig, SteveRobotConfig
from sim.table import Table


def ycb_urdf(name: str) -> str:
    """Path to a vendored YCB object's URDF."""
    return os.path.join(YCB_ROOT, name, "model.urdf")


def ycb_names() -> list[str]:
    """Every vendored object that ships a URDF."""
    return sorted(d for d in os.listdir(YCB_ROOT) if os.path.isfile(ycb_urdf(d)))


class SteveSimEnv:
    """Simulation environment with Steve mobile manipulator, table, and target object."""

    def __init__(
        self,
        render: bool = True,
        intrinsics: CameraIntrinsics | None = None,
        ycb_object: str = DEFAULT_YCB_OBJECT,
        config: SimConfig | None = None,
    ) -> None:
        self.config = config or SimConfig()
        self.intrinsics = intrinsics or self.config.intrinsics
        self.ycb_object = ycb_object
        self.render = render

        # Connect to PyBullet quietly
        connection_mode = p.GUI if render else p.DIRECT
        with _suppress_c_output():
            self.client_id = p.connect(connection_mode)
        self._p = p

        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._p.setGravity(0, 0, self.config.gravity)
        self._p.setTimeStep(1.0 / self.config.hz)

        # Build world
        self.plane_id = self._p.loadURDF("plane.urdf")
        self._load_robot()
        self.table = Table(self._p, self.config.table)
        self._place_object()

    def _load_robot(self) -> None:
        """Load Steve URDF and configure joint indices."""
        urdf_path = os.path.join(os.path.dirname(__file__), "models", "steve.urdf")
        base_pos = list(self.config.robot.base_pos)
        base_orn = self._p.getQuaternionFromEuler(list(self.config.robot.base_orn_euler))

        self.robot_id = self._p.loadURDF(
            urdf_path,
            basePosition=base_pos,
            baseOrientation=base_orn,
            useFixedBase=True,
            flags=self._p.URDF_USE_INERTIA_FROM_FILE,
        )

        # Index joints and links
        num_joints = self._p.getNumJoints(self.robot_id)
        self.joint_name_to_id: dict[str, int] = {}
        self.link_name_to_id: dict[str, int] = {}
        self.joints: dict[str, tuple[int, Any]] = {}

        for i in range(num_joints):
            info = self._p.getJointInfo(self.robot_id, i)
            j_name = info[1].decode("utf-8")
            l_name = info[12].decode("utf-8")
            self.joint_name_to_id[j_name] = i
            self.link_name_to_id[l_name] = i
            self.joints[j_name] = (i, info)

        self.arm_joint_names = list(self.config.robot.arm_joint_names)
        self.arm_joint_indices = [self.joint_name_to_id[n] for n in self.arm_joint_names]

        self.camera_link = self.link_name_to_id.get(
            self.config.robot.camera_link_name,
            self.link_name_to_id.get("tool0", 0),
        )
        self.tool_tip_id = self.link_name_to_id.get("tool0", self.camera_link)

        # Reset arm to ready pose
        self.reset_robot(self.config.robot.initial_arm_joints)

    def reset_robot(self, joint_values: Sequence[float]) -> None:
        """Reset arm joints to specified angles."""
        for idx, val in zip(self.arm_joint_indices, joint_values):
            self._p.resetJointState(self.robot_id, idx, float(val))

    @property
    def table_surface_z(self) -> float:
        """Exact physical top surface elevation of the table."""
        return self.table.surface_z

    @property
    def table_top_z(self) -> float:
        """Backwards-compatible alias for table_surface_z."""
        return self.table.surface_z

    @property
    def table_aabb(self) -> tuple[np.ndarray, np.ndarray]:
        """Exact physical 3D bounding box (lo, hi) of the table."""
        return self.table.aabb

    @property
    def table_center(self) -> np.ndarray:
        """Midpoint (x, y) of the table."""
        return self.table.center_xy

    @property
    def table_id(self) -> int:
        """PyBullet body ID of the table."""
        return self.table.table_id

    def base_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Current world pose (pos, orn_xyzw) of Steve's base link."""
        pos, orn = self._p.getBasePositionAndOrientation(self.robot_id)
        return np.asarray(pos, dtype=float), np.asarray(orn, dtype=float)

    def arm_base_pos(self) -> np.ndarray:
        """World position of the UR5 arm base mount."""
        ur5_base_id = self.link_name_to_id.get("ur5_base_link")
        if ur5_base_id is not None:
            st = self._p.getLinkState(self.robot_id, ur5_base_id)
            return np.asarray(st[0], dtype=float)
        return self.base_pose()[0] + np.array([0.158, 0.0, 0.766])

    def max_reach(self) -> float:
        """Nominal reach of the UR5 arm."""
        return 0.85

    def object_size(self) -> np.ndarray:
        """Dimensions of the object bounding box."""
        lo, hi = self._p.getAABB(self.obj_id)
        return np.asarray(hi) - np.asarray(lo)

    def object_radius(self) -> float:
        """Radius of bounding sphere for the target object."""
        return float(np.linalg.norm(self.object_size()) / 2.0)

    def framing_distance(self) -> float:
        """Distance from object center required to frame the object."""
        return float(self.object_size().max()) / (2 * np.tan(np.radians(self.intrinsics.fov) / 2))

    def safe_orbit_radius(self) -> float:
        """Closest safe distance for candidate camera viewpoints."""
        return max(self.object_radius() + self.intrinsics.near, self.framing_distance())

    def orbit_shell(self) -> tuple[float, float]:
        """Inner and outer radius for candidate camera viewpoints around target."""
        r_min = self.safe_orbit_radius()
        d = float(np.linalg.norm(self.obj_pos[:2] - self.arm_base_pos()[:2]))
        slack = max(0.0, self.max_reach() - d - r_min)
        return r_min, r_min + slack * ORBIT_DEPTH_FRACTION

    def _place_object(self) -> None:
        """Spawn the YCB object flush on the table surface without dropping."""
        target_center_xy = self.table_center + np.asarray(self.config.placement_offset_from_center)
        temp_spawn_pos = [target_center_xy[0], target_center_xy[1], self.table_surface_z + 0.05]

        self.obj_id = self._p.loadURDF(
            ycb_urdf(self.ycb_object),
            basePosition=temp_spawn_pos,
            useFixedBase=False,
        )

        # Query object bottom z to place flush on table surface
        lo, hi = self._p.getAABB(self.obj_id)
        pos, orn = self._p.getBasePositionAndOrientation(self.obj_id)
        flush_z = pos[2] + (self.table_surface_z - lo[2])

        self._p.resetBasePositionAndOrientation(
            self.obj_id,
            [target_center_xy[0], target_center_xy[1], flush_z],
            orn,
        )

        # Settle physics briefly to verify static stability
        for _ in range(self.config.settle_steps):
            self._p.stepSimulation()
            v, w = self._p.getBaseVelocity(self.obj_id)
            if max(np.abs(v).max(), np.abs(w).max()) < self.config.settle_velocity_tol:
                break

        lo_settled, hi_settled = self._p.getAABB(self.obj_id)
        self.obj_pos = (np.asarray(lo_settled) + np.asarray(hi_settled)) / 2.0

    def capture_frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Capture RGB-D frame and camera pose from the eye-in-hand sensor."""
        link_state = self._p.getLinkState(self.robot_id, self.camera_link)
        t_cam = np.asarray(link_state[0], dtype=float)

        rgb, depth_m, view_matrix, _ = capture_rgbd(
            t_cam, self.obj_pos, WORLD_UP_Z, self.intrinsics, physics_client_id=self.client_id
        )
        return rgb, depth_m, view_matrix, t_cam

    def get_joint_positions(self) -> np.ndarray:
        """Current joint angles of the UR5 arm."""
        states = [self._p.getJointState(self.robot_id, idx)[0] for idx in self.arm_joint_indices]
        return np.asarray(states, dtype=float)

    def get_current_joint_config(self) -> list[float]:
        """Backwards-compatible joint angles list."""
        return self.get_joint_positions().tolist()

    def execute_joint_states(self, joint_angles: Sequence[float], absolute: bool = True) -> None:
        """Set arm joint states directly (shelf_gym compatibility)."""
        for idx, val in zip(self.arm_joint_indices, joint_angles):
            self._p.resetJointState(self.robot_id, idx, float(val))
        self._p.stepSimulation()

    def _wait_for_arm_at_rest(self, steps: int = 20) -> None:
        """Step physics simulation until arm motion settles."""
        for _ in range(steps):
            self._p.stepSimulation()

    def _snap_to_joint_targets(self, targets: Sequence[float]) -> None:
        """Snap arm joints directly to final target configuration."""
        for idx, val in zip(self.arm_joint_indices, targets):
            self._p.resetJointState(self.robot_id, idx, float(val))
        self._p.stepSimulation()

    def execute_trajectory(
        self,
        joint_positions_list: Sequence[np.ndarray],
        visualizer: Any = None,
    ) -> None:
        """Execute a planned joint trajectory smoothly."""
        for q in joint_positions_list:
            for idx, val in zip(self.arm_joint_indices, q):
                self._p.resetJointState(self.robot_id, idx, float(val))
            self._p.stepSimulation()
            if visualizer is not None:
                visualizer.update_robot_pose(self)

    def close(self) -> None:
        """Disconnect PyBullet session."""
        if self._p.isConnected(self.client_id):
            with _suppress_c_output():
                self._p.disconnect(self.client_id)
