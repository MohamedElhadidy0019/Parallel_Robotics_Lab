"""Simulation environment for Steve mobile manipulator, inspection table, and YCB objects."""

import os
import time
from typing import Any, Sequence

import contextlib
import numpy as np

@contextlib.contextmanager
def _suppress_c_output():
    yield


import pybullet as p
import pybullet_data

from sim.camera import capture_from_pose
from nbv_planner.observations import Observation
from scipy.spatial.transform import Rotation
from nbv_planner.camera import CameraIntrinsics
from nbv_planner.config import (
    DEFAULT_YCB_OBJECT,
    ORBIT_DEPTH_FRACTION,
    WORLD_UP_Z,
    YCB_ROOT,
    ycb_names,
    ycb_urdf,
)
from sim.config import SimConfig, TableConfig, SteveRobotConfig
from sim.table import Table


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

        self.client_id = -1
        self._p = p

        # Connect to PyBullet quietly
        connection_mode = p.GUI if render else p.DIRECT
        with _suppress_c_output():
            cid = self._p.connect(connection_mode)
        if cid is None or cid < 0:
            raise ConnectionError(f"Failed to connect to PyBullet physics server (client_id={cid})")
        self.client_id = cid

        try:
            self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
            self._p.setGravity(0, 0, self.config.gravity)
            self._p.setTimeStep(1.0 / self.config.hz)

            # Build world
            self.plane_id = self._p.loadURDF("plane.urdf")
            self._load_robot()
            self.table = Table(self._p, self.config.table)
            self._place_object()
        except BaseException as orig_exc:
            try:
                self.close()
            except Exception as cleanup_exc:
                orig_exc.cleanup_error = cleanup_exc
                orig_exc.__context__ = cleanup_exc
                if hasattr(orig_exc, "add_note"):
                    try:
                        orig_exc.add_note(f"SteveSimEnv cleanup failed during constructor abort: {cleanup_exc}")
                    except Exception:
                        pass
            raise

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
        self.camera_visual_rgba = [
            s[7] for s in self._p.getVisualShapeData(self.robot_id, physicsClientId=self.client_id)
            if s[1] == self.camera_link
        ]

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

    def current_arm_joints(self) -> np.ndarray:
        """Current joint angles for the 6 UR5 arm joints."""
        return np.array(
            [self._p.getJointState(self.robot_id, i, physicsClientId=self.client_id)[0]
             for i in self.arm_joint_indices],
            dtype=np.float32,
        )

    def object_pose_and_dims(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """World position, orientation (xyzw), and bounding box extents of target object."""
        pos, orn = self._p.getBasePositionAndOrientation(self.obj_id, physicsClientId=self.client_id)
        lo, hi = self._p.getAABB(self.obj_id, physicsClientId=self.client_id)
        dims = np.asarray(hi, dtype=float) - np.asarray(lo, dtype=float)
        return np.asarray(pos, dtype=float), np.asarray(orn, dtype=float), dims

    def camera_world_transform(self) -> np.ndarray:
        state = self._p.getLinkState(self.robot_id, self.camera_link,
                                    computeForwardKinematics=True, physicsClientId=self.client_id)
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_quat(state[5]).as_matrix()
        transform[:3, 3] = state[4]
        return transform

    def camera_world_pos(self) -> np.ndarray:
        return self.camera_world_transform()[:3, 3].copy()

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

    def capture_observation(self) -> Observation:
        transform = self.camera_world_transform()
        # The optical frame sits inside the D435 housing mesh. A real sensor never sees its own
        # housing, and TinyRenderer stalls for minutes rasterizing geometry around the eye.
        for rgba in self.camera_visual_rgba:
            self._p.changeVisualShape(self.robot_id, self.camera_link, rgbaColor=[*rgba[:3], 0.0],
                                      physicsClientId=self.client_id)
        try:
            rgb, depth, _, _ = capture_from_pose(transform, self.intrinsics, self.client_id)
        finally:
            for rgba in self.camera_visual_rgba:
                self._p.changeVisualShape(self.robot_id, self.camera_link, rgbaColor=list(rgba),
                                          physicsClientId=self.client_id)
        return Observation(rgb, depth, self.intrinsics, transform, time.monotonic())

    def capture_frame(self):
        observation = self.capture_observation()
        return observation.rgb, observation.depth_m, observation.view_matrix, observation.camera_position

    def settle_arm(self, timeout_s=2.0, velocity_tolerance=0.01) -> bool:
        deadline = time.monotonic() + timeout_s
        stable = 0
        while time.monotonic() < deadline:
            self._p.stepSimulation(physicsClientId=self.client_id)
            states = self._p.getJointStates(self.robot_id, self.arm_joint_indices, physicsClientId=self.client_id)
            stable = stable + 1 if max(abs(state[1]) for state in states) < velocity_tolerance else 0
            if stable >= 10:
                return True
        return False

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
        joint_positions_list: Sequence[Sequence[float]] | np.ndarray,
        visualizer: Any = None,
    ) -> tuple[np.ndarray, float]:
        """Execute a planned joint trajectory smoothly in PyBullet and settle physics.

        Returns:
            (achieved_camera_world_pos, execution_duration_ms)
        """
        t_exec_0 = time.perf_counter()
        traj_arr = np.asarray(joint_positions_list)
        if traj_arr.ndim == 3:
            traj_arr = traj_arr[0]

        for step in traj_arr:
            self.execute_joint_states(step.tolist() if hasattr(step, "tolist") else list(step), absolute=True)
            if getattr(self, "render", False):
                time.sleep(1.0 / 120.0)
            if visualizer is not None and getattr(visualizer, "enabled", False):
                visualizer.update_robot_pose(self)
                if not getattr(self, "render", False):
                    time.sleep(1.0 / 120.0)

        self._wait_for_arm_at_rest()
        last_step = traj_arr[-1]
        self._snap_to_joint_targets(last_step if isinstance(last_step, list) else last_step.tolist())
        if visualizer is not None and getattr(visualizer, "enabled", False):
            visualizer.update_robot_pose(self)
        exec_ms = (time.perf_counter() - t_exec_0) * 1000.0

        return self.camera_world_pos(), exec_ms

    def close(self) -> None:
        """Disconnect PyBullet session."""
        cid = getattr(self, "client_id", -1)
        if cid is not None and cid >= 0:
            pybullet_mod = getattr(self, "_p", p)
            try:
                if not pybullet_mod.isConnected(physicsClientId=cid):
                    self.client_id = -1
                    return
            except Exception:
                pass

            disconnect_succeeded = False
            try:
                with _suppress_c_output():
                    pybullet_mod.disconnect(physicsClientId=cid)
                disconnect_succeeded = True
            finally:
                try:
                    still_connected = pybullet_mod.isConnected(physicsClientId=cid)
                except Exception:
                    still_connected = None

                if still_connected in (False, 0) or (disconnect_succeeded and still_connected is None):
                    self.client_id = -1

            if still_connected in (True, 1):
                raise RuntimeError(f"PyBullet client {cid} remained connected after disconnect attempt")


def move_camera_to(
    env: SteveSimEnv,
    t_target_world: np.ndarray,
    q_target_world: np.ndarray,
    visualizer: Any = None,
    return_timing: bool = False,
) -> tuple:
    """Convenience simulation helper: plan and execute motion to a camera pose."""
    from nbv_planner.config import MAX_POSE_ERROR_M
    from nbv_planner.motion_planning import build_world_config, plan_motion_single

    t_base, q_base = env.base_pose()
    t_obj, q_obj, obj_dims = env.object_pose_and_dims()
    lo, hi = getattr(env, "table_aabb", (None, None))
    world_cfg = build_world_config(
        t_base_world=t_base,
        q_base_world_xyzw=q_base,
        table_lo=lo,
        table_hi=hi,
        t_obj_world=t_obj,
        q_obj_world_xyzw=q_obj,
        obj_dims=obj_dims,
    )
    ok, traj, opt_ms = plan_motion_single(
        t_target_world=np.asarray(t_target_world),
        q_target_world=np.asarray(q_target_world),
        current_joints=env.current_arm_joints(),
        t_base_world=t_base,
        q_base_world_xyzw=q_base,
        world_config=world_cfg,
        arm_joint_names=env.arm_joint_names,
    )
    if not ok or traj is None:
        t_now = env.camera_world_pos()
        if return_timing:
            return False, t_now, opt_ms, 0.0
        return False, t_now

    t_achieved, exec_ms = env.execute_trajectory(traj, visualizer=visualizer)
    err = float(np.linalg.norm(t_achieved - np.asarray(t_target_world)))
    reached = err <= MAX_POSE_ERROR_M
    if return_timing:
        return reached, t_achieved, opt_ms, exec_ms
    return reached, t_achieved
