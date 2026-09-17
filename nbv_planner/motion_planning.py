"""Collision-aware arm motion to a target camera pose via CuRobo MotionGen."""

import os
import time

import numpy as np

from dataclasses import dataclass
from typing import Sequence

from nbv_planner.config import (
    COLLISION_SPHERE_BUFFER_M,
    CUROBO_CONFIGS_DIR,
    FINETUNE_TRAJOPT_FILE,
    GRADIENT_TRAJOPT_FILE,
    MOTION_PLAN_MAX_ATTEMPTS,
    ROBOT_CONFIG_PATH,
    TABLE_CLEARANCE_M,
    URDF_PATH,
)
from nbv_planner.reachability import quaternion_xyzw_to_wxyz, world_poses_to_base_link_frame


@dataclass
class TrajectoryPlan:
    """Collision-free joint trajectory computed by cuRobo."""
    success: bool
    trajectory: np.ndarray | None  # (T, num_joints)
    joint_names: list[str]
    best_candidate_idx: int = -1
    optimization_ms: float = 0.0

_MOTION_GEN = None
_MOTION_GEN_WARMUP_MS = 0.0


def get_motion_gen_warmup_ms() -> float:
    """Monotonic duration of the cuRobo MotionGen warm-up phase in milliseconds."""
    return _MOTION_GEN_WARMUP_MS


def _build_robot_config(tensor_args, collision_sphere_buffer: float = COLLISION_SPHERE_BUFFER_M):
    from curobo.types.robot import RobotConfig
    from curobo.util_file import load_yaml

    cfg = load_yaml(ROBOT_CONFIG_PATH)
    cfg["robot_cfg"]["kinematics"]["urdf_path"] = URDF_PATH
    cfg["robot_cfg"]["kinematics"]["asset_root_path"] = os.path.dirname(URDF_PATH)
    cfg["robot_cfg"]["kinematics"]["collision_sphere_buffer"] = float(collision_sphere_buffer)
    return RobotConfig.from_dict(cfg, tensor_args=tensor_args)


def warmup_motion_gen(world_config=None, collision_sphere_buffer: float = COLLISION_SPHERE_BUFFER_M, reporter=None):
    """Explicitly warm up cuRobo MotionGen once and record monotonic duration."""
    global _MOTION_GEN, _MOTION_GEN_WARMUP_MS
    if _MOTION_GEN is not None:
        return _MOTION_GEN
    from curobo.geom.sdf.world import CollisionCheckerType
    from curobo.types.base import TensorDeviceType
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

    tensor_args = TensorDeviceType()
    cfg = MotionGenConfig.load_from_robot_config(
        _build_robot_config(tensor_args, collision_sphere_buffer=collision_sphere_buffer),
        world_config,
        tensor_args,
        interpolation_dt=0.04,
        trajopt_tsteps=32,
        use_cuda_graph=False,
        gradient_trajopt_file=GRADIENT_TRAJOPT_FILE,
        finetune_trajopt_file=FINETUNE_TRAJOPT_FILE,
        collision_checker_type=CollisionCheckerType.PRIMITIVE,
    )
    mg = MotionGen(cfg)
    t_w0 = time.perf_counter()
    if reporter is not None:
        reporter.start_stage("warmup")
    try:
        mg.warmup(enable_graph=False, warmup_js_trajopt=False)
    finally:
        if reporter is not None:
            reporter.end_stage("warmup")
    _MOTION_GEN_WARMUP_MS = (time.perf_counter() - t_w0) * 1000.0
    _MOTION_GEN = mg
    return _MOTION_GEN


def get_motion_gen(world_config=None, collision_sphere_buffer: float = COLLISION_SPHERE_BUFFER_M, reporter=None):
    """Build or update MotionGen once and reuse across the process."""
    global _MOTION_GEN
    if _MOTION_GEN is None:
        warmup_motion_gen(world_config=world_config, collision_sphere_buffer=collision_sphere_buffer, reporter=reporter)
    elif world_config is not None:
        _MOTION_GEN.update_world(world_config)
    return _MOTION_GEN




def build_world_config(
    t_base_world: np.ndarray,
    q_base_world_xyzw: np.ndarray,
    table_lo: np.ndarray | None = None,
    table_hi: np.ndarray | None = None,
    t_obj_world: np.ndarray | None = None,
    q_obj_world_xyzw: np.ndarray | None = None,
    obj_dims: np.ndarray | None = None,
    table_center_world: np.ndarray | None = None,
    table_dims: Sequence[float] | None = None,
    q_table_world_xyzw: np.ndarray | None = None,
):
    """Construct cuRobo collision cuboids for table and object in base_link frame.

    Pure mathematical function: zero simulator dependencies.
    """
    from curobo.geom.types import Cuboid, WorldConfig

    cuboids = []

    # 1. Table obstacle
    if table_lo is not None and table_hi is not None:
        t_table_w = (np.asarray(table_lo) + np.asarray(table_hi)) / 2.0
        dims_table = (np.asarray(table_hi) - np.asarray(table_lo)).tolist()
    elif table_center_world is not None and table_dims is not None:
        t_table_w = np.asarray(table_center_world, dtype=float).copy()
        dims_table = list(table_dims)
        if q_table_world_xyzw is None:
            t_table_w[2] -= TABLE_CLEARANCE_M
    else:
        t_table_w = None
        dims_table = None

    if t_table_w is not None and dims_table is not None:
        q_tab = q_table_world_xyzw if q_table_world_xyzw is not None else np.array([0.0, 0.0, 0.0, 1.0])
        t_table_base, q_table_base_xyzw = world_poses_to_base_link_frame(
            t_table_w[None, :],
            np.asarray(q_tab)[None, :],
            t_base_world,
            q_base_world_xyzw,
        )
        table = Cuboid(
            name="table",
            pose=[*t_table_base[0].tolist(), *quaternion_xyzw_to_wxyz(q_table_base_xyzw)[0].tolist()],
            dims=dims_table,
        )
        cuboids.append(table)

    # 2. Target object obstacle
    if t_obj_world is not None and obj_dims is not None:
        q_obj_w = q_obj_world_xyzw if q_obj_world_xyzw is not None else np.array([0.0, 0.0, 0.0, 1.0])
        t_obj_base, q_obj_base_xyzw = world_poses_to_base_link_frame(
            np.asarray(t_obj_world)[None, :],
            np.asarray(q_obj_w)[None, :],
            t_base_world,
            q_base_world_xyzw,
        )
        obj = Cuboid(
            name="scanned_object",
            pose=[*t_obj_base[0].tolist(), *quaternion_xyzw_to_wxyz(q_obj_base_xyzw)[0].tolist()],
            dims=(np.asarray(obj_dims) * 1.15).tolist(),
        )
        cuboids.append(obj)

    return WorldConfig(cuboid=cuboids)




def plan_motion_batch(
    t_targets_world: np.ndarray,
    q_targets_world: np.ndarray,
    current_joints: np.ndarray,
    t_base_world: np.ndarray,
    q_base_world_xyzw: np.ndarray,
    world_config=None,
    arm_joint_names: Sequence[str] = (
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ),
    max_attempts: int = MOTION_PLAN_MAX_ATTEMPTS,
    enable_graph: bool = False,
) -> tuple[bool, int, np.ndarray | None, float]:
    """Plan collision-free trajectories on GPU in parallel for a batch of candidate camera poses.

    Pure mathematical and GPU function: zero simulator dependencies.
    Returns: (success, best_batch_index, trajectory_numpy_array, opt_duration_ms)
    """
    import torch
    from curobo.types.math import Pose
    from curobo.types.robot import JointState
    from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

    B = len(t_targets_world)
    if B == 0:
        return False, -1, None, 0.0

    motion_gen = get_motion_gen(world_config=world_config)

    t_targets_base, q_targets_base_xyzw = world_poses_to_base_link_frame(
        np.asarray(t_targets_world),
        np.asarray(q_targets_world),
        t_base_world,
        q_base_world_xyzw,
    )
    goal = Pose(
        position=motion_gen.tensor_args.to_device(
            np.ascontiguousarray(t_targets_base, dtype=np.float32)
        ),
        quaternion=motion_gen.tensor_args.to_device(
            np.ascontiguousarray(quaternion_xyzw_to_wxyz(q_targets_base_xyzw), dtype=np.float32)
        ),
    )

    q_start = JointState.from_position(
        motion_gen.tensor_args.to_device(
            np.repeat(np.asarray(current_joints, dtype=np.float32)[None, :], B, axis=0)
        ),
        joint_names=list(arm_joint_names),
    )

    t_opt_0 = time.perf_counter()
    result = motion_gen.plan_batch(
        q_start,
        goal,
        MotionGenPlanConfig(
            max_attempts=max_attempts,
            enable_graph=enable_graph,
            enable_graph_attempt=1 if enable_graph else None,
        ),
    )
    opt_ms = (time.perf_counter() - t_opt_0) * 1000.0

    succ_idx = torch.where(result.success)[0]
    if len(succ_idx) == 0:
        if hasattr(result, "status"):
            print(f"[cuRobo batch opt failed] status: {result.status}")
        return False, -1, None, opt_ms

    best_k = int(succ_idx[0].item())
    plan_obj = result.interpolated_plan
    if isinstance(plan_obj, list):
        traj_item = plan_obj[best_k]
    elif hasattr(plan_obj, "__getitem__") and hasattr(plan_obj, "batch_size") and plan_obj.batch_size > 1:
        traj_item = plan_obj[best_k]
    else:
        traj_item = plan_obj

    if hasattr(traj_item, "trim_trajectory") and hasattr(result, "path_buffer_last_tstep"):
        try:
            last_t = result.path_buffer_last_tstep[best_k]
            traj_item = traj_item.trim_trajectory(0, last_t)
        except Exception:
            pass

    idx = [traj_item.joint_names.index(name) for name in arm_joint_names]
    pos = traj_item.position
    if hasattr(pos, "cpu"):
        pos_np = pos.cpu().numpy()
    else:
        pos_np = np.asarray(pos)

    if pos_np.ndim == 3:
        if pos_np.shape[0] > 1 and best_k < pos_np.shape[0]:
            pos_np = pos_np[best_k]
        else:
            pos_np = pos_np[0]

    trajectory = pos_np[:, idx]

    return True, best_k, trajectory, opt_ms


def plan_motion_single(
    t_target_world: np.ndarray,
    q_target_world: np.ndarray,
    current_joints: np.ndarray,
    t_base_world: np.ndarray,
    q_base_world_xyzw: np.ndarray,
    world_config=None,
    arm_joint_names: Sequence[str] = (
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ),
    max_attempts: int = MOTION_PLAN_MAX_ATTEMPTS,
    enable_graph: bool = False,
) -> tuple[bool, np.ndarray | None, float]:
    """Single pose planning wrapper around plan_motion_batch."""
    ok, _, traj, opt_ms = plan_motion_batch(
        t_targets_world=np.asarray(t_target_world)[None, :],
        q_targets_world=np.asarray(q_target_world)[None, :],
        current_joints=current_joints,
        t_base_world=t_base_world,
        q_base_world_xyzw=q_base_world_xyzw,
        world_config=world_config,
        arm_joint_names=arm_joint_names,
        max_attempts=max_attempts,
        enable_graph=enable_graph,
    )
    return ok, traj, opt_ms


