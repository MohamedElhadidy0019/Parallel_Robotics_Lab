"""Collision-aware arm motion to a target camera pose via CuRobo MotionGen."""

import os
import time

import numpy as np

from nbv_core.config import (
    COLLISION_SPHERE_BUFFER_M,
    CUROBO_CONFIGS_DIR,
    FINETUNE_TRAJOPT_FILE,
    GRADIENT_TRAJOPT_FILE,
    MAX_POSE_ERROR_M,
    MOTION_PLAN_MAX_ATTEMPTS,
    ROBOT_CONFIG_PATH,
    TABLE_CLEARANCE_M,
    TABLE_COLLISION_HALF_HEIGHT,
    URDF_PATH,
)
from nbv_core.reachability import quaternion_xyzw_to_wxyz, world_poses_to_base_link_frame

_MOTION_GEN = None


def _build_robot_config(tensor_args, collision_sphere_buffer: float = COLLISION_SPHERE_BUFFER_M):
    from curobo.types.robot import RobotConfig
    from curobo.util_file import load_yaml

    cfg = load_yaml(ROBOT_CONFIG_PATH)
    cfg["robot_cfg"]["kinematics"]["urdf_path"] = URDF_PATH
    cfg["robot_cfg"]["kinematics"]["asset_root_path"] = os.path.dirname(URDF_PATH)
    cfg["robot_cfg"]["kinematics"]["collision_sphere_buffer"] = float(collision_sphere_buffer)
    return RobotConfig.from_dict(cfg, tensor_args=tensor_args)


def _get_motion_gen(env):
    """Build MotionGen once and reuse across the process."""
    global _MOTION_GEN
    if _MOTION_GEN is None:
        from curobo.geom.sdf.world import CollisionCheckerType
        from curobo.types.base import TensorDeviceType
        from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

        robot_cfg_obj = getattr(getattr(env, "config", None), "robot", None)
        buf = getattr(robot_cfg_obj, "collision_sphere_buffer", COLLISION_SPHERE_BUFFER_M)

        tensor_args = TensorDeviceType()
        cfg = MotionGenConfig.load_from_robot_config(
            _build_robot_config(tensor_args, collision_sphere_buffer=buf),
            _build_world_config(env),
            tensor_args,
            interpolation_dt=0.02,
            trajopt_tsteps=32,
            use_cuda_graph=False,
            gradient_trajopt_file=GRADIENT_TRAJOPT_FILE,
            finetune_trajopt_file=FINETUNE_TRAJOPT_FILE,
            collision_checker_type=CollisionCheckerType.PRIMITIVE,
        )
        _MOTION_GEN = MotionGen(cfg)
        _MOTION_GEN.warmup()
    return _MOTION_GEN


def _build_world_config(env):
    """Table + object as cuboid obstacles in the base_link frame."""
    from curobo.geom.types import Cuboid, WorldConfig

    t_base_world, q_base_world_xyzw = env.base_pose()

    if hasattr(env, "table_aabb"):
        lo, hi = env.table_aabb
        t_table_world = (lo + hi) / 2.0
        table_dims = (hi - lo).tolist()
    else:
        r = env.max_reach()
        t_table_world = np.array([
            t_base_world[0],
            t_base_world[1],
            env.table_top_z - TABLE_CLEARANCE_M - TABLE_COLLISION_HALF_HEIGHT,
        ])
        table_dims = [2 * r, 2 * r, 2 * TABLE_COLLISION_HALF_HEIGHT]

    t_table_base, q_table_base_xyzw = world_poses_to_base_link_frame(
        t_table_world[None, :],
        np.array([[0.0, 0.0, 0.0, 1.0]]),
        t_base_world,
        q_base_world_xyzw,
    )
    table = Cuboid(
        name="table",
        pose=[*t_table_base[0].tolist(), *quaternion_xyzw_to_wxyz(q_table_base_xyzw)[0].tolist()],
        dims=table_dims,
    )

    t_obj_world, q_obj_world_xyzw = env._p.getBasePositionAndOrientation(
        env.obj_id, physicsClientId=env.client_id
    )
    aabb_min, aabb_max = env._p.getAABB(env.obj_id, physicsClientId=env.client_id)
    obj_dims = (np.asarray(aabb_max) - np.asarray(aabb_min)) * 1.15
    t_obj_base, q_obj_base_xyzw = world_poses_to_base_link_frame(
        np.asarray(t_obj_world)[None, :],
        np.asarray(q_obj_world_xyzw)[None, :],
        t_base_world,
        q_base_world_xyzw,
    )
    obj = Cuboid(
        name="scanned_object",
        pose=[*t_obj_base[0].tolist(), *quaternion_xyzw_to_wxyz(q_obj_base_xyzw)[0].tolist()],
        dims=obj_dims.tolist(),
    )
    return WorldConfig(cuboid=[table, obj])


def move_camera_to(
    env, t_target_world: np.ndarray, q_target_world, visualizer=None, return_timing: bool = False
) -> tuple:
    """Plan a collision-free trajectory to a camera pose and drive the arm along it.

    If return_timing is False, returns (reached, t_achieved_world).
    If return_timing is True, returns (reached, t_achieved_world, opt_ms, exec_ms).
    """
    from curobo.types.math import Pose
    from curobo.types.robot import JointState
    from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

    motion_gen = _get_motion_gen(env)
    motion_gen.update_world(_build_world_config(env))

    t_base_world, q_base_world_xyzw = env.base_pose()
    t_target_base, q_target_base_xyzw = world_poses_to_base_link_frame(
        np.asarray(t_target_world)[None, :],
        np.asarray(q_target_world)[None, :],
        t_base_world,
        q_base_world_xyzw,
    )
    goal = Pose(
        position=motion_gen.tensor_args.to_device(
            np.ascontiguousarray(t_target_base, dtype=np.float32)
        ),
        quaternion=motion_gen.tensor_args.to_device(
            np.ascontiguousarray(quaternion_xyzw_to_wxyz(q_target_base_xyzw), dtype=np.float32)
        ),
    )

    live = [env._p.getJointState(env.robot_id, i, physicsClientId=env.client_id)[0]
            for i in env.arm_joint_indices]
    q_start = JointState.from_position(
        motion_gen.tensor_args.to_device(np.asarray(live, dtype=np.float32)[None, :]),
        joint_names=env.arm_joint_names,
    )

    t_opt_0 = time.perf_counter()
    result = motion_gen.plan_single(q_start, goal, MotionGenPlanConfig(max_attempts=MOTION_PLAN_MAX_ATTEMPTS))
    opt_ms = (time.perf_counter() - t_opt_0) * 1000.0

    if not bool(result.success.item()):
        t_now = np.array(
            env._p.getLinkState(env.robot_id, env.camera_link, physicsClientId=env.client_id)[0]
        )
        if return_timing:
            return False, t_now, opt_ms, 0.0
        return False, t_now

    plan = result.get_interpolated_plan()
    idx = [plan.joint_names.index(name) for name in env.arm_joint_names]
    trajectory = plan.position[:, idx].cpu().numpy()

    t_exec_0 = time.perf_counter()
    for step in trajectory:
        env.execute_joint_states(step.tolist(), absolute=True)
        if getattr(env, "render", False):
            time.sleep(1.0 / 120.0)
        if visualizer is not None and getattr(visualizer, "enabled", False):
            visualizer.update_robot_pose(env)
            if not getattr(env, "render", False):
                time.sleep(1.0 / 120.0)

    env._wait_for_arm_at_rest()
    env._snap_to_joint_targets(trajectory[-1].tolist())
    if visualizer is not None and getattr(visualizer, "enabled", False):
        visualizer.update_robot_pose(env)
    exec_ms = (time.perf_counter() - t_exec_0) * 1000.0

    t_achieved = np.array(
        env._p.getLinkState(env.robot_id, env.camera_link, physicsClientId=env.client_id)[0]
    )
    err = float(np.linalg.norm(t_achieved - np.asarray(t_target_world)))
    reached = err <= MAX_POSE_ERROR_M
    if return_timing:
        return reached, t_achieved, opt_ms, exec_ms
    return reached, t_achieved


def move_camera_to_batch(
    env, t_targets_world: np.ndarray, q_targets_world: np.ndarray, visualizer=None, return_timing: bool = False
) -> tuple:
    """Plan collision-free trajectories for a batch of candidate camera poses on GPU.

    Optimizes paths for all candidates in parallel, executing the first successful path.
    Returns: (reached, t_achieved, best_batch_idx, opt_ms, exec_ms) if return_timing else (reached, t_achieved, best_batch_idx)
    """
    import torch
    from curobo.types.math import Pose
    from curobo.types.robot import JointState
    from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

    B = len(t_targets_world)
    if B == 0:
        return (False, None, -1, 0.0, 0.0) if return_timing else (False, None, -1)

    motion_gen = _get_motion_gen(env)
    motion_gen.update_world(_build_world_config(env))

    t_base_world, q_base_world_xyzw = env.base_pose()
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

    live = [env._p.getJointState(env.robot_id, i, physicsClientId=env.client_id)[0]
            for i in env.arm_joint_indices]
    q_start = JointState.from_position(
        motion_gen.tensor_args.to_device(np.repeat(np.asarray(live, dtype=np.float32)[None, :], B, axis=0)),
        joint_names=env.arm_joint_names,
    )

    t_opt_0 = time.perf_counter()
    result = motion_gen.plan_batch(
        q_start, goal, MotionGenPlanConfig(max_attempts=MOTION_PLAN_MAX_ATTEMPTS, enable_graph=False, enable_graph_attempt=None)
    )
    opt_ms = (time.perf_counter() - t_opt_0) * 1000.0

    succ_idx = torch.where(result.success)[0]
    if len(succ_idx) == 0:
        t_now = np.array(
            env._p.getLinkState(env.robot_id, env.camera_link, physicsClientId=env.client_id)[0]
        )
        return (False, t_now, -1, opt_ms, 0.0) if return_timing else (False, t_now, -1)

    best_k = int(succ_idx[0].item())
    traj_item = result.interpolated_plan[best_k].trim_trajectory(0, result.path_buffer_last_tstep[best_k])
    idx = [traj_item.joint_names.index(name) for name in env.arm_joint_names]
    trajectory = traj_item.position[:, idx].cpu().numpy()

    t_exec_0 = time.perf_counter()
    for step in trajectory:
        env.execute_joint_states(step.tolist(), absolute=True)
        if getattr(env, "render", False):
            time.sleep(1.0 / 120.0)
        if visualizer is not None and getattr(visualizer, "enabled", False):
            visualizer.update_robot_pose(env)
            if not getattr(env, "render", False):
                time.sleep(1.0 / 120.0)

    env._wait_for_arm_at_rest()
    env._snap_to_joint_targets(trajectory[-1].tolist())
    if visualizer is not None and getattr(visualizer, "enabled", False):
        visualizer.update_robot_pose(env)
    exec_ms = (time.perf_counter() - t_exec_0) * 1000.0

    t_achieved = np.array(
        env._p.getLinkState(env.robot_id, env.camera_link, physicsClientId=env.client_id)[0]
    )
    err = float(np.linalg.norm(t_achieved - np.asarray(t_targets_world[best_k])))
    reached = err <= MAX_POSE_ERROR_M
    if return_timing:
        return reached, t_achieved, best_k, opt_ms, exec_ms
    return reached, t_achieved, best_k
