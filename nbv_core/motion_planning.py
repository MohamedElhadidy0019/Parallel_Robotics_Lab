"""Collision-aware arm motion to a target camera pose via CuRobo MotionGen."""

import os

import numpy as np

from nbv_core.config import (
    CUROBO_CONFIGS_DIR,
    FINETUNE_TRAJOPT_FILE,
    GRADIENT_TRAJOPT_FILE,
    MAX_POSE_ERROR_M,
    ROBOT_CONFIG_PATH,
    TABLE_CLEARANCE_M,
    TABLE_COLLISION_HALF_HEIGHT,
    URDF_PATH,
)
from nbv_core.reachability import quaternion_xyzw_to_wxyz, world_poses_to_base_link_frame

_MOTION_GEN = None


def _build_robot_config(tensor_args):
    from curobo.types.robot import RobotConfig
    from curobo.util_file import load_yaml

    cfg = load_yaml(ROBOT_CONFIG_PATH)
    cfg["robot_cfg"]["kinematics"]["urdf_path"] = URDF_PATH
    cfg["robot_cfg"]["kinematics"]["asset_root_path"] = os.path.dirname(URDF_PATH)
    return RobotConfig.from_dict(cfg, tensor_args=tensor_args)


def _get_motion_gen(env):
    """Build MotionGen once and reuse across the process."""
    global _MOTION_GEN
    if _MOTION_GEN is None:
        from curobo.geom.sdf.world import CollisionCheckerType
        from curobo.types.base import TensorDeviceType
        from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

        tensor_args = TensorDeviceType()
        cfg = MotionGenConfig.load_from_robot_config(
            _build_robot_config(tensor_args),
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

    r = env.max_reach()
    t_table_world = np.array([
        t_base_world[0],
        t_base_world[1],
        env.table_top_z - TABLE_CLEARANCE_M - TABLE_COLLISION_HALF_HEIGHT,
    ])
    t_table_base, q_table_base_xyzw = world_poses_to_base_link_frame(
        t_table_world[None, :],
        np.array([[0.0, 0.0, 0.0, 1.0]]),
        t_base_world,
        q_base_world_xyzw,
    )
    table = Cuboid(
        name="table",
        pose=[*t_table_base[0].tolist(), *quaternion_xyzw_to_wxyz(q_table_base_xyzw)[0].tolist()],
        dims=[2 * r, 2 * r, 2 * TABLE_COLLISION_HALF_HEIGHT],
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


def move_camera_to(env, t_target_world: np.ndarray, q_target_world) -> tuple[bool, np.ndarray]:
    """Plan a collision-free trajectory to a camera pose and drive the arm along it.

    Returns (reached, t_achieved_world).
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

    result = motion_gen.plan_single(q_start, goal, MotionGenPlanConfig(max_attempts=5))
    if not bool(result.success.item()):
        t_now = np.array(
            env._p.getLinkState(env.robot_id, env.camera_link, physicsClientId=env.client_id)[0]
        )
        return False, t_now

    plan = result.get_interpolated_plan()
    idx = [plan.joint_names.index(name) for name in env.arm_joint_names]
    trajectory = plan.position[:, idx].cpu().numpy()

    for step in trajectory:
        env.execute_joint_states(step.tolist(), absolute=True)

    env._wait_for_arm_at_rest()
    env._snap_to_joint_targets(trajectory[-1].tolist())

    t_achieved = np.array(
        env._p.getLinkState(env.robot_id, env.camera_link, physicsClientId=env.client_id)[0]
    )
    err = float(np.linalg.norm(t_achieved - np.asarray(t_target_world)))
    return err <= MAX_POSE_ERROR_M, t_achieved
