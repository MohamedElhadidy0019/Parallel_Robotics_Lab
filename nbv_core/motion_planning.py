"""
Collision-aware arm motion via CuRobo's MotionGen - replaces move_camera_to's naive,
collision-blind PyBullet-IK glide. All IK/trajectory-planning/collision-checking logic here is
CuRobo's own solver, not hand-written - this module only wires: robot config (collision
spheres - see curobo_configs/ur5_robotiq_85_camera.yml), world obstacles (table + object, from
live PyBullet state), and executing the returned trajectory through PyBullet position control.

Why this exists: with move_camera_to's old glide, the arm regularly passed straight through
itself and (once the object was repositioned closer to the base for full-orbit reachability -
see project memory) through the scanned object itself, knocking it 100-300mm out of place
mid-scan. MotionGen plans an explicitly collision-free trajectory against both the robot's own
geometry (self-collision) and registered world obstacles (object + table).
"""
import os

import numpy as np
import torch
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util_file import load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

from nbv_core.reachability import quaternion_xyzw_to_wxyz, world_poses_to_base_link_frame

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF_PATH = os.path.join(
    PROJECT_ROOT, "third_party/shelf_gym_repo/shelf_gym/meshes/urdf/ur5_robotiq_85.urdf"
)
CUROBO_CONFIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "curobo_configs")
ROBOT_CONFIG_PATH = os.path.join(CUROBO_CONFIGS_DIR, "ur5_robotiq_85_camera.yml")
# Patched copies of CuRobo's own gradient_trajopt.yml/finetune_trajopt.yml with every
# lbfgs.use_cuda_*_kernel flag forced off - stock configs crash with "CUDA error: invalid
# argument" inside curobolib's lbfgs_step_cu kernel on this GTX 1060/Pascal build (first hit
# the moment this project used MotionGen/trajopt - IKSolver, used everywhere else, never
# exercises this kernel). Disabling cuda_graph alone did not fix it; only turning off the
# custom CUDA kernels themselves (falling back to CuRobo's own plain-PyTorch/JIT LBFGS
# implementation, same file, same algorithm) did. See project memory for the full diagnosis.
GRADIENT_TRAJOPT_FILE = os.path.join(CUROBO_CONFIGS_DIR, "gradient_trajopt.yml")
FINETUNE_TRAJOPT_FILE = os.path.join(CUROBO_CONFIGS_DIR, "finetune_trajopt.yml")

MAX_POSE_ERROR_M = 0.015

_MOTION_GEN: MotionGen | None = None


def _build_robot_config(tensor_args: TensorDeviceType) -> RobotConfig:
    cfg_dict = load_yaml(ROBOT_CONFIG_PATH)
    cfg_dict["robot_cfg"]["kinematics"]["urdf_path"] = URDF_PATH
    cfg_dict["robot_cfg"]["kinematics"]["asset_root_path"] = os.path.dirname(URDF_PATH)
    return RobotConfig.from_dict(cfg_dict, tensor_args=tensor_args)


def _get_motion_gen(env) -> MotionGen:
    """Built once (expensive: CUDA graph warmup) and reused for every move_camera_to call."""
    global _MOTION_GEN
    if _MOTION_GEN is None:
        tensor_args = TensorDeviceType()
        robot_cfg = _build_robot_config(tensor_args)
        motion_gen_cfg = MotionGenConfig.load_from_robot_config(
            robot_cfg,
            _build_world_config(env),  # an initial world is required to initialize world_coll_checker at
                                        # all - update_world() (called on every move_camera_to call to
                                        # refresh obstacle poses) only REPLACES an already-initialized one
            tensor_args,
            interpolation_dt=0.02,
            trajopt_tsteps=32,
            use_cuda_graph=False,
            gradient_trajopt_file=GRADIENT_TRAJOPT_FILE,
            finetune_trajopt_file=FINETUNE_TRAJOPT_FILE,
            collision_checker_type=CollisionCheckerType.PRIMITIVE,  # our world is just cuboids (table +
                                                                     # object bounding box) - the default
                                                                     # MESH checker pulls in NVIDIA's `warp`
                                                                     # library's torch interop, which isn't
                                                                     # working in this env (AttributeError:
                                                                     # module 'warp' has no attribute
                                                                     # 'torch'); PRIMITIVE avoids that
                                                                     # dependency entirely and is the right
                                                                     # checker type for primitive obstacles.
        )
        motion_gen = MotionGen(motion_gen_cfg)
        motion_gen.warmup()
        _MOTION_GEN = motion_gen
    return _MOTION_GEN


TABLE_COLLISION_TOP_WORLD_Z = 0.78  # see _build_world_config's table-clearance note
TABLE_COLLISION_HALF_HEIGHT = 0.15


def _build_world_config(env) -> WorldConfig:
    """
    Table + object as box obstacles, in the robot's base_link frame (MotionGen, like the
    IKSolver in reachability.py, plans relative to base_link - see
    world_poses_to_base_link_frame's docstring for why). Table footprint (x/y half-extents
    [0.50, 0.65] centered at [0.0, 0.15]) matches nbv_environment.NBVEnv2._build_scene()'s real
    tabletop slab - not read back live since it's static and PyBullet doesn't expose
    half-extents via a query API for a createCollisionShape body.

    The collision model's table TOP is deliberately lower than the real table surface
    (world z=0.90) - the robot base sits flush with the real table (base_link origin is also at
    z=0.90), so a table modeled at true height always overlaps the shoulder_link collision
    sphere (radius 0.1m, centered right at the base) even at the arm's resting pose, making
    every plan fail with INVALID_START_STATE_WORLD_COLLISION before it even starts (confirmed
    via motion_gen.check_start_state - see project memory). Real UR5 mounts have an analogous
    small clearance between the mounting flange and the shoulder joint's actual rotation
    center. Dropping the collision top to z=0.78 (12cm clearance, covering the shoulder
    sphere's radius + margin) fixes the false collision while all real scan candidates in this
    project sit at z>=1.0 (>20cm above this line) - collision protection for the arm swinging
    toward table height elsewhere in reach space is unaffected.
    """
    t_base_world, q_base_world_xyzw = env._p.getBasePositionAndOrientation(env.robot_id)
    t_base_world = np.array(t_base_world)
    q_base_world_xyzw = np.array(q_base_world_xyzw)

    t_table_world = np.array([0.0, 0.15, TABLE_COLLISION_TOP_WORLD_Z - TABLE_COLLISION_HALF_HEIGHT])
    t_table_base, q_table_base_xyzw = world_poses_to_base_link_frame(
        t_table_world[None, :], np.array([[0.0, 0.0, 0.0, 1.0]]), t_base_world, q_base_world_xyzw,
    )
    q_table_base_wxyz = quaternion_xyzw_to_wxyz(q_table_base_xyzw)[0]
    table = Cuboid(
        name="table",
        pose=[*t_table_base[0].tolist(), *q_table_base_wxyz.tolist()],
        dims=[1.00, 1.30, TABLE_COLLISION_HALF_HEIGHT * 2],
    )

    t_obj_world, q_obj_world_xyzw = env._p.getBasePositionAndOrientation(env.obj_id)
    t_obj_world = np.array(t_obj_world)
    q_obj_world_xyzw = np.array(q_obj_world_xyzw)
    aabb_min, aabb_max = env._p.getAABB(env.obj_id)
    obj_dims = (np.array(aabb_max) - np.array(aabb_min)) * 1.15  # small safety margin

    t_obj_base, q_obj_base_xyzw = world_poses_to_base_link_frame(
        t_obj_world[None, :], q_obj_world_xyzw[None, :], t_base_world, q_base_world_xyzw,
    )
    q_obj_base_wxyz = quaternion_xyzw_to_wxyz(q_obj_base_xyzw)[0]
    obj = Cuboid(
        name="scanned_object",
        pose=[*t_obj_base[0].tolist(), *q_obj_base_wxyz.tolist()],
        dims=obj_dims.tolist(),
    )

    return WorldConfig(cuboid=[table, obj])


def move_camera_to(
    env, t_target_world: np.ndarray, q_target_world: list[float],
) -> tuple[bool, np.ndarray]:
    """
    Collision-aware replacement for full_pipeline.py's original move_camera_to - same
    (env, t_target_world, q_target_world) -> (reached, t_achieved_world) contract, so nothing
    else in full_pipeline.py needs to change. Plans a self-collision- and
    object/table-collision-aware trajectory via CuRobo's MotionGen, then executes it through
    PyBullet position control one interpolated waypoint at a time.
    """
    motion_gen = _get_motion_gen(env)
    motion_gen.update_world(_build_world_config(env))

    tensor_args = motion_gen.tensor_args
    t_base_world, q_base_world_xyzw = env._p.getBasePositionAndOrientation(env.robot_id)
    t_target_base, q_target_base_xyzw = world_poses_to_base_link_frame(
        np.asarray(t_target_world)[None, :], np.asarray(q_target_world)[None, :],
        np.array(t_base_world), np.array(q_base_world_xyzw),
    )
    q_target_base_wxyz = quaternion_xyzw_to_wxyz(q_target_base_xyzw)[0]
    goal_pose = Pose(
        position=tensor_args.to_device(t_target_base.astype(np.float32)),
        quaternion=tensor_args.to_device(q_target_base_wxyz[None, :].astype(np.float32)),
    )

    arm_joint_names = [
        env._p.getJointInfo(env.robot_id, idx)[1].decode("utf-8") for idx in env.arm_joint_indices
    ]
    live_positions = [env._p.getJointState(env.robot_id, idx)[0] for idx in env.arm_joint_indices]
    q_start = JointState.from_position(
        tensor_args.to_device(np.array(live_positions, dtype=np.float32)[None, :]),
        joint_names=arm_joint_names,
    )

    result = motion_gen.plan_single(q_start, goal_pose, MotionGenPlanConfig(max_attempts=3))
    if not bool(result.success.item()):
        t_current_world = np.array(env._p.getLinkState(env.robot_id, env.camera_link)[0])
        return False, t_current_world

    plan = result.get_interpolated_plan()
    plan_joint_names = plan.joint_names
    arm_idx_in_plan = [plan_joint_names.index(name) for name in arm_joint_names]
    trajectory = plan.position[:, arm_idx_in_plan].cpu().numpy()

    for step in trajectory:
        env.execute_joint_states(step.tolist(), absolute=True)

    env._wait_for_arm_at_rest()
    env._snap_to_joint_targets(trajectory[-1].tolist())

    t_achieved_world = np.array(env._p.getLinkState(env.robot_id, env.camera_link)[0])
    pose_error_m = float(np.linalg.norm(t_achieved_world - np.asarray(t_target_world)))
    return pose_error_m <= MAX_POSE_ERROR_M, t_achieved_world
