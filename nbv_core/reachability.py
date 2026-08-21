"""Sample camera viewpoints around an object and cache which ones the arm can reach."""

import os
import numpy as np

from nbv_core.config import (
    BASE_LINK,
    CACHE_DIR,
    EE_LINK,
    ELEVATION_DEG,
    IK_NUM_SEEDS,
    IK_POSITION_THRESHOLD_M,
    IK_ROTATION_THRESHOLD_RAD,
    N_AZIMUTH,
    N_RADIUS,
    NEAR_VERTICAL_COSINE,
    WORLD_UP_Y_FALLBACK,
    WORLD_UP_Z,
    XYZW_TO_WXYZ,
)


def cache_path_for(ycb_object: str) -> str:
    """Path to the reachability cache for one object."""
    return os.path.join(CACHE_DIR, f"reachability_{ycb_object}.npz")


def camera_lookat_quaternion_xyzw(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Camera at `eye` pointing at `target`, as an xyzw quaternion (+X right, +Y down, +Z forward)."""
    from scipy.spatial.transform import Rotation

    z_axis = target - eye
    z_axis = z_axis / np.linalg.norm(z_axis)
    world_up = WORLD_UP_Z
    if abs(np.dot(z_axis, world_up)) > NEAR_VERTICAL_COSINE:
        world_up = WORLD_UP_Y_FALLBACK
    x_axis = np.cross(world_up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    return Rotation.from_matrix(np.column_stack([x_axis, y_axis, z_axis])).as_quat()


def quaternion_xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    return np.asarray(q_xyzw)[..., XYZW_TO_WXYZ]


def sample_candidate_camera_poses(
    t_obj_world: np.ndarray,
    radius: tuple[float, float, int],
    elevation_deg: tuple[float, float, int] = ELEVATION_DEG,
    n_azimuth: int = N_AZIMUTH,
    z_min_world: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Viewpoints on a hemisphere shell around the object, each looking at its centre.

    radius / elevation_deg are (min, max, count). z_min_world drops candidates under a
    known obstacle (tabletop). Azimuth sweeps a full 2π.

    Returns (positions (N,3), quaternions xyzw (N,4)).
    """
    radii = np.linspace(radius[0], radius[1], radius[2])
    phis = np.radians(np.linspace(elevation_deg[0], elevation_deg[1], elevation_deg[2]))
    thetas = np.linspace(0.0, 2.0 * np.pi, n_azimuth, endpoint=False)

    r, theta, phi = np.meshgrid(radii, thetas, phis, indexing="ij")
    offsets = np.stack([
        r * np.cos(phi) * np.cos(theta),
        r * np.cos(phi) * np.sin(theta),
        r * np.sin(phi),
    ], axis=-1).reshape(-1, 3)

    t_candidates = offsets + t_obj_world
    if z_min_world is not None:
        t_candidates = t_candidates[t_candidates[:, 2] >= z_min_world]

    q_candidates = np.stack([camera_lookat_quaternion_xyzw(t, t_obj_world) for t in t_candidates])
    return t_candidates, q_candidates


def world_poses_to_base_link_frame(
    t_world: np.ndarray,
    q_world_xyzw: np.ndarray,
    t_base_world: np.ndarray,
    q_base_world_xyzw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-express world-frame poses in the base_link frame that CuRobo solves IK in."""
    from scipy.spatial.transform import Rotation

    R_base = Rotation.from_quat(q_base_world_xyzw)
    R_world = Rotation.from_quat(q_world_xyzw)

    t_base = R_base.inv().apply(t_world - t_base_world)
    q_base = (R_base.inv() * R_world).as_quat()

    return t_base, q_base


def ik_filter(
    urdf_path: str,
    base_link: str,
    ee_link: str,
    t_candidates_world: np.ndarray,
    q_candidates_world_xyzw: np.ndarray,
    t_base_world: np.ndarray,
    q_base_world_xyzw: np.ndarray,
    position_threshold: float = IK_POSITION_THRESHOLD_M,
    rotation_threshold: float = IK_ROTATION_THRESHOLD_RAD,
    num_seeds: int = IK_NUM_SEEDS,
) -> tuple[np.ndarray, np.ndarray]:
    """Which candidates the arm can IK to. Returns (reachable (N,), q_joints (N,6)), zeroed where not."""
    import torch
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.types.robot import RobotConfig
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

    tensor_args = TensorDeviceType()
    robot_cfg = RobotConfig.from_basic(urdf_path, base_link, ee_link, tensor_args)
    ik_solver = IKSolver(
        IKSolverConfig.load_from_robot_config(
            robot_cfg,
            world_model=None,  # kinematics only; obstacles are the motion planner's job
            rotation_threshold=rotation_threshold,
            position_threshold=position_threshold,
            num_seeds=num_seeds,
            self_collision_check=False,
            self_collision_opt=False,
            tensor_args=tensor_args,
            use_cuda_graph=True,
        )
    )

    # CuRobo solves in base_link frame, not world.
    t_base, q_base_xyzw = world_poses_to_base_link_frame(
        t_candidates_world, q_candidates_world_xyzw, t_base_world, q_base_world_xyzw
    )

    # It also wants contiguous float32 and wxyz quaternions.
    position = np.ascontiguousarray(t_base, dtype=np.float32)
    quaternion = np.ascontiguousarray(quaternion_xyzw_to_wxyz(q_base_xyzw), dtype=np.float32)
    goal = Pose(tensor_args.to_device(position), tensor_args.to_device(quaternion))

    result = ik_solver.solve_batch(goal)
    torch.cuda.synchronize()

    reachable = result.success.squeeze(-1).cpu().numpy().astype(bool)
    q_joints = result.solution.squeeze(1).cpu().numpy().astype(np.float32)
    q_joints[~reachable] = 0.0  # so an unreachable row can't be mistaken for a valid config
    return reachable, q_joints


def save_cache(
    cache_path: str,
    t_candidates_world: np.ndarray,
    q_candidates_world_xyzw: np.ndarray,
    reachable: np.ndarray,
    q_joints: np.ndarray,
    urdf_path: str,
    base_link: str,
    ee_link: str,
) -> None:
    """Write one object's cache. Poses stay in world frame."""
    np.savez(
        cache_path,
        t_candidates_world=t_candidates_world.astype(np.float32),
        q_candidates_world_xyzw=q_candidates_world_xyzw.astype(np.float32),
        reachable=reachable,
        q_joints=q_joints,
        urdf_path=urdf_path,
        base_link=base_link,
        ee_link=ee_link,
    )


def load_reachability_cache(cache_path: str) -> dict:
    """Load a pre-computed reachability cache."""
    data = np.load(cache_path, allow_pickle=False)
    return {key: data[key] for key in data.files}


def build_cache_for_object(ycb_object: str | None = None) -> str:
    """Build a reachability cache for one object. Returns the path written."""
    # Deferred — the rest of this module is importable without a sim or GPU.
    from nbv_core.sim_env import ASSET_PATH, DEFAULT_YCB_OBJECT, SimEnv

    ycb_object = ycb_object or DEFAULT_YCB_OBJECT
    os.makedirs(CACHE_DIR, exist_ok=True)

    env = SimEnv(render=False, ycb_object=ycb_object)
    t_obj_world = env.obj_pos
    r_min, r_max = env.orbit_shell()
    table_top_z = env.table_top_z
    t_base_world, q_base_world_xyzw = env.base_pose()
    env.close()

    t_cand, q_cand = sample_candidate_camera_poses(
        t_obj_world,
        radius=(r_min, r_max, N_RADIUS),
        elevation_deg=ELEVATION_DEG,
        n_azimuth=N_AZIMUTH,
        z_min_world=table_top_z,
    )
    urdf_path = os.path.join(ASSET_PATH, "ur5_robotiq_85.urdf")
    reachable, q_joints = ik_filter(
        urdf_path, BASE_LINK, EE_LINK, t_cand, q_cand, t_base_world, q_base_world_xyzw
    )

    cache_path = cache_path_for(ycb_object)
    save_cache(cache_path, t_cand, q_cand, reachable, q_joints, urdf_path, BASE_LINK, EE_LINK)

    print(f"{ycb_object}: {reachable.sum()}/{len(reachable)} reachable "
          f"at r={r_min:.2f}..{r_max:.2f} -> {os.path.basename(cache_path)}")
    return cache_path


if __name__ == "__main__":
    import argparse

    from nbv_core.sim_env import DEFAULT_YCB_OBJECT, ycb_names

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("object", nargs="?", default=DEFAULT_YCB_OBJECT, choices=ycb_names())
    parser.add_argument("--all", action="store_true", help="build a cache for every object")
    args = parser.parse_args()

    for name in ycb_names() if args.all else [args.object]:
        build_cache_for_object(name)
