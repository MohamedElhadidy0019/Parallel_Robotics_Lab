"""
Candidate camera-viewpoint sampling + CuRobo IK-reachability caching
(Step A of the NBV plan). Two independent halves live here:

  - sample_candidate_camera_poses(): pure geometry, no CuRobo/PyBullet
    dependency - a denser/more general version of the fixed half-orbit
    move_through_orbit() already uses (nbv_environment.py), extended to a
    hemisphere shell (multiple radii x azimuth x elevation) instead of one
    fixed radius/height ring.
  - build_and_save_reachability_cache(): drives CuRobo's IKSolver over the
    sampled candidates and writes the reachable/unreachable result (+ solved
    joint config for reachable ones) to an .npz file, so it's solved once
    and reused rather than re-solved live every run.

Orientation for every candidate uses the same camera-frame lookat
convention as nbv_environment.NBVEnv2._camera_lookat_quaternion (local +Z
forward, local +Y up) - the convention camera_geometry.capture_rgb_and_depth
actually renders with, and the one dummy_camera_link's IK target must match.
Deliberately re-derived here (not imported off an NBVEnv2 instance) so this
module has no PyBullet dependency and can sample candidates without a
running simulation.
"""
import numpy as np

XYZW_TO_WXYZ = np.array([3, 0, 1, 2])


def camera_lookat_quaternion_xyzw(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Same convention/derivation as NBVEnv2._camera_lookat_quaternion, PyBullet xyzw order."""
    from scipy.spatial.transform import Rotation

    z_axis = target - eye
    z_axis = z_axis / np.linalg.norm(z_axis)
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(z_axis, world_up)) > 0.99:
        world_up = np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(world_up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    R = np.column_stack([x_axis, y_axis, z_axis])
    return Rotation.from_matrix(R).as_quat()


def quaternion_xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    """(..., 4) PyBullet/scipy xyzw -> (..., 4) CuRobo wxyz."""
    return np.asarray(q_xyzw)[..., XYZW_TO_WXYZ]


def world_poses_to_base_link_frame(
    t_world: np.ndarray, q_world_xyzw: np.ndarray,
    t_base_world: np.ndarray, q_base_world_xyzw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    CuRobo's RobotConfig.from_basic(..., base_link=...) solves IK for goal
    poses expressed relative to that link, NOT the PyBullet world frame -
    RobotEnv loads the UR5 at world position [0, 0, 0.9] (see
    ur5_environment.py's initial_position), so feeding it raw world-frame
    candidates silently offsets every target by that much and tanks
    reachability (~2/360 candidates "reachable" before this fix, vs. the
    fixed-orbit baseline's ~80%). t_base_world/q_base_world_xyzw should come
    from a live env: self._p.getBasePositionAndOrientation(self.robot_id).
    """
    from scipy.spatial.transform import Rotation

    R_base_world = Rotation.from_quat(q_base_world_xyzw)
    t_base = R_base_world.inv().apply(t_world - t_base_world[None, :])
    q_base = (R_base_world.inv() * Rotation.from_quat(q_world_xyzw)).as_quat()
    return t_base, q_base


def sample_candidate_camera_poses(
    t_obj_world: np.ndarray,
    r_min: float,
    r_max: float,
    n_radius: int = 3,
    n_theta: int = 24,
    phi_min_deg: float = 10.0,
    phi_max_deg: float = 75.0,
    n_phi: int = 5,
    z_min_world: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Hemisphere-shell candidate camera poses around t_obj_world: spherical
    coordinates (radius r, azimuth theta full 2pi, elevation phi above the
    horizontal plane), each pose oriented to look straight at the object
    center. Full 2pi azimuth is intentionally sampled (not just the
    robot-facing pi..2pi half move_through_orbit uses) - the CuRobo
    reachability filter naturally rejects the unreachable far side, and
    caching makes the extra samples a one-time cost, not a per-run one.

    z_min_world (e.g. table height) drops candidates that would put the
    camera below/through a known obstacle plane; None skips that filter.

    Returns (t_candidates_world (N, 3), q_candidates_world_xyzw (N, 4)).
    """
    radii = np.linspace(r_min, r_max, n_radius)
    thetas = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
    phis = np.radians(np.linspace(phi_min_deg, phi_max_deg, n_phi))

    r_grid, theta_grid, phi_grid = np.meshgrid(radii, thetas, phis, indexing="ij")
    r_grid, theta_grid, phi_grid = r_grid.ravel(), theta_grid.ravel(), phi_grid.ravel()

    offsets = np.stack([
        r_grid * np.cos(phi_grid) * np.cos(theta_grid),
        r_grid * np.cos(phi_grid) * np.sin(theta_grid),
        r_grid * np.sin(phi_grid),
    ], axis=-1)
    t_candidates_world = t_obj_world[None, :] + offsets

    if z_min_world is not None:
        keep = t_candidates_world[:, 2] >= z_min_world
        t_candidates_world = t_candidates_world[keep]

    q_candidates_world_xyzw = np.stack([
        camera_lookat_quaternion_xyzw(t, t_obj_world) for t in t_candidates_world
    ], axis=0)
    return t_candidates_world, q_candidates_world_xyzw


def build_and_save_reachability_cache(
    urdf_path: str,
    base_link: str,
    ee_link: str,
    t_candidates_world: np.ndarray,
    q_candidates_world_xyzw: np.ndarray,
    t_base_world: np.ndarray,
    q_base_world_xyzw: np.ndarray,
    cache_path: str,
    position_threshold: float = 0.005,
    rotation_threshold: float = 0.05,
    num_seeds: int = 20,
) -> None:
    """
    Batch-IK-checks every candidate via CuRobo (RobotConfig.from_basic +
    IKSolver, exactly as validated in curobo_ur5_ik_test.py) and saves the
    reachable mask + solved joint config (zeroed for unreachable candidates)
    to cache_path as an .npz - computed once here, loaded by every later
    planner run instead of re-solved live.

    t_base_world/q_base_world_xyzw: the robot's base_link pose in world
    frame (e.g. env._p.getBasePositionAndOrientation(env.robot_id)) - CuRobo
    solves IK relative to base_link, not world, so candidates are
    transformed into that frame before querying (see
    world_poses_to_base_link_frame). Everything saved to cache_path stays in
    world frame - the frame every other consumer in this project uses.
    """
    import torch
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.types.robot import RobotConfig
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

    tensor_args = TensorDeviceType()
    robot_cfg = RobotConfig.from_basic(urdf_path, base_link, ee_link, tensor_args)
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        None,
        rotation_threshold=rotation_threshold,
        position_threshold=position_threshold,
        num_seeds=num_seeds,
        self_collision_check=False,
        self_collision_opt=False,
        tensor_args=tensor_args,
        use_cuda_graph=True,
    )
    ik_solver = IKSolver(ik_config)

    t_candidates_base, q_candidates_base_xyzw = world_poses_to_base_link_frame(
        t_candidates_world, q_candidates_world_xyzw, t_base_world, q_base_world_xyzw,
    )
    q_wxyz = np.ascontiguousarray(quaternion_xyzw_to_wxyz(q_candidates_base_xyzw))
    position = tensor_args.to_device(np.ascontiguousarray(t_candidates_base.astype(np.float32)))
    quaternion = tensor_args.to_device(q_wxyz.astype(np.float32)).contiguous()
    goal = Pose(position, quaternion)

    result = ik_solver.solve_batch(goal)
    torch.cuda.synchronize()

    reachable = result.success.squeeze(-1).cpu().numpy().astype(bool)
    q_joints = result.solution.squeeze(1).cpu().numpy().astype(np.float32)
    q_joints[~reachable] = 0.0

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
    print(f"Reachability cache: {reachable.sum()}/{len(reachable)} candidates reachable "
          f"-> saved to {cache_path}")


def load_reachability_cache(cache_path: str) -> dict:
    """Loads the .npz written by build_and_save_reachability_cache back into a plain dict."""
    data = np.load(cache_path, allow_pickle=False)
    return {key: data[key] for key in data.files}
