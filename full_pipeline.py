"""
The whole scan -> reconstruct -> grasp pipeline in one file, so the full
shape is visible at a glance. Three functions are swap points for work that
isn't real yet - each marked TODO below, each with a signature designed so
replacing its body is enough (the surrounding loop/orchestration doesn't
need to change):

  - move_camera_to()          motion execution - PyBullet IK today, CuRobo later
  - select_next_view_pose()   NBV - fixed orbit today, real planner later
  - compute_grasp_pose()      grasp - nothing at all today

Running this script today still produces a real, working combined point
cloud (mustard-bottle-only, same filtering/back-projection as
scan_and_save_mustard_only.py, no ICP) using the current stand-ins.

Usage:
  python full_pipeline.py            # headless, save the combined cloud
  python full_pipeline.py --view     # also open an Open3D window at the end
"""
import argparse
import os

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation, Slerp

from nbv_environment import NBVEnv2
from nbv_core.camera_geometry import capture_rgb_and_depth, compute_intrinsics, backproject_depth, edge_discontinuity_mask

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures", "full_pipeline")
NUM_VIEWS = 20
ORBIT_HEIGHT_M = 1.2
EDGE_DISCONTINUITY_THRESHOLD_M = 0.02
MAX_POSE_ERROR_M = 0.015  # matches move_through_orbit's reachability-reject tolerance
GLIDE_STEP_SIZE_M = 0.01  # ~1cm of camera travel per glide waypoint - keeps motion smooth whether the
                          # jump is small (adjacent orbit views) or large (e.g. the first move, from
                          # wherever the arm starts to the first viewpoint), instead of a fixed step
                          # count that's too coarse for big jumps and wastefully fine for small ones
MIN_GLIDE_STEPS = 5
MAX_JOINT_STEP_RAD = np.radians(60)  # a single glide step's IK solution jumping further than this from
                                      # the arm's live joint state signals a bad IK branch, not real motion


def get_arm_joint_limits(env: NBVEnv2) -> tuple[np.ndarray, np.ndarray]:
    lowers, uppers = [], []
    for idx in env.arm_joint_indices:
        info = env._p.getJointInfo(env.robot_id, idx)
        lowers.append(info[8])
        uppers.append(info[9])
    return np.array(lowers), np.array(uppers)


def solve_camera_ik(
    env: NBVEnv2, t_world: np.ndarray, q_world: list[float],
    joint_lowers: np.ndarray, joint_uppers: np.ndarray,
) -> list[float]:
    """
    Live-joint-state-seeded IK (get_ik_joints(link=camera_link)) - matches
    orbit_and_capture's proven-smooth approach, no resetJointState
    teleporting, so no GUI flicker. Falls back to the reference-reseeded,
    branch-deterministic _get_camera_ik_joints (visibly teleports the GUI
    once, but only when actually needed) if the live-seeded solution
    violates a joint limit or jumps implausibly far from the arm's current
    state - the failure mode ("crazy" motion) that motivated
    _get_camera_ik_joints in the first place, which mainly bites on large
    excursions through the harder/near-singular parts of the orbit (e.g.
    the very first move, straight from the arm's rest pose to the far edge
    of the orbit, with no warm-up).
    """
    live_joint_pos = np.array([env._p.getJointState(env.robot_id, idx)[0] for idx in env.arm_joint_indices])
    joint_targets = np.array(env.get_ik_joints(t_world.tolist(), q_world, link=env.camera_link))

    within_limits = bool(np.all(joint_targets >= joint_lowers) and np.all(joint_targets <= joint_uppers))
    max_step_rad = float(np.max(np.abs(joint_targets - live_joint_pos)))
    if not within_limits or max_step_rad > MAX_JOINT_STEP_RAD:
        joint_targets = np.array(env._get_camera_ik_joints(t_world.tolist(), q_world))

    return joint_targets.tolist()


def move_camera_to(
    env: NBVEnv2, t_target_world: np.ndarray, q_target_world: list[float],
) -> tuple[bool, np.ndarray]:
    """
    TODO (-> CuRobo): moves the arm so the camera reaches the given
    world-frame pose. Currently: glides through fine intermediate poses
    (linear position interpolation + orientation slerp), one waypoint per
    ~GLIDE_STEP_SIZE_M of camera travel, from wherever the camera currently
    is to the target, driving PyBullet IK + position control at each step -
    one smooth continuous move, not a single big jump, generalizing
    move_through_orbit's fixed-orbit glide to work between any two
    arbitrary poses (needed once NBV starts picking non-orbit viewpoints).
    Swapping in CuRobo means replacing this function's body with proper
    trajectory planning to the same (t_target_world, q_target_world) - same
    return contract (reached, t_achieved_world) - nothing else in this file
    needs to change.

    Each glide step's IK goes through solve_camera_ik() - live-seeded (no
    GUI-flickering resetJointState teleport) in the common case, falling
    back to the teleport-protected solver only on the rare step where the
    live-seeded solution is actually implausible. See solve_camera_ik()'s
    docstring for why.

    Returns (reached, t_achieved_world) - reached=False if the settled
    camera pose is too far from the target (e.g. out of the arm's reach).
    """
    t_current_world = np.array(env._p.getLinkState(env.robot_id, env.camera_link)[0])
    q_current_world = env._p.getLinkState(env.robot_id, env.camera_link)[1]
    slerp = Slerp([0, 1], Rotation.from_quat([q_current_world, q_target_world]))
    joint_lowers, joint_uppers = get_arm_joint_limits(env)

    travel_distance_m = float(np.linalg.norm(t_target_world - t_current_world))
    n_steps = max(MIN_GLIDE_STEPS, int(np.ceil(travel_distance_m / GLIDE_STEP_SIZE_M)))

    for alpha in np.linspace(0.0, 1.0, n_steps):
        is_final_step = alpha == 1.0
        t_step_world = t_current_world + alpha * (t_target_world - t_current_world)
        q_step_world = q_target_world if is_final_step else slerp([alpha]).as_quat()[0].tolist()
        joint_targets = solve_camera_ik(env, t_step_world, q_step_world, joint_lowers, joint_uppers)
        env.execute_joint_states(joint_targets, absolute=is_final_step)

    env._wait_for_arm_at_rest()
    env._snap_to_joint_targets(joint_targets)

    t_achieved_world = np.array(env._p.getLinkState(env.robot_id, env.camera_link)[0])
    pose_error_m = float(np.linalg.norm(t_achieved_world - t_target_world))
    return pose_error_m <= MAX_POSE_ERROR_M, t_achieved_world


def select_next_view_pose(
    env: NBVEnv2, accumulated_points_so_far: list[np.ndarray], view_index: int,
) -> tuple[np.ndarray, list[float]] | None:
    """
    TODO (-> nbv_planner.py): ignores accumulated_points_so_far entirely
    right now - just returns the next pose from a fixed, precomputed orbit
    (same half-orbit as move_through_orbit), or None once NUM_VIEWS is
    reached. Real NBV replaces this function's body: build/update an
    occupancy voxel grid from accumulated_points_so_far, ray-cast candidate
    viewpoints against it, greedily pick whichever resolves the most
    unknown space, and return None once a stopping criterion (not a fixed
    view count) is met. The while-loop in run_scan_and_reconstruct() below
    already has the shape this needs - decide one pose at a time, stop via
    None - so it doesn't change.
    """
    if view_index >= NUM_VIEWS:
        return None
    radius = env._compute_safe_orbit_radius()
    theta = np.pi + (view_index / (NUM_VIEWS - 1)) * np.pi  # matches np.linspace(pi, 2pi, NUM_VIEWS, endpoint=True)
    t_target_world = np.array([
        env.obj_pos[0] + radius * np.cos(theta),
        env.obj_pos[1] + radius * np.sin(theta),
        ORBIT_HEIGHT_M,
    ])
    q_target_world = env._camera_lookat_quaternion(t_target_world, env.obj_pos)
    return t_target_world, q_target_world


def capture_and_backproject_view(env: NBVEnv2, view_index: int) -> np.ndarray:
    """Capture one view, keep only mustard-bottle pixels, back-project to world-frame points. Real/final, not a placeholder."""
    _, depth_mm, t_cam_world, q_cam_world, body_ids = capture_rgb_and_depth(
        env._p, env.robot_id, env.camera_link,
        env.camera.width, env.camera.height, env.camera.fov,
        env.camera.near, env.camera.far,
    )

    depth_m = depth_mm.astype(np.float64) / 1000.0
    is_mustard = body_ids == env.obj_id
    is_not_edge_artifact = edge_discontinuity_mask(depth_m, threshold_m=EDGE_DISCONTINUITY_THRESHOLD_M)
    is_mustard_and_trustworthy = is_mustard & is_not_edge_artifact

    fx, fy, cx, cy = compute_intrinsics(env.camera.width, env.camera.height, env.camera.fov)
    points_world_grid = backproject_depth(depth_mm, t_cam_world, q_cam_world, fx, fy, cx, cy)
    points_world = points_world_grid[is_mustard_and_trustworthy]

    print(f"View {view_index}: {len(points_world)} mustard-only points")
    return points_world


def compute_grasp_pose(points_world: np.ndarray):
    """
    TODO (-> grasp_planner.py): nothing implemented yet. Replace this with
    the real grasp planner - a PCA-based heuristic on the reconstructed
    point cloud (find the object's principal axes, pick an approach
    direction/pose from them), NOT shelf_gym's own grasping_utils.py
    box-face sampler (off limits per project constraints). Currently a
    no-op stub.
    """
    print("TODO: grasp pose not implemented yet - skipping.")
    return None


def run_scan_and_reconstruct(env: NBVEnv2) -> np.ndarray:
    accumulated_points: list[np.ndarray] = []
    view_index = 0
    while True:
        next_view = select_next_view_pose(env, accumulated_points, view_index)
        if next_view is None:
            break
        t_target_world, q_target_world = next_view

        reached, t_achieved_world = move_camera_to(env, t_target_world, q_target_world)
        if not reached:
            pose_error_m = float(np.linalg.norm(t_achieved_world - t_target_world))
            print(f"View {view_index}: skipped - camera settled {pose_error_m * 1000:.1f}mm "
                  f"from the intended pose (likely beyond the arm's reach here)")
        else:
            accumulated_points.append(capture_and_backproject_view(env, view_index))

        view_index += 1

    return np.concatenate(accumulated_points, axis=0) if accumulated_points else np.zeros((0, 3))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gui", action="store_true", help="Show the PyBullet GUI (watch the robot move).")
    parser.add_argument("--view", action="store_true", help="Open an Open3D window with the combined cloud at the end.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    env = NBVEnv2(render=args.gui)

    points_world = run_scan_and_reconstruct(env)
    print(f"\nReconstructed {len(points_world)} points total")

    npy_path = os.path.join(OUTPUT_DIR, "combined_pointcloud.npy")
    np.save(npy_path, points_world)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_world)
    ply_path = os.path.join(OUTPUT_DIR, "combined_pointcloud.ply")
    o3d.io.write_point_cloud(ply_path, pcd)
    print(f"Saved point cloud to {npy_path} and {ply_path}")

    grasp_pose = compute_grasp_pose(points_world)
    print(f"Grasp pose: {grasp_pose}")

    if args.view:
        # PyBullet's EGL renderer (used for every depth capture above) and Open3D's
        # interactive viewer both want exclusive access to the same GPU/X resource in
        # one process - release PyBullet's hold on it first or draw_geometries() fails
        # with a GLX BadAccess error.
        env.close()
        print("Opening Open3D viewer (drag to rotate, scroll to zoom, close the window to continue)...")
        o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()
