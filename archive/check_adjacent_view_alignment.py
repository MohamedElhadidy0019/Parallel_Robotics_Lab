"""
Direct test of a specific question: is the "ghosting" seen when combining
multiple fixed-orbit views a per-view systematic bug (camera pose or
back-projection wrong in a way that varies with view angle), or just sparse
angular coverage (adjacent views don't overlap much on a curved object)?

Test: capture 4 viewpoints packed tightly together (3 degrees apart) around
CENTER_THETA_DEG. Since these cameras are almost looking at the same spot
from almost the same angle, their visible surfaces should overlap heavily.
  - If they line up into one tight, coherent patch -> per-view reconstruction
    is correct there, ghosting nearby is a sparse-coverage/framing issue.
  - If even these near-identical viewpoints show offset/non-overlapping
    patches -> there's a real per-view systematic bug specific to this part
    of the arm's range of motion.

First run (CENTER_THETA_DEG=270, the "MID"/natural-pose angle) came back
clean (~1-2mm agreement). This run targets ~218 degrees instead - the region
where the 20-view mustard-only scan showed a cluster of anomalously-tall,
badly-overlapping views (2-6) - to check whether alignment breaks down
specifically in that harder part of the arm's configuration space.

Usage: python check_adjacent_view_alignment.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from nbv_environment import NBVEnv2
from nbv_core.camera_geometry import capture_rgb_and_depth, compute_intrinsics, backproject_depth, edge_discontinuity_mask

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures", "adjacent_view_alignment_check")
CENTER_THETA_DEG = 218.0  # matches the peak-anomaly view (view 4) from the 20-view scan
DELTA_DEG = 3.0
NUM_CLOSE_VIEWS = 4
HEIGHT_M = 1.2


def capture_mustard_points(env: NBVEnv2) -> np.ndarray:
    _, depth_mm, t_cam_world, q_cam_world, body_ids = capture_rgb_and_depth(
        env._p, env.robot_id, env.camera_link,
        env.camera.width, env.camera.height, env.camera.fov,
        env.camera.near, env.camera.far,
    )
    fx, fy, cx, cy = compute_intrinsics(env.camera.width, env.camera.height, env.camera.fov)
    points_world_grid = backproject_depth(depth_mm, t_cam_world, q_cam_world, fx, fy, cx, cy)

    depth_m = depth_mm.astype(np.float64) / 1000.0
    is_mustard = body_ids == env.obj_id
    is_not_edge_artifact = edge_discontinuity_mask(depth_m)
    return points_world_grid[is_mustard & is_not_edge_artifact]


def move_to(env: NBVEnv2, theta: float, radius: float) -> None:
    viewpoint = np.array([
        env.obj_pos[0] + radius * np.cos(theta),
        env.obj_pos[1] + radius * np.sin(theta),
        HEIGHT_M,
    ])
    orientation = env._lookat_quaternion(viewpoint, env.obj_pos)
    joint_targets = env.get_ik_joints(viewpoint.tolist(), orientation)
    env.execute_joint_states(joint_targets, absolute=True)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    env = NBVEnv2(render=False)
    radius = env._compute_safe_orbit_radius()

    deltas_deg = [(-1 + i) * DELTA_DEG for i in range(NUM_CLOSE_VIEWS)]  # -3, 0, 3, 6 degrees from center
    center_theta = np.radians(CENTER_THETA_DEG)
    thetas = [center_theta + np.radians(d) for d in deltas_deg]

    all_points = []
    view_index_per_point = []
    for i, theta in enumerate(thetas):
        move_to(env, theta, radius)
        points = capture_mustard_points(env)
        print(f"Close-view {i} (center {deltas_deg[i]:+.0f} deg, theta={np.degrees(theta):.1f} deg): "
              f"{len(points)} points, centroid={np.round(points.mean(axis=0), 4)}")
        np.save(os.path.join(OUTPUT_DIR, f"close_view_theta{CENTER_THETA_DEG:.0f}_{i}.npy"), points)
        all_points.append(points)
        view_index_per_point.append(np.full(len(points), i))

    # Hard number, not a visual guess: for each pair of consecutive close
    # views, how far is each point in one from its NEAREST neighbor in the
    # other? If the two views genuinely overlap, this should be tiny (sub-mm
    # to a few mm - about the spacing between neighboring points on the same
    # surface). If they're offset, it'll be much larger.
    print()
    for i in range(len(all_points) - 1):
        tree = cKDTree(all_points[i + 1])
        nearest_dist, _ = tree.query(all_points[i])
        print(f"View {i} -> nearest point in view {i+1}: "
              f"mean={nearest_dist.mean()*1000:.2f}mm  median={np.median(nearest_dist)*1000:.2f}mm  "
              f"max={nearest_dist.max()*1000:.2f}mm")

    combined_points = np.concatenate(all_points, axis=0)
    view_index_per_point = np.concatenate(view_index_per_point)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        combined_points[:, 0], combined_points[:, 1], combined_points[:, 2],
        s=2, c=view_index_per_point, cmap='tab10', vmin=0, vmax=9,
    )
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(f"{NUM_CLOSE_VIEWS} viewpoints, {DELTA_DEG} deg apart, around theta={CENTER_THETA_DEG:.0f} deg")
    fig.colorbar(scatter, ax=ax, label="close-view index", shrink=0.6)
    figure_path = os.path.join(OUTPUT_DIR, f"adjacent_views_by_index_theta{CENTER_THETA_DEG:.0f}.png")
    fig.savefig(figure_path, dpi=150)
    print(f"\nSaved {figure_path}")
    print("If the colors are tightly interleaved in one patch -> per-view reconstruction is correct "
          "(ghosting = sparse coverage, more views would help).")
    print("If the colors form separate offset patches even at 3 degrees apart -> real per-view bug "
          "(more views would make ghosting worse, not better).")


if __name__ == "__main__":
    main()
