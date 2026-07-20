"""
Correctness check for our own depth -> world-frame back-projection math
(nbv_core.camera_geometry.backproject_depth).

We capture one frame using shelf_gym's own camera pipeline
(Camera.get_cam_in_hand), which independently computes its own world-frame
point cloud internally via Open3D. That point cloud is NOT used anywhere in
the real pipeline - nbv_environment.py never calls get_cam_in_hand(), it uses
our own capture_rgb_and_depth() instead. Here, we only use shelf_gym's version
as a REFERENCE ANSWER: a second, independent implementation of the same
"depth image -> 3D points" conversion, to check our own math against.

Usage: python test_backprojection.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nbv_environment import NBVEnv2
from nbv_core.camera_geometry import compute_intrinsics, backproject_depth, flatten_valid_points

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "captures", "backprojection_test")

# How close our points need to be to the reference points to call this a pass.
MAX_MEAN_ERROR_M = 0.005  # 5 mm
MAX_P99_ERROR_M = 0.02    # 2 cm


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    env = NBVEnv2(render=False)

    # The only call to get_cam_in_hand() in the whole project - used here
    # purely to get a reference point cloud to check our own math against.
    reference_capture = env.camera.get_cam_in_hand(
        env.robot_id, env.camera_link,
        remove_gripper=False, client_id=env.client_id, no_conversion=True
    )
    depth_image_mm = reference_capture['transformed_depth']
    reference_points_world = reference_capture['point_cloud']['numpy'][:, :, :3]

    # Camera pose at the moment of capture - the same pose reference_capture
    # was rendered from. t = translation (position), q = orientation quaternion.
    t_cam_world, q_cam_world = env._p.getLinkState(
        env.robot_id, env.camera_link, computeForwardKinematics=True
    )[:2]

    # Our own back-projection, from the exact same depth image and camera pose.
    fx, fy, cx, cy = compute_intrinsics(env.camera.width, env.camera.height, env.camera.fov)
    our_points_world = backproject_depth(depth_image_mm, t_cam_world, q_cam_world, fx, fy, cx, cy)

    # Ignore background pixels (no real surface hit) in both point clouds.
    max_depth_mm = env.camera.far * 1000.0 * 0.98
    is_real_surface_hit = depth_image_mm < max_depth_mm
    print(f"Real surface-hit pixels: {is_real_surface_hit.sum()} / {is_real_surface_hit.size}")

    point_errors_m = np.linalg.norm(
        our_points_world[is_real_surface_hit] - reference_points_world[is_real_surface_hit], axis=-1
    )
    print("Distance between our points and the reference points, per pixel (meters):")
    print(f"  mean:   {point_errors_m.mean():.6f}")
    print(f"  median: {np.median(point_errors_m):.6f}")
    print(f"  max:    {point_errors_m.max():.6f}")
    print(f"  p99:    {np.percentile(point_errors_m, 99):.6f}")

    test_passed = (
        point_errors_m.mean() < MAX_MEAN_ERROR_M
        and np.percentile(point_errors_m, 99) < MAX_P99_ERROR_M
    )
    if test_passed:
        print(f"PASS - mean error under {MAX_MEAN_ERROR_M * 1000:.0f}mm "
              f"and p99 under {MAX_P99_ERROR_M * 100:.0f}cm")
    else:
        print("FAIL - error too large, check axis/sign conventions in backproject_depth")

    # Visual sanity dump (headless-safe PNG) of our own points only.
    our_valid_points_world = flatten_valid_points(our_points_world, depth_image_mm, max_depth_mm)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        our_valid_points_world[:, 0], our_valid_points_world[:, 1], our_valid_points_world[:, 2],
        s=1, c=our_valid_points_world[:, 2], cmap='viridis',
    )
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(f"Own back-projected point cloud ({len(our_valid_points_world)} pts)")
    figure_path = os.path.join(OUTPUT_DIR, "own_pointcloud.png")
    fig.savefig(figure_path, dpi=150)
    print(f"Saved visual sanity check to {figure_path}")

    np.save(os.path.join(OUTPUT_DIR, "own_points.npy"), our_valid_points_world)


if __name__ == "__main__":
    main()
