import numpy as np
import open3d as o3d
from nbv_environment import NBVEnv2
from nbv_core.camera_geometry import capture_rgb_and_depth, compute_intrinsics, backproject_depth, edge_discontinuity_mask
from compare_pointcloud_to_mesh import load_ground_truth_mesh_world, INPUT_DIR
import os

env = NBVEnv2(render=False)
radius = env._compute_safe_orbit_radius()
angles = np.linspace(np.pi, 2 * np.pi, 20)

def move_to(theta, absolute=True):
    viewpoint = np.array([env.obj_pos[0] + radius*np.cos(theta), env.obj_pos[1] + radius*np.sin(theta), 1.2])
    orientation = env._camera_lookat_quaternion(viewpoint, env.obj_pos)
    joint_targets = env._get_camera_ik_joints(viewpoint.tolist(), orientation)
    env.execute_joint_states(joint_targets, absolute=absolute)
    return viewpoint

def capture_points(min_conf=None):
    _, depth_mm, t_cam_world, q_cam_world, body_ids = capture_rgb_and_depth(
        env._p, env.robot_id, env.camera_link,
        env.camera.width, env.camera.height, env.camera.fov,
        env.camera.near, env.camera.far,
    )
    depth_m = depth_mm.astype(np.float64) / 1000.0
    is_mustard = body_ids == env.obj_id
    is_ok = edge_discontinuity_mask(depth_m, threshold_m=0.02)
    keep = is_mustard & is_ok
    fx, fy, cx, cy = compute_intrinsics(env.camera.width, env.camera.height, env.camera.fov)
    grid = backproject_depth(depth_mm, t_cam_world, q_cam_world, fx, fy, cx, cy)
    return grid[keep], np.array(t_cam_world), np.array(q_cam_world)

mesh = load_ground_truth_mesh_world_dummy = None  # placeholder, mesh needs object_pose.npz; build manually below

MID = 3*np.pi/2
move_to(MID, absolute=True)
for t in np.linspace(MID, angles[0], num=16)[1:]:
    move_to(t, absolute=False)

target_views = {4, 5, 6, 13}
results = {}
for view_index, theta in enumerate(angles):
    if view_index > 0:
        for t in np.linspace(angles[view_index-1], theta, num=10)[1:-1]:
            move_to(t, absolute=False)
    move_to(theta, absolute=True)

    if view_index not in target_views:
        continue

    # Normal settle: same as current pipeline (velocity-based, ~20-50 steps)
    for _ in range(50):
        velocities = [env._p.getJointState(env.robot_id, idx)[1] for idx in env.arm_joint_indices]
        if max(abs(v) for v in velocities) < 0.005:
            break
        env.step_simulation(env.per_step_iterations)
    pts_normal, t1, q1 = capture_points()

    # Deep settle: force 500 more physics steps regardless of velocity
    for _ in range(500):
        env.step_simulation(env.per_step_iterations)
    pts_deep, t2, q2 = capture_points()

    dt_mm = np.linalg.norm(t2 - t1) * 1000
    dq_deg = np.degrees(2 * np.arccos(np.clip(abs(np.dot(q1, q2)), 0, 1)))
    results[view_index] = (pts_normal, pts_deep, dt_mm, dq_deg)
    print(f"View {view_index}: cam moved {dt_mm:.3f}mm, rotated {dq_deg:.4f}deg during 500 extra settle steps")

t_obj_world, q_obj_world = env._p.getBasePositionAndOrientation(env.obj_id)
np.savez(os.path.join(INPUT_DIR, "object_pose.npz"),
         t_obj_world=np.array(t_obj_world), q_obj_world=np.array(q_obj_world))
mesh = load_ground_truth_mesh_world()
scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

def signed_bias(pts):
    query = o3d.core.Tensor(pts.astype(np.float32))
    ans = scene.compute_closest_points(query)
    closest = ans['points'].numpy()
    normals = ans['primitive_normals'].numpy()
    diff = pts - closest
    signed = np.einsum('ij,ij->i', diff, normals) * 1000.0
    return signed.mean(), signed.std()

print(f"\n{'view':>4} {'normal_bias_mm':>15} {'deep_settle_bias_mm':>20}")
for view_index, (pts_normal, pts_deep, dt_mm, dq_deg) in results.items():
    b1, s1 = signed_bias(pts_normal)
    b2, s2 = signed_bias(pts_deep)
    print(f"{view_index:>4} {b1:>15.2f} {b2:>20.2f}")
