"""
Same fixed 20-view orbit scan as scan_and_save_mustard_only.py, but using
shelf_gym's OWN existing Camera.get_cam_in_hand()/get_pointcloud() pipeline
for capture + depth-to-3D reconstruction, instead of nbv_core.camera_geometry
(this project's original implementation).

Why: a small (~0-3.6mm) per-view bias against the ground-truth mesh remains
even with camera pose now proven essentially exact (see
project_pointcloud_layering_bug memory - camera pose/IK/settling ruled out
via three separate experiments; two ICP-based post-hoc registration
attempts also didn't help, arguing against a simple per-view rigid-alignment
explanation). Comparing against shelf_gym's own capture/reconstruction
pipeline checks whether this residual is specific to our own backprojection
math, or shows up just the same using the sim's built-in utilities - which
would point at something further upstream (e.g. the rendered depth buffer
itself) rather than a bug in this project's own code.

Does NOT modify or replace anything in the existing pipeline
(scan_and_save_mustard_only.py, nbv_core/, compare_pointcloud_to_mesh.py,
etc. are all untouched) - this is a separate, standalone comparison, saving
to its own captures/scan_and_save_mustard_only_shelfgym/ directory. The
orbit motion itself (move_through_orbit) is reused as-is unchanged - only
the capture/reconstruction step is swapped for shelf_gym's own.

shelf_gym's get_cam_in_hand() doesn't give per-object segmentation for free
without its own instance_to_class_conversion setup (which this project
deliberately never wired up), so "mustard-only" filtering here still reuses
this project's own body-ID segmentation call and edge_discontinuity_mask -
only the CAMERA CAPTURE + DEPTH-TO-3D RECONSTRUCTION step is shelf_gym's,
not the object-isolation step.

Usage: python scan_and_save_mustard_only_shelfgym.py
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import trimesh

from nbv_environment import NBVEnv2
from nbv_core.camera_geometry import edge_discontinuity_mask
from nbv_core.io_utils import save_depth_image
from compare_pointcloud_to_mesh import MESH_PATH, R_MESH_BASELINK, R_LINK_INERTIAL, T_LINK_INERTIAL

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures", "scan_and_save_mustard_only_shelfgym")
NUM_VIEWS = 20
ORBIT_HEIGHT_M = 1.2
EDGE_DISCONTINUITY_THRESHOLD_M = 0.02


def save_view(view_index: int, env: NBVEnv2) -> None:
    # shelf_gym's own existing capture + depth-to-3D pipeline (Open3D's
    # create_from_depth_image internally, a different code path than this
    # project's own nbv_core.camera_geometry.backproject_depth).
    result = env.camera.get_cam_in_hand(env.robot_id, env.camera_link, remove_gripper=False, no_conversion=True)
    depth_mm = result["transformed_depth"]

    # NOT using result["point_cloud"]["numpy"] - found and confirmed a real
    # bug in shelf_gym's own Camera.ogl_vm_to_o3d(): its OpenGL-view-matrix
    # to-Open3D conjugation is wrong (verified: the camera-LOCAL points from
    # create_from_depth_image are correct - e.g. depth=406mm gives local
    # Z=0.406m almost exactly - but ogl_vm_to_o3d's world transform corrupts
    # them, up to hundreds of mm at oblique angles; tried both its invert=True
    # default and invert=False branch, both wrong). Fix: rebuild the local
    # point cloud the same way (their depth_to_o3d + projection_matrix_to_
    # intrinsic + Open3D's create_from_depth_image, kept as shelf_gym's own),
    # but place it in world coordinates using PyBullet's own getLinkState
    # pose directly (already validated extensively elsewhere this session)
    # instead of their conversion. Only actual axis difference between
    # Open3D's local convention (Y-down) and this project's (Y-up) is a
    # single Y flip - X and Z already agree, confirmed by matching this
    # project's own backprojection to within 0.04-0.68mm across a full frame.
    depth_o3d = env.camera.depth_to_o3d(result["depth"])
    intrinsic_matrix = env.camera.projection_matrix_to_intrinsic(env.camera.projection_matrix)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        env.camera.width, env.camera.height, intrinsic_matrix[0, 0], intrinsic_matrix[1, 1],
        env.camera.cx, env.camera.cy,
    )
    pcd_local = o3d.geometry.PointCloud().create_from_depth_image(
        o3d.geometry.Image(depth_o3d), intrinsic, depth_scale=1000.0, depth_trunc=1000.0,
        stride=1, project_valid_depth_only=False,
    )
    points_local = np.asarray(pcd_local.points).reshape(env.camera.height, env.camera.width, 3, order="C")

    cam_pos, cam_quat = env._p.getLinkState(env.robot_id, env.camera_link, computeForwardKinematics=True)[:2]
    R_cam_world = np.array(env._p.getMatrixFromQuaternion(cam_quat)).reshape(3, 3)
    flip_y = np.diag([1.0, -1.0, 1.0])
    points_world_grid = points_local @ (R_cam_world @ flip_y).T + np.asarray(cam_pos)

    # Reuses the SAME view/projection matrices get_cam_in_hand just set on
    # env.camera, so this segmentation is geometrically consistent with the
    # capture above - only the object-isolation step is this project's own.
    _, _, _, _, segmentation_raw = env._p.getCameraImage(
        env.camera.width, env.camera.height, env.camera.view_matrix, env.camera.projection_matrix,
        shadow=False, renderer=env._p.ER_BULLET_HARDWARE_OPENGL,
    )
    segmentation_raw = np.array(segmentation_raw)
    body_ids = np.where(segmentation_raw >= 0, segmentation_raw & ((1 << 24) - 1), -1)

    depth_m = depth_mm.astype(np.float64) / 1000.0
    is_mustard = body_ids == env.obj_id
    is_not_edge_artifact = edge_discontinuity_mask(depth_m, threshold_m=EDGE_DISCONTINUITY_THRESHOLD_M)
    is_mustard_and_trustworthy = is_mustard & is_not_edge_artifact
    print(f"View {view_index}: {is_mustard.sum()} / {is_mustard.size} pixels are the mustard bottle "
          f"({is_mustard_and_trustworthy.sum()} kept after also dropping edge artifacts)")

    points_world = points_world_grid[is_mustard_and_trustworthy]

    depth_mm_mustard_only = np.where(is_mustard_and_trustworthy, depth_mm, env.camera.far * 1000.0)
    depth_image_path = os.path.join(OUTPUT_DIR, f"depth_mustard_{view_index:02d}.png")
    save_depth_image(depth_mm_mustard_only, depth_image_path, near_m=env.camera.near, far_m=env.camera.far)

    point_cloud_path = os.path.join(OUTPUT_DIR, f"pointcloud_mustard_{view_index:02d}.npy")
    np.save(point_cloud_path, points_world)

    print(f"View {view_index}: saved {len(points_world)} mustard-only points\n")


def compare_to_mesh(combined_points: np.ndarray, t_obj_world: np.ndarray, q_obj_world: np.ndarray) -> None:
    """Same ground-truth mesh placement as compare_pointcloud_to_mesh.py (imported constants, not duplicated)."""
    import pybullet as pb

    R_inertial_world = np.array(pb.getMatrixFromQuaternion(q_obj_world)).reshape(3, 3)
    R_baselink_world = R_inertial_world @ R_LINK_INERTIAL.T
    t_baselink_world = R_inertial_world @ (-R_LINK_INERTIAL.T @ T_LINK_INERTIAL) + t_obj_world

    tm = trimesh.load(MESH_PATH, process=False)
    if isinstance(tm, trimesh.Scene):
        tm = trimesh.util.concatenate(list(tm.geometry.values()))
    v_mesh = np.asarray(tm.vertices)
    v_world = (R_baselink_world @ (R_MESH_BASELINK @ v_mesh.T)).T + t_baselink_world

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(v_world)
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(tm.faces))

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    query = o3d.core.Tensor(combined_points.astype(np.float32))
    d_mm = scene.compute_distance(query).numpy() * 1000.0
    print(f"\nshelf_gym-pipeline point-to-mesh distance over {len(d_mm)} points:")
    print(f"  mean={d_mm.mean():.2f}mm median={np.median(d_mm):.2f}mm "
          f"rms={np.sqrt((d_mm**2).mean()):.2f}mm within5mm={(d_mm<5).mean()*100:.1f}%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gui", action="store_true", help="Show the PyBullet GUI (watch the robot move).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    env = NBVEnv2(render=args.gui)

    env.move_through_orbit(
        n_views=NUM_VIEWS,
        height=ORBIT_HEIGHT_M,
        on_stop=lambda view_index: save_view(view_index, env),
    )

    t_obj_world, q_obj_world = env._p.getBasePositionAndOrientation(env.obj_id)
    t_obj_world = np.array(t_obj_world)
    q_obj_world = np.array(q_obj_world)
    np.savez(os.path.join(OUTPUT_DIR, "object_pose.npz"), t_obj_world=t_obj_world, q_obj_world=q_obj_world)

    import glob
    paths = sorted(glob.glob(os.path.join(OUTPUT_DIR, "pointcloud_mustard_*.npy")))
    combined_points = np.concatenate([np.load(p) for p in paths], axis=0)
    combined_path = os.path.join(OUTPUT_DIR, "combined_mustard_pointcloud.npy")
    np.save(combined_path, combined_points)
    print(f"Done. {NUM_VIEWS} views, {len(combined_points)} combined points saved to {OUTPUT_DIR}")

    compare_to_mesh(combined_points, t_obj_world, q_obj_world)


if __name__ == "__main__":
    main()
