"""
Own depth -> world-frame point cloud back-projection.

Deliberately independent of shelf_gym's Camera.get_cam_in_hand()/
get_pointcloud() (which does the same job internally via Open3D) - this
module is the project's own capture + reconstruction math, only consuming
raw sensor fields (a depth image in millimeters, plus the camera's world
pose).
"""
from typing import TypedDict

import numpy as np
import pybullet as pb
from pybullet_utils.bullet_client import BulletClient


class CapturedFrame(TypedDict):
    """What NBVEnv2.capture_frame() returns - declared here so `frame['...']`
    is known/autocompleted anywhere a captured frame gets passed around."""
    rgb: np.ndarray                # (H, W, 3) uint8
    transformed_depth: np.ndarray  # (H, W) uint16, millimeters
    points_world: np.ndarray       # (N, 3), only points that passed all filters
    points_world_grid: np.ndarray  # (H, W, 3), unfiltered, one point per pixel
    valid_mask: np.ndarray         # (H, W) bool, range_valid_mask & edge_valid_mask
    range_valid_mask: np.ndarray   # (H, W) bool, depth within [min_depth_m, max_depth_m)
    edge_valid_mask: np.ndarray    # (H, W) bool, not a silhouette-edge "flying pixel"
    body_ids: np.ndarray           # (H, W) int32, which PyBullet body each pixel belongs to (-1 = none)
    cam_pos: np.ndarray            # (3,) camera world position
    cam_quat: np.ndarray           # (4,) camera world orientation quaternion


def capture_rgb_and_depth(
    sim: BulletClient,
    robot_id: int,
    camera_link: int,
    width: int,
    height: int,
    fov_deg: float,
    near_m: float,
    far_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Render one frame from the robot's wrist camera and return a real,
    metric depth image - our own replacement for shelf_gym's
    Camera.get_cam_in_hand()/get_image(), which does the same job but
    bundled together with gripper-removal/class-segmentation/point-cloud
    logic we don't want or need here.

    Returns:
        rgb:      (H, W, 3) uint8
        depth_mm: (H, W) uint16, real-world distance from the camera in millimeters
        cam_pos:  (3,) camera world position
        cam_quat: (4,) camera world orientation quaternion
        body_ids: (H, W) int32, the PyBullet body unique ID visible at each
                  pixel (-1 = nothing there, i.e. background). This is
                  ground-truth ONLY because it's a simulator - a real depth
                  camera has no equivalent, so use this for debugging/
                  isolating objects, not as part of the "real" NBV/
                  reconstruction pipeline.
    """
    # Step 1: where is the camera, and which way is it pointing?
    cam_pos, cam_quat = sim.getLinkState(robot_id, camera_link, computeForwardKinematics=True)[:2]
    rotation = np.array(sim.getMatrixFromQuaternion(cam_quat)).reshape(3, 3)
    forward_direction = rotation @ np.array([0.0, 0.0, 1.0])  # camera's local +Z, in world frame
    up_direction = rotation @ np.array([0.0, 1.0, 0.0])       # camera's local +Y, in world frame

    view_matrix = sim.computeViewMatrix(
        cameraEyePosition=cam_pos,
        cameraTargetPosition=np.asarray(cam_pos) + forward_direction,
        cameraUpVector=up_direction,
    )
    projection_matrix = sim.computeProjectionMatrixFOV(
        fov=fov_deg, aspect=width / height, nearVal=near_m, farVal=far_m,
    )

    # Step 2: actually render the image. The 5th value is a segmentation
    # buffer: one entry per pixel encoding which body+link rasterized there.
    _, _, rgba, ogl_depth, segmentation_raw = sim.getCameraImage(
        width, height, view_matrix, projection_matrix,
        shadow=False, renderer=sim.ER_BULLET_HARDWARE_OPENGL,
    )
    rgb = np.array(rgba)[:, :, :3].astype(np.uint8)

    # Step 3: PyBullet's raw depth buffer is NOT real-world distance - it's
    # an OpenGL-normalized value in [0, 1], non-linear in true depth. This
    # formula undoes that non-linearity using only the near/far clip planes,
    # giving back real meters (then converted to millimeters for the rest
    # of the pipeline).
    ogl_depth = np.array(ogl_depth)
    depth_m = (2.0 * near_m * far_m) / (far_m + near_m - (2.0 * ogl_depth - 1.0) * (far_m - near_m))
    depth_mm = np.round(depth_m * 1000.0).astype(np.uint16)

    # Step 4: decode the segmentation buffer down to plain body IDs. Each
    # entry is bodyUniqueId + ((linkIndex + 1) << 24) when something was hit,
    # or -1 for background - we only care about which BODY was hit here, so
    # mask off the link-index bits.
    segmentation_raw = np.array(segmentation_raw)
    body_ids = np.where(segmentation_raw >= 0, segmentation_raw & ((1 << 24) - 1), -1)

    return rgb, depth_mm, np.asarray(cam_pos), np.asarray(cam_quat), body_ids


def compute_intrinsics(width: int, height: int, fov_deg: float) -> tuple[float, float, float, float]:
    """
    Pinhole intrinsics matching PyBullet's computeProjectionMatrixFOV convention,
    where `fov_deg` is the *vertical* field of view.

    fx == fy falls out of the projection matrix algebra once aspect = width/height
    is substituted in - both reduce to height / (2 * tan(fov/2)).

    Returns: (fx, fy, cx, cy)
    """
    fov_rad = np.radians(fov_deg)
    fy = height / (2.0 * np.tan(fov_rad / 2.0))
    fx = fy
    cx = width / 2.0
    cy = height / 2.0
    return fx, fy, cx, cy


def backproject_depth(
    transformed_depth: np.ndarray,
    cam_pos: np.ndarray,
    cam_quat: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """
    Back-project a depth image into world-frame 3D points.

    Args:
        transformed_depth: (H, W) array, real-world depth in millimeters
        cam_pos: (3,) camera world position
        cam_quat: (4,) camera world orientation quaternion, pybullet (x,y,z,w) convention
        fx, fy, cx, cy: pinhole intrinsics from compute_intrinsics()

    Returns:
        (H, W, 3) array of world-frame points, one per pixel.
        Camera-local convention: +Z forward (viewing direction), +Y up,
        matching the camera link's local axes used in capture_rgb_and_depth().
    """
    height, width = transformed_depth.shape
    R = np.array(pb.getMatrixFromQuaternion(cam_quat)).reshape(3, 3)

    depth_m = transformed_depth.astype(np.float64) / 1000.0

    uu, vv = np.meshgrid(np.arange(width), np.arange(height))  # both (H, W)

    Z = depth_m
    X = (uu - cx) / fx * Z
    Y = -(vv - cy) / fy * Z  # image row grows downward, but local +Y is up

    points_local = np.stack([X, Y, Z], axis=-1)  # (H, W, 3)
    points_world = points_local @ R.T + np.asarray(cam_pos)
    return points_world


def edge_discontinuity_mask(depth_m: np.ndarray, threshold_m: float = 0.02) -> np.ndarray:
    """
    True where a pixel's depth is consistent with its immediate neighbors,
    False at silhouette-edge pixels whose depth jumps sharply relative to
    a neighbor (PyBullet's rasterizer can emit an interpolated depth value
    at an object/background boundary - "flying pixels" - that back-project
    into a long spurious streak connecting near and far surfaces).
    """
    padded = np.pad(depth_m, 1, mode='edge')
    center = padded[1:-1, 1:-1]
    neighbor_diffs = np.stack([
        np.abs(center - padded[:-2, 1:-1]),   # up
        np.abs(center - padded[2:, 1:-1]),    # down
        np.abs(center - padded[1:-1, :-2]),   # left
        np.abs(center - padded[1:-1, 2:]),    # right
    ], axis=0)
    return neighbor_diffs.max(axis=0) < threshold_m


def flatten_valid_points(
    points_world: np.ndarray,
    transformed_depth: np.ndarray,
    max_depth_mm: float,
) -> np.ndarray:
    """
    Flatten an (H, W, 3) point grid to an (N, 3) point list, dropping pixels
    that hit nothing (depth at/near the camera's far plane = background).
    """
    valid = transformed_depth < max_depth_mm
    return points_world[valid]
