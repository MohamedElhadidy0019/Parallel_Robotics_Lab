"""RGB-D capture -> backprojection -> world frame."""

import argparse
import time

import numpy as np
import pybullet as p
import pybullet_data

from nbv_core.camera import (
    CameraIntrinsics,
    backproject_depth,
    capture_rgbd,
    depth_buffer_to_linear,
    edge_discontinuity_mask,
    transform_points,
)
from nbv_core.sim_env import DEFAULT_YCB_OBJECT, ycb_names, ycb_urdf

# OpenGL cam frame (X right, Y up, Z back) -> optical (X right, Y down, Z fwd).
T_OPENGL_OPTICAL = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)

INTRINSICS = CameraIntrinsics(width=320, height=240, fov=60.0, near=0.1, far=2.0)


def connect(gui: bool = False) -> int:
    cid = p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    if gui:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=cid)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=cid)
    return cid


def seat_on_plane(oid: int, pos, cid: int):
    """Sit an object's base exactly on z=0, keeping its orientation."""
    lo, _ = p.getAABB(oid, physicsClientId=cid)
    _, orn = p.getBasePositionAndOrientation(oid, physicsClientId=cid)
    p.resetBasePositionAndOrientation(
        oid, [pos[0], pos[1], pos[2] - lo[2]], orn, physicsClientId=cid
    )


def scene(cid: int, urdf: str = "cube_small.urdf", pos=(0.0, 0.0, 0.025), seat: bool = False) -> int:
    """Ground plane plus one object. YCB urdfs need seat=True; they aren't authored to
    rest at the origin, and dropping them tips the cans onto their sides."""
    p.loadURDF("plane.urdf", physicsClientId=cid)
    oid = p.loadURDF(urdf, list(pos), physicsClientId=cid)
    if seat:
        seat_on_plane(oid, pos, cid)
    return oid


def look_at(cam_pos, target_pos, intr: CameraIntrinsics, cid: int):
    """Capture from cam_pos aimed at target_pos. Returns (world points, colours, pose)."""
    rgb, depth, view, _ = capture_rgbd(
        cam_pos, target_pos, np.array([0.0, 0.0, 1.0]), intr, physics_client_id=cid
    )
    # Clip short of `far`: background pixels saturate the buffer into a wall at that depth.
    pts, colors = backproject_depth(depth, intr, rgb=rgb, max_depth=intr.far * 0.9)
    T_world_cam = np.linalg.inv(view) @ T_OPENGL_OPTICAL
    return transform_points(pts, T_world_cam), colors, T_world_cam


def crop_mask(pts, center=(0.0, 0.0), half=0.15, z_min=0.005):
    """Box around the object, standing in for segmentation.

    A height cut alone is not enough: the ground plane reconstructs with a few mm of error
    at range, leaving far-field points above z_min.
    """
    return (
        (np.abs(pts[:, 0] - center[0]) < half)
        & (np.abs(pts[:, 1] - center[1]) < half)
        & (pts[:, 2] > z_min)
    )


def crop(pts, **kw):
    return pts[crop_mask(pts, **kw)]


# --- tests -------------------------------------------------------------------


def test_depth_buffer_maps_ends_and_between():
    """0 -> near, 1 -> far, strictly increasing between."""
    near, far = 0.1, 2.0
    buf = np.array([0.0, 0.5, 1.0])
    d = depth_buffer_to_linear(buf, near, far)

    assert np.isclose(d[0], near) and np.isclose(d[-1], far)
    assert np.all(np.diff(d) > 0), "depth must grow as the buffer value grows"


def test_edge_mask_flags_depth_step():
    """A flat wall with one step in it: only the pixels straddling the step get dropped."""
    depth = np.full((8, 8), 0.5, np.float32)
    depth[:, 4:] = 0.9

    keep = edge_discontinuity_mask(depth, threshold_m=0.02)
    assert not keep[:, 3].any() and not keep[:, 4].any()  # both sides of the step
    assert keep[:, :3].all() and keep[:, 5:].all()  # flat regions survive
    # A gradient under threshold must not be mistaken for an edge.
    assert edge_discontinuity_mask(np.tile(np.linspace(0.5, 0.55, 8), (8, 1))).all()


def test_backprojection_lands_on_object():
    """A cube reconstructs at the pose it was spawned at."""
    cid = connect()
    try:
        oid = scene(cid)
        _, hi = p.getAABB(oid, physicsClientId=cid)

        pts, _, _ = look_at(np.array([0.4, 0.4, 0.4]), np.array([0.0, 0.0, 0.025]), INTRINSICS, cid)
        obj = crop(pts)

        assert len(obj) > 100
        # One view sees roughly two faces, so the centroid pulls toward the camera.
        assert np.allclose(obj[:, :2].mean(axis=0), 0.0, atol=0.03)
        assert abs(obj[:, 2].max() - hi[2]) < 0.005
    finally:
        p.disconnect(cid)


def check_ycb_object(name: str = DEFAULT_YCB_OBJECT):
    """Same check on a real mesh. Returns (point count, height error)."""
    cid = connect()
    try:
        oid = scene(cid, ycb_urdf(name), pos=(0.0, 0.0, 0.0), seat=True)
        lo, hi = np.array(p.getAABB(oid, physicsClientId=cid))
        center = (lo + hi) / 2

        pts, _, _ = look_at(np.array([0.45, 0.0, 0.25]), center, INTRINSICS, cid)
        obj = crop(pts, center=center[:2], z_min=0.01)
        # Point count tracks projected area, so it varies with object size -- this only
        # catches "reconstructed nothing".
        assert len(obj) > 200, f"only {len(obj)} pts"

        z_err = abs(obj[:, 2].max() - hi[2])
        xy = obj[:, :2].mean(axis=0)
        # getAABB is a padded broadphase box, so it overstates extent: a coarse reference.
        assert z_err < 0.01, f"top off by {z_err * 1e3:.1f}mm"
        assert np.allclose(xy, center[:2], atol=0.05), f"centroid {xy} vs {center[:2]}"
        return len(obj), z_err
    finally:
        p.disconnect(cid)


def test_ycb_object():
    """The default object reconstructs."""
    check_ycb_object()


def test_every_ycb_object_reconstructs():
    """Every vendored object must reconstruct, whatever its size."""
    for name in ycb_names():
        check_ycb_object(name)


# --- debug viz ---------------------------------------------------------------


def draw_cloud(pts, colors, cid: int, n: int = 5000, flat=None, size: int = 2):
    """Draw a subsample of the cloud. `flat` forces one colour instead of the per-point
    ones, which are invisible against the object they came off."""
    idx = np.random.choice(len(pts), min(n, len(pts)), replace=False)
    rgb = np.tile(flat, (len(idx), 1)) if flat is not None else colors[idx] / 255.0
    # Wants float colours, not uint8, and slows badly on a full-resolution cloud.
    p.addUserDebugPoints(pts[idx].tolist(), np.asarray(rgb).tolist(), size, physicsClientId=cid)


def draw_camera(cam_pos, T_world_cam, intr: CameraIntrinsics, cid: int, d: float = 0.3):
    """Axis triad and frustum. Built from our own intrinsics, so points escaping it mean
    the intrinsics disagree with PyBullet's projection."""
    eye = cam_pos.tolist()
    for ax, rgb in enumerate(([1, 0, 0], [0, 1, 0], [0, 0, 1])):
        p.addUserDebugLine(
            eye, (cam_pos + T_world_cam[:3, ax] * 0.1).tolist(), rgb, 3, physicsClientId=cid
        )

    px = np.array([[0, 0], [intr.width, 0], [intr.width, intr.height], [0, intr.height]], np.float32)
    corners = transform_points(
        np.stack([
            (px[:, 0] - intr.cx) * d / intr.fx,
            (px[:, 1] - intr.cy) * d / intr.fy,
            np.full(4, d, np.float32),
        ], axis=-1),
        T_world_cam,
    )
    for i in range(4):
        p.addUserDebugLine(eye, corners[i].tolist(), [1, 1, 0], 1, physicsClientId=cid)
        p.addUserDebugLine(
            corners[i].tolist(), corners[(i + 1) % 4].tolist(), [1, 1, 0], 1, physicsClientId=cid
        )


def hold(cid: int):
    """Keep the window open. Debug items persist, so nothing needs stepping."""
    while p.isConnected(cid):
        time.sleep(1 / 60)


def show(name: str = DEFAULT_YCB_OBJECT):
    """One object with its cloud and camera frustum."""
    cid = connect(gui=True)
    oid = scene(cid, ycb_urdf(name), pos=(0.0, 0.0, 0.0), seat=True)
    lo, hi = np.array(p.getAABB(oid, physicsClientId=cid))
    center = (lo + hi) / 2
    cam = np.array([0.45, 0.0, 0.25])

    pts, colors, T = look_at(cam, center, INTRINSICS, cid)
    # Points land exactly on the mesh surface, so the solid mesh hides them.
    p.changeVisualShape(oid, -1, rgbaColor=[0.6, 0.6, 0.6, 0.25], physicsClientId=cid)

    # Split the budget: the object is a small share of the cloud, so one uniform subsample
    # would spend nearly all of it on the ground plane.
    m = crop_mask(pts, center=center[:2], z_min=0.01)
    draw_cloud(pts[m], colors[m], cid, n=8000, flat=[0.0, 1.0, 0.0], size=3)
    draw_cloud(pts[~m], colors[~m], cid, n=2000)
    draw_camera(cam, T, INTRINSICS, cid)
    p.resetDebugVisualizerCamera(1.0, 50, -25, center.tolist(), physicsClientId=cid)
    hold(cid)


def show_all(gui: bool = True, spacing: float = 0.4, cols: int = 4):
    """Every object in one scene, each with its own capture drawn over it."""
    cid = connect(gui=gui)
    p.loadURDF("plane.urdf", physicsClientId=cid)
    names = ycb_names()

    for i, name in enumerate(names):
        pos = ((i % cols) * spacing, (i // cols) * spacing, 0.0)
        oid = p.loadURDF(ycb_urdf(name), list(pos), physicsClientId=cid)
        seat_on_plane(oid, pos, cid)
        lo, hi = np.array(p.getAABB(oid, physicsClientId=cid))
        center = (lo + hi) / 2

        # Capture before loading the next object: spacing is under one camera standoff, so
        # a neighbour that already existed would sit between this camera and its target.
        cam = center + np.array([0.45, 0.0, 0.2])
        pts, _, T = look_at(cam, center, INTRINSICS, cid)
        p.changeVisualShape(oid, -1, rgbaColor=[0.6, 0.6, 0.6, 0.25], physicsClientId=cid)
        draw_cloud(crop(pts, center=center[:2], z_min=0.01), None, cid, 4000, [0.0, 1.0, 0.0], 3)
        # Short frustum, or neighbouring columns overlap.
        draw_camera(cam, T, INTRINSICS, cid, d=0.12)
        p.addUserDebugText(
            name.replace("Ycb", ""), [pos[0], pos[1], hi[2] + 0.04], [1, 1, 1], 0.7,
            physicsClientId=cid,
        )

    span = np.array([(cols - 1) * spacing, (len(names) // cols) * spacing, 0.0]) / 2
    p.resetDebugVisualizerCamera(2.2, 50, -35, span.tolist(), physicsClientId=cid)
    hold(cid) if gui else p.disconnect(cid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("object", nargs="?", default=DEFAULT_YCB_OBJECT, choices=ycb_names())
    parser.add_argument("--gui", action="store_true", help="show the capture in PyBullet")
    parser.add_argument("--all", action="store_true", help="every object, not just one")
    args = parser.parse_args()

    if args.gui:
        show_all() if args.all else show(args.object)
    else:
        test_depth_buffer_maps_ends_and_between()
        print("ok    test_depth_buffer_maps_ends_and_between")
        test_edge_mask_flags_depth_step()
        print("ok    test_edge_mask_flags_depth_step")
        test_backprojection_lands_on_object()
        print("ok    test_backprojection_lands_on_object")

        for name in ycb_names() if args.all else [args.object]:
            n, z_err = check_ycb_object(name)
            print(f"ok    {name:<20} pts={n:<6} z_err={z_err * 1e3:.1f}mm")
