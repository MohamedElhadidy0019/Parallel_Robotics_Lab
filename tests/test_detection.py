"""Single-frame tabletop detection on synthetic ray-cast frames. CPU only."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from nbv_planner.camera import CameraIntrinsics
from nbv_planner.detection import TablePlane, detect_object, fit_table_plane, table_aligned_box
from nbv_planner.observations import Observation
from nbv_planner.reachability import camera_lookat_quaternion_xyzw

TABLE_Z = 0.75
TABLE_HALF = 0.4
BOX_CENTER = np.array([0.8, 0.05, TABLE_Z + 0.1])
BOX_SIZE = np.array([0.10, 0.06, 0.20])
INTRINSICS = CameraIntrinsics(width=320, height=240)


def box_ray_hits(origin, directions, center, size):
    lo, hi = center - size / 2, center + size / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (lo - origin) / directions, (hi - origin) / directions
    t_near = np.nanmax(np.minimum(t1, t2), axis=1)
    t_far = np.nanmin(np.maximum(t1, t2), axis=1)
    return np.where((t_near <= t_far) & (t_far > 0), t_near, np.inf)


def render(eye, box_center=BOX_CENTER, box_size=BOX_SIZE, with_box=True):
    world_from_camera = np.eye(4)
    world_from_camera[:3, :3] = Rotation.from_quat(camera_lookat_quaternion_xyzw(eye, box_center)).as_matrix()
    world_from_camera[:3, 3] = eye

    v, u = np.mgrid[0:INTRINSICS.height, 0:INTRINSICS.width]
    rays_camera = np.stack([(u - INTRINSICS.cx) / INTRINSICS.fx, (v - INTRINSICS.cy) / INTRINSICS.fy,
                            np.ones_like(u, dtype=float)], axis=-1).reshape(-1, 3)
    rays_world = rays_camera @ world_from_camera[:3, :3].T

    with np.errstate(divide="ignore", invalid="ignore"):
        t_table = (TABLE_Z - eye[2]) / rays_world[:, 2]
        hit = eye + t_table[:, None] * rays_world
    on_table = (t_table > 0) & np.all(np.abs(hit[:, :2] - [0.8, 0.0]) <= TABLE_HALF, axis=1)
    t = np.where(on_table, t_table, np.inf)
    if with_box:
        t = np.minimum(t, box_ray_hits(eye, rays_world, box_center, box_size))

    depth = np.where(np.isfinite(t), t, 0.0).reshape(INTRINSICS.height, INTRINSICS.width)
    rgb = np.zeros((INTRINSICS.height, INTRINSICS.width, 3), dtype=np.uint8)
    return Observation(rgb, depth.astype(np.float32), INTRINSICS, world_from_camera, 0.0)


def box_surface_points(center, size, yaw, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    local = rng.uniform(-0.5, 0.5, (n, 3)) * size
    face = rng.integers(0, 3, n)
    local[np.arange(n), face] = np.sign(local[np.arange(n), face]) * size[face] / 2
    return local @ Rotation.from_euler("z", yaw).as_matrix().T + center


FLAT_TABLE = TablePlane(np.array([0.0, 0.0, 1.0]), -TABLE_Z, np.zeros((0, 3)))


def test_fit_table_plane_ignores_object_points():
    rng = np.random.default_rng(0)
    table = np.c_[rng.uniform(0.4, 1.2, (20000, 2)), np.full(20000, TABLE_Z)]
    points = np.vstack([table, box_surface_points(BOX_CENTER, BOX_SIZE, 0.0)])

    plane = fit_table_plane(points)

    assert plane.normal[2] == pytest.approx(1.0, abs=1e-3)
    assert plane.z_at(BOX_CENTER[:2]) == pytest.approx(TABLE_Z, abs=2e-3)


def test_fit_table_plane_rejects_vertical_planes():
    rng = np.random.default_rng(0)
    wall = np.c_[np.full(5000, 1.0), rng.uniform(-0.5, 0.5, (5000, 2))]
    with pytest.raises(ValueError):
        fit_table_plane(wall, max_planes=1)


def test_table_aligned_box_recovers_rotated_box():
    points = box_surface_points(BOX_CENTER, BOX_SIZE, np.radians(30))

    box = table_aligned_box(points, FLAT_TABLE)

    np.testing.assert_allclose(box.center, BOX_CENTER, atol=2e-3)
    np.testing.assert_allclose(box.size, BOX_SIZE, atol=2e-3)
    assert box.yaw == pytest.approx(np.radians(30), abs=np.radians(1))


def test_detect_object_from_top_view_recovers_box():
    observation = render(eye=BOX_CENTER + [-0.15, 0.0, 0.45])

    detection = detect_object(observation)

    np.testing.assert_allclose(detection.box.center, BOX_CENTER, atol=0.01)
    np.testing.assert_allclose(np.sort(detection.box.size[:2]), np.sort(BOX_SIZE[:2]), atol=0.01)
    assert detection.box.size[2] == pytest.approx(BOX_SIZE[2], abs=0.01)
    assert np.all(detection.table.height(detection.points_world) > 0.0)


def test_detect_object_prompts_segmenter_with_object_box():
    observation = render(eye=BOX_CENTER + [-0.3, 0.0, 0.3])

    class RecordingSegmenter:
        def segment(self, rgb, box):
            self.box = box
            mask = np.zeros(rgb.shape[:2], dtype=bool)
            mask[box[1]:box[3] + 1, box[0]:box[2] + 1] = True
            return mask

    segmenter = RecordingSegmenter()
    detection = detect_object(observation, segmenter)

    x0, y0, x1, y1 = segmenter.box
    assert detection.prompt_box == segmenter.box
    assert 0 < x0 < x1 < INTRINSICS.width - 1 and 0 < y0 < y1 < INTRINSICS.height - 1
    assert detection.box.center[2] == pytest.approx(BOX_CENTER[2], abs=0.02)


def test_detect_object_on_empty_table_raises():
    with pytest.raises(ValueError):
        detect_object(render(eye=BOX_CENTER + [-0.3, 0.0, 0.3], with_box=False))
