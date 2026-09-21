"""Multi-view refinement of the object box and the ring of scan views. CPU only."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from nbv_planner.detection import TableBox, detect_object
from nbv_planner.object_estimate import ObjectEstimate, mesh_alignment
from nbv_planner.scan_views import ring_radius, ring_viewpoints
from tests.test_detection import BOX_CENTER, BOX_SIZE, INTRINSICS, render


def view_from(azimuth_deg: float, distance: float = 0.45, elevation_deg: float = 40.0) -> np.ndarray:
    azimuth, elevation = np.radians(azimuth_deg), np.radians(elevation_deg)
    return BOX_CENTER + distance * np.array([np.cos(elevation) * np.cos(azimuth),
                                             np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])


def test_level_view_misses_depth_and_more_views_recover_it():
    estimate = ObjectEstimate()
    first = estimate.add(detect_object(render(view_from(180.0, elevation_deg=0.0))))
    assert first.size[:2].min() < 0.01

    for azimuth in (90.0, 0.0, -90.0):
        estimate.add(detect_object(render(view_from(azimuth)), prior=estimate.box))

    np.testing.assert_allclose(estimate.box.center, BOX_CENTER, atol=0.005)
    np.testing.assert_allclose(np.sort(estimate.box.size[:2]), np.sort(BOX_SIZE[:2]), atol=0.006)
    assert estimate.box.size[2] == pytest.approx(BOX_SIZE[2], abs=0.006)


def test_estimate_converges_when_views_agree():
    estimate = ObjectEstimate()
    for azimuth in (180.0, 90.0, 0.0, -90.0, -90.0):
        estimate.add(detect_object(render(view_from(azimuth)), prior=estimate.box))
    assert estimate.converged
    assert len(estimate.boxes) == 5


def test_prior_box_rejects_points_far_from_the_object():
    far_box = TableBox(center=BOX_CENTER + [0.3, 0.3, 0.0], size=BOX_SIZE, yaw=0.0)
    with pytest.raises(ValueError):
        detect_object(render(view_from(180.0)), prior=far_box)


def test_ring_viewpoints_look_at_box_and_skip_the_start_side():
    box = TableBox(center=BOX_CENTER, size=BOX_SIZE, yaw=0.0)
    start = view_from(180.0)

    slots, positions, quaternions = ring_viewpoints(box, INTRINSICS, start, n_views=8)

    assert set(slots) == set(range(1, 9))
    forward = Rotation.from_quat(quaternions).as_matrix()[:, :, 2]
    to_center = (BOX_CENTER - positions) / np.linalg.norm(BOX_CENTER - positions, axis=1, keepdims=True)
    np.testing.assert_allclose(np.einsum("ij,ij->i", forward, to_center), 1.0, atol=1e-6)
    assert np.all(np.linalg.norm(positions - BOX_CENTER, axis=1) >= ring_radius(box, INTRINSICS) - 1e-9)

    first_slot = positions[slots == 1][0] - BOX_CENTER
    start_offset = start - BOX_CENTER
    angle = np.degrees(np.arccos(first_slot[:2] @ start_offset[:2]
                                 / np.linalg.norm(first_slot[:2]) / np.linalg.norm(start_offset[:2])))
    assert angle == pytest.approx(360.0 / 9, abs=0.5)


def test_mesh_alignment_accepts_matching_mesh_and_rejects_shifted_one():
    import trimesh

    estimate = ObjectEstimate()
    for azimuth in (180.0, 90.0, 0.0, -90.0):
        estimate.add(detect_object(render(view_from(azimuth)), prior=estimate.box))

    matching = trimesh.creation.box(extents=BOX_SIZE, transform=trimesh.transformations.translation_matrix(BOX_CENTER))
    shifted = trimesh.creation.box(extents=BOX_SIZE, transform=trimesh.transformations.translation_matrix(BOX_CENTER + [0.03, 0.0, 0.0]))

    assert mesh_alignment(estimate, matching).aligned()
    alignment = mesh_alignment(estimate, shifted)
    assert not alignment.aligned()
    assert alignment.center_error == pytest.approx(0.03, abs=0.005)
