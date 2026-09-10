"""Unit tests for coverage tracking and surface sampling."""

import numpy as np
import pytest

from nbv_planner.coverage import (
    CoverageTracker,
    build_coverage_colored_mesh,
    load_ycb_mesh,
    sample_surface_points_and_normals,
    transform_mesh,
    ycb_mesh_path,
)
from sim.env import DEFAULT_YCB_OBJECT, ycb_names


def test_ycb_mesh_paths_exist():
    """All vendored YCB objects resolve a valid visual OBJ path."""
    for name in ycb_names():
        path = ycb_mesh_path(name)
        assert path.endswith(".obj")


def test_load_ycb_mesh_and_transform():
    """Load mesh and apply translation + orientation."""
    mesh = load_ycb_mesh(DEFAULT_YCB_OBJECT)
    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0

    pos = np.array([0.5, 0.2, 0.1])
    orn_xyzw = np.array([0.0, 0.0, 0.0, 1.0])
    transformed = transform_mesh(mesh, pos, orn_xyzw)

    assert np.allclose(transformed.vertices.mean(axis=0), mesh.vertices.mean(axis=0) + pos, atol=1e-5)


def test_surface_sampling_and_base_exclusion():
    """Sample points/normals and verify base exclusion threshold."""
    mesh = load_ycb_mesh(DEFAULT_YCB_OBJECT)
    n_requested = 2000
    points, normals = sample_surface_points_and_normals(mesh, n_samples=n_requested)

    assert points.shape == (n_requested, 3)
    assert normals.shape == (n_requested, 3)

    # Unit length normals check
    lengths = np.linalg.norm(normals, axis=-1)
    assert np.allclose(lengths, 1.0, atol=1e-3)

    # Base exclusion check
    mid_z = float(points[:, 2].mean())
    pts_ex, nrm_ex = sample_surface_points_and_normals(mesh, n_samples=n_requested, base_exclusion_z=mid_z)
    assert len(pts_ex) < n_requested
    assert (pts_ex[:, 2] >= mid_z).all()


def test_coverage_tracker_lifecycle():
    """Coverage tracker lifecycle: update, accumulate, fraction, and reset."""
    # Create simple unit sphere surface points
    n_pts = 1000
    phi = np.random.uniform(0, np.pi, n_pts)
    theta = np.random.uniform(0, 2 * np.pi, n_pts)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    surf_pts = np.stack([x, y, z], axis=-1).astype(np.float32)
    surf_nrm = surf_pts.copy()

    tracker = CoverageTracker(surf_pts, surf_nrm, seen_distance_threshold_m=0.01)

    assert tracker.coverage_fraction() == 0.0
    assert len(tracker.get_unseen_points()) == n_pts

    # Empty cloud update
    assert tracker.update(np.zeros((0, 3))) == 0

    # Distant point cloud update (should not mark anything seen)
    distant_cloud = surf_pts[:100] + 5.0
    assert tracker.update(distant_cloud) == 0
    assert tracker.coverage_fraction() == 0.0

    # Exact match on first 200 surface points
    first_batch = surf_pts[:200]
    newly_seen = tracker.update(first_batch)
    assert newly_seen == 200
    assert np.isclose(tracker.coverage_fraction(), 0.2)
    assert len(tracker.get_unseen_points()) == 800
    assert len(tracker.get_seen_points()) == 200

    # Repeat same batch - 0 newly seen
    assert tracker.update(first_batch) == 0

    # Reset
    tracker.reset()
    assert tracker.coverage_fraction() == 0.0
    assert len(tracker.get_unseen_points()) == n_pts


def test_coverage_colored_mesh():
    """Generate colored Open3D triangle mesh from tracker state."""
    mesh = load_ycb_mesh(DEFAULT_YCB_OBJECT)
    pts, nrm = sample_surface_points_and_normals(mesh, n_samples=500)
    tracker = CoverageTracker(pts, nrm, seen_distance_threshold_m=0.01)

    # Mark subset seen
    tracker.update(pts[:100])

    o3d_mesh = build_coverage_colored_mesh(mesh, tracker, base_exclusion_z=None)
    assert len(o3d_mesh.vertices) == len(mesh.vertices)
    assert len(o3d_mesh.triangles) == len(mesh.faces)
    assert len(o3d_mesh.vertex_colors) == len(mesh.vertices)


def test_transform_mesh_applies_urdf_scale():
    """Verify transform_mesh respects scale attributes declared in URDF."""
    mesh_chips_raw = load_ycb_mesh("YcbChipsCan")
    t_mesh_chips = transform_mesh(mesh_chips_raw, np.zeros(3), np.array([0, 0, 0, 1]), obj_name="YcbChipsCan")
    assert np.isclose(t_mesh_chips.extents[0], mesh_chips_raw.extents[0] * 0.95, atol=1e-4)
    assert np.isclose(t_mesh_chips.extents[1], mesh_chips_raw.extents[1] * 0.95, atol=1e-4)
    assert np.isclose(t_mesh_chips.extents[2], mesh_chips_raw.extents[2] * 1.0, atol=1e-4)

    mesh_chef_raw = load_ycb_mesh("YcbMasterChefCan")
    t_mesh_chef = transform_mesh(mesh_chef_raw, np.zeros(3), np.array([0, 0, 0, 1]), obj_name="YcbMasterChefCan")
    assert np.isclose(t_mesh_chef.extents[0], mesh_chef_raw.extents[0] * 0.7, atol=1e-4)
    assert np.isclose(t_mesh_chef.extents[1], mesh_chef_raw.extents[1] * 0.7, atol=1e-4)
    assert np.isclose(t_mesh_chef.extents[2], mesh_chef_raw.extents[2] * 0.7, atol=1e-4)
