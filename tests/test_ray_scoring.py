"""Unit tests for CUDA ray scoring kernel and candidate visibility."""

import time
import numpy as np
import pytest

from nbv_planner.coverage import (
    load_ycb_mesh,
    sample_surface_points_and_normals,
)
from nbv_planner.ray_scoring import score_candidate_views
from nbv_planner.reachability import sample_candidate_camera_poses
from sim.env import DEFAULT_YCB_OBJECT


def test_front_facing_and_backface():
    """Front-facing point is scored visible; back-facing point is culled."""
    # Point at origin with normal pointing +Z
    pts = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    normals = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    triangles = np.zeros((0, 3, 3), dtype=np.float32)  # No occluders

    # Cam 1: +Z looking down at point (+Z normal) -> front-facing
    # Cam 2: -Z looking up at point (+Z normal) -> backface
    cams = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ], dtype=np.float32)

    scores, vis = score_candidate_views(cams, pts, normals, triangles, backface_margin=0.0)
    assert scores[0] == 1
    assert scores[1] == 0
    assert vis[0, 0] == 1
    assert vis[1, 0] == 0


def test_occlusion_blocking():
    """Triangle directly between camera and point blocks visibility."""
    # Point at origin
    pts = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    normals = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)

    # Camera at (0, 0, 2)
    cams = np.array([[0.0, 0.0, 2.0]], dtype=np.float32)

    # Occluding triangle at z=1.0 spanning x,y in [-0.5, 0.5]
    tri_blocking = np.array([
        [[-0.5, -0.5, 1.0], [0.5, -0.5, 1.0], [0.0, 0.5, 1.0]],
    ], dtype=np.float32)

    scores, vis = score_candidate_views(cams, pts, normals, tri_blocking)
    assert scores[0] == 0
    assert vis[0, 0] == 0

    # Non-blocking triangle off to the side at x=10.0
    tri_aside = np.array([
        [[10.0, 0.0, 1.0], [11.0, 0.0, 1.0], [10.0, 1.0, 1.0]],
    ], dtype=np.float32)

    scores_aside, vis_aside = score_candidate_views(cams, pts, normals, tri_aside)
    assert scores_aside[0] == 1
    assert vis_aside[0, 0] == 1


def test_ycb_mustard_scoring_performance():
    """Score 72 candidate views against 3000 surface points on real YCB mesh."""
    mesh = load_ycb_mesh(DEFAULT_YCB_OBJECT)
    triangles = np.asarray(mesh.vertices[mesh.faces], dtype=np.float32)

    pts, normals = sample_surface_points_and_normals(mesh, n_samples=3000)
    cams, _ = sample_candidate_camera_poses(
        np.array([0.0, 0.0, 0.0]), radius=(0.3, 0.5, 2), n_azimuth=36
    )

    t0 = time.perf_counter()
    scores, vis = score_candidate_views(cams, pts, normals, triangles)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert len(scores) == len(cams)
    assert vis.shape == (len(cams), len(pts))
    assert (scores > 0).any()
    assert (scores < len(pts)).all()
    print(f"Scored {len(cams)} views x {len(pts)} points ({len(triangles)} tris) in {elapsed_ms:.2f}ms")


def test_empty_inputs():
    """Handle empty candidates and empty points gracefully."""
    triangles = np.zeros((0, 3, 3), dtype=np.float32)

    # Empty cams
    scores, vis = score_candidate_views(
        np.zeros((0, 3), dtype=np.float32),
        np.ones((10, 3), dtype=np.float32),
        np.ones((10, 3), dtype=np.float32),
        triangles,
    )
    assert len(scores) == 0
    assert vis.shape == (0, 10)

    # Empty points
    scores, vis = score_candidate_views(
        np.ones((5, 3), dtype=np.float32),
        np.zeros((0, 3), dtype=np.float32),
        np.zeros((0, 3), dtype=np.float32),
        triangles,
    )
    assert len(scores) == 5
    assert (scores == 0).all()
    assert vis.shape == (5, 0)
