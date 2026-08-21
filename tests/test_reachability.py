"""Candidate sampling, the base-link frame change, and the caches on disk.

Pure math and file reads -- no robot, no sim, no GPU.
"""

import os

import numpy as np
from scipy.spatial.transform import Rotation

from nbv_core.reachability import (
    cache_path_for,
    camera_lookat_quaternion_xyzw,
    load_reachability_cache,
    quaternion_xyzw_to_wxyz,
    sample_candidate_camera_poses,
    world_poses_to_base_link_frame,
)
from nbv_core.sim_env import ycb_names

# Arbitrary. These tests check relationships that hold for any object position.
OBJ = np.array([0.4, 0.3, 1.0])

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")


def test_lookat_points_at_target():
    """+Z of the returned frame must aim from eye to target."""
    eye = OBJ + np.array([0.4, 0.2, 0.3])
    R = Rotation.from_quat(camera_lookat_quaternion_xyzw(eye, OBJ)).as_matrix()

    want = (OBJ - eye) / np.linalg.norm(OBJ - eye)
    assert np.allclose(R[:, 2], want, atol=1e-6)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-6)  # orthonormal
    assert np.isclose(np.linalg.det(R), 1.0)  # right-handed, not mirrored


def test_lookat_handles_vertical_view():
    """Straight up: +Z world-up would degenerate, so it must fall back to +Y."""
    R = Rotation.from_quat(camera_lookat_quaternion_xyzw(OBJ - np.array([0.0, 0.0, 1.0]), OBJ)).as_matrix()

    assert np.allclose(R[:, 2], [0.0, 0.0, 1.0], atol=1e-6)  # still aims at the target
    assert np.allclose(R, np.eye(3), atol=1e-6)  # fallback gives x=+X, y=+Y, z=+Z


def test_quaternion_reorder():
    assert np.allclose(quaternion_xyzw_to_wxyz(np.array([1.0, 2.0, 3.0, 4.0])), [4.0, 1.0, 2.0, 3.0])


# --- candidate sampling ------------------------------------------------------


def test_candidates_sit_on_the_shell():
    t, q = sample_candidate_camera_poses(OBJ, radius=(0.3, 0.5, 2), n_azimuth=36)

    assert len(t) == 2 * 36 * 3
    assert len(q) == len(t)

    r = np.linalg.norm(t - OBJ, axis=1)
    assert r.min() > 0.3 - 1e-6 and r.max() < 0.5 + 1e-6
    assert (t[:, 2] > OBJ[2]).all(), "phi range is above horizontal, so every candidate is too"
    # Azimuth sweeps a full circle, so candidates surround the object.
    assert (t[:, 0] > OBJ[0]).any() and (t[:, 0] < OBJ[0]).any()
    assert (t[:, 1] > OBJ[1]).any() and (t[:, 1] < OBJ[1]).any()


def test_candidates_all_look_at_object():
    t, q = sample_candidate_camera_poses(OBJ, radius=(0.3, 0.5, 1),
                                        elevation_deg=(35.0, 35.0, 1), n_azimuth=8)
    for ti, qi in zip(t, q):
        fwd = Rotation.from_quat(qi).as_matrix()[:, 2]
        want = (OBJ - ti) / np.linalg.norm(OBJ - ti)
        assert np.allclose(fwd, want, atol=1e-6)


def test_z_min_filter_drops_low_candidates():
    z = OBJ[2] + 0.15
    t, _ = sample_candidate_camera_poses(OBJ, radius=(0.3, 0.5, 2), z_min_world=z)
    assert len(t) > 0 and (t[:, 2] >= z).all()


# --- world -> base_link ------------------------------------------------------


def test_base_frame_removes_the_mount_offset():
    """A base mounted above the origin: in its own frame that offset must be gone."""
    t_base, q_base = np.array([0.0, 0.0, 0.9]), np.array([0.0, 0.0, 0.0, 1.0])
    t_w = np.array([[0.3, 0.2, 1.2]])
    q_w = np.array([[0.0, 0.0, 0.0, 1.0]])

    t_out, _ = world_poses_to_base_link_frame(t_w, q_w, t_base, q_base)
    assert np.allclose(t_out[0], [0.3, 0.2, 0.3], atol=1e-6)


def test_base_frame_handles_rotated_base():
    """Round-trip: a pose expressed in the base frame, pushed back out, returns to world."""
    t_base = np.array([0.1, -0.2, 0.9])
    q_base = Rotation.from_euler("z", 0.7).as_quat()
    t_w = np.array([[0.4, 0.3, 1.3], [0.0, 0.0, 1.0]])
    q_w = np.stack([Rotation.from_euler("y", 0.3).as_quat()] * 2)

    t_out, q_out = world_poses_to_base_link_frame(t_w, q_w, t_base, q_base)
    R = Rotation.from_quat(q_base)
    assert np.allclose(R.apply(t_out) + t_base, t_w, atol=1e-6)
    assert np.allclose((R * Rotation.from_quat(q_out)).as_matrix(),
                       Rotation.from_quat(q_w).as_matrix(), atol=1e-6)


# --- the cache nbv_core.reachability wrote -----------------------------------


def each_cache():
    """Every cache on disk, as (object name, contents)."""
    found = [(n, cache_path_for(n)) for n in ycb_names()]
    found = [(n, path) for n, path in found if os.path.exists(path)]
    assert found, "no caches built -- run: python -m nbv_core.reachability --all"
    return [(n, load_reachability_cache(path)) for n, path in found]


def test_cache_has_every_field_the_planner_reads():
    for name, c in each_cache():
        for key in ("t_candidates_world", "q_candidates_world_xyzw", "reachable", "q_joints",
                    "urdf_path", "base_link", "ee_link"):
            assert key in c, f"{name}: missing {key}"
        assert str(c["ee_link"]) == "dummy_camera_link", f"{name}: solved for the wrong link"


def test_cache_rows_line_up():
    """Every array indexes the same candidate list, so one mask can select across all of them."""
    for name, c in each_cache():
        n = len(c["t_candidates_world"])
        assert len(c["q_candidates_world_xyzw"]) == n, name
        assert len(c["reachable"]) == n, name
        assert len(c["q_joints"]) == n, name
        assert c["reachable"].dtype == bool, name
        assert c["q_joints"].shape[1] == len(ARM_JOINTS), f"{name}: unexpected joint count"


def test_cache_is_usable():
    """A cache where almost nothing is reachable means the placement or frames are wrong."""
    for name, c in each_cache():
        frac = c["reachable"].mean()
        assert frac > 0.5, f"{name}: only {frac:.0%} reachable -- suspect the base_link transform"


def test_unreachable_rows_are_zeroed():
    """So an unreachable row can't be mistaken for a valid joint config downstream."""
    for name, c in each_cache():
        bad = c["q_joints"][~c["reachable"]]
        assert bad.size == 0 or np.all(bad == 0.0), name
        good = c["q_joints"][c["reachable"]]
        assert good.size == 0 or not np.all(good == 0.0), f"{name}: reachable rows have no solution"


def test_cached_poses_all_aim_at_one_point():
    """Every candidate looks at the object, so their view rays must meet there.

    Solves for the point nearest all the rays; positions and orientations saved out of
    step would stop converging.
    """
    for name, c in each_cache():
        t = c["t_candidates_world"].astype(np.float64)
        quats = c["q_candidates_world_xyzw"]
        fwd = np.array([Rotation.from_quat(q).as_matrix()[:, 2] for q in quats])

        # Summing (I - d d^T) projects out each ray's own direction.
        proj = np.eye(3) - fwd[:, :, None] * fwd[:, None, :]
        focus = np.linalg.solve(proj.sum(0), np.einsum("nij,nj->i", proj, t))

        to_focus = focus - t
        to_focus /= np.linalg.norm(to_focus, axis=1, keepdims=True)
        aim = np.einsum("ij,ij->i", fwd, to_focus)
        assert np.all(aim > 0.999), f"{name}: cameras do not share a target"
        assert np.linalg.norm(t - focus, axis=1).min() > 0.0, f"{name}: focus sits on the shell"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"ok    {name}")
            except AssertionError as e:
                print(f"FAIL  {name}  {e}")
