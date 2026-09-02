"""Reachability + motion planning against a live sim: robot, object, GPU.

Kept apart from test_reachability.py, which is pure math and runs anywhere.
These build the IK reachability from scratch and drive the arm, so they are slow.

    python tests/test_reachability_integration.py
    python tests/test_reachability_integration.py --gui
"""

import os
import sys

import numpy as np

from nbv_core.config import BASE_LINK, EE_LINK
from nbv_core.reachability import (
    ik_filter,
    sample_candidate_camera_poses,
)
from nbv_core.sim_env import ASSET_PATH, SimEnv, ycb_names


def candidates_for(env) -> tuple[np.ndarray, np.ndarray]:
    """The same shell nbv_core.reachability builds, straight off a live env."""
    r_min, r_max = env.orbit_shell()
    return sample_candidate_camera_poses(
        env.obj_pos, radius=(r_min, r_max, 2), n_azimuth=36, z_min_world=env.table_top_z
    )

URDF_PATH = os.path.join(ASSET_PATH, "ur5_robotiq_85.urdf")


def reachability_for(env) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample the shell off a live env and IK-filter it. Returns (t, q, reachable, q_joints)."""
    t, q = candidates_for(env)
    t_base, q_base = env.base_pose()
    reachable, q_joints = ik_filter(URDF_PATH, BASE_LINK, EE_LINK, t, q, t_base, q_base)
    return t, q, reachable, q_joints


def test_candidates_clear_the_real_object():
    """No candidate may sit inside the object or under the table."""
    env = SimEnv(render=False)
    try:
        t, _ = candidates_for(env)
        assert len(t) > 0
        assert (t[:, 2] >= env.table_top_z).all(), "candidate below the tabletop"
        r = np.linalg.norm(t - env.obj_pos, axis=1)
        assert r.min() >= env.object_radius(), "candidate inside the object"
    finally:
        env.close()


def test_candidates_frame_the_real_object():
    """Every standoff must be far enough that the object fits in the image."""
    env = SimEnv(render=False)
    try:
        t, _ = candidates_for(env)
        r = np.linalg.norm(t - env.obj_pos, axis=1)
        assert r.min() >= env.framing_distance() - 1e-6, (
            f"closest candidate {r.min():.3f}, needs {env.framing_distance():.3f} to frame it"
        )
    finally:
        env.close()


def test_every_ycb_object_gives_a_usable_shell():
    """Any of the 12 must produce a shell that is non-empty, framed, and within reach."""
    for name in ycb_names():
        env = SimEnv(render=False, ycb_object=name)
        try:
            r_min, r_max = env.orbit_shell()
            d = np.linalg.norm(env.obj_pos[:2] - env.base_pose()[0][:2])
            assert r_min < r_max, f"{name}: empty shell"
            assert r_min >= env.framing_distance() - 1e-9, f"{name}: shell too close to frame it"
            assert d + r_max < env.max_reach(), f"{name}: far side out of reach"
        finally:
            env.close()


def test_enough_candidates_are_reachable():
    """Most of the shell should be reachable; a bad base_link transform would kill most of it."""
    env = SimEnv(render=False)
    try:
        _, _, reachable, q_joints = reachability_for(env)
        assert len(reachable) > 0
        frac = reachable.mean()
        assert frac > 0.5, f"only {frac:.0%} reachable -- suspect the base_link transform"
        assert not np.all(q_joints[reachable] == 0.0), "reachable rows have no solution"
        assert np.all(q_joints[~reachable] == 0.0), "unreachable rows are not zeroed"
    finally:
        env.close()


def test_ik_joints_put_the_camera_where_promised():
    """IK solved the poses we asked for, in the right frame.

    A wrong base_link transform would put the camera ~tens of cm away from the candidate.
    """
    env = SimEnv(render=False)
    try:
        t, _, reachable, q_joints = reachability_for(env)
        idx = np.where(reachable)[0]
        assert len(idx) > 0
        for i in idx[::max(1, len(idx) // 8)]:
            for j, qi in zip(env.arm_joint_indices, q_joints[i]):
                env._p.resetJointState(env.robot_id, j, float(qi), physicsClientId=env.client_id)
            t_cam = np.array(
                env._p.getLinkState(env.robot_id, env.camera_link, physicsClientId=env.client_id)[0]
            )
            assert np.linalg.norm(t_cam - t[i]) < 0.02, (
                f"camera {np.round(t_cam, 3)} vs target {np.round(t[i], 3)}"
            )
    finally:
        env.close()


def test_motion_planner_reaches_a_reachable_candidate():
    """The full loop: IK → plan → execute → camera lands on the target."""
    from nbv_core.motion_planning import move_camera_to

    env = SimEnv(render=False)
    try:
        t, q, reachable, _ = reachability_for(env)
        idx = np.where(reachable)[0]
        assert len(idx) > 0, "no reachable candidates at all"

        cam = np.array(
            env._p.getLinkState(env.robot_id, env.camera_link, physicsClientId=env.client_id)[0]
        )
        order = idx[np.argsort(np.linalg.norm(t[idx] - cam, axis=1))]
        for attempt, i in enumerate(order[:6]):
            ok, t_achieved = move_camera_to(env, t[i], q[i].tolist())
            if ok:
                assert np.linalg.norm(t_achieved - t[i]) < 0.02
                return
        raise AssertionError(
            f"no plan succeeded in {attempt + 1} attempts from nearest candidates"
        )
    finally:
        env.close()


def test_reachable_set_surrounds_the_object():
    """Reachable candidates span a wide azimuth arc, not one narrow slice."""
    env = SimEnv(render=False)
    try:
        t, _, reachable, _ = reachability_for(env)
        idx = np.where(reachable)[0]
        assert len(idx) > 0
        angles = np.arctan2(t[idx, 1] - env.obj_pos[1], t[idx, 0] - env.obj_pos[0])
        bins, _ = np.histogram(angles % (2 * np.pi), bins=8, range=[0, 2 * np.pi])
        # Must cover at least half the circle.
        assert (bins > 0).sum() > 4, f"only {(bins > 0).sum()}/8 azimuth bins have reachable poses"
    finally:
        env.close()


# --- gui ---------------------------------------------------------------------


def show() -> None:
    """Live view: IK → motion-plan to a reachable candidate, then run the test suite."""
    import time

    from nbv_core.motion_planning import move_camera_to

    env = SimEnv(render=True, ycb_object=sys.argv[2] if len(sys.argv) > 2 else "YcbMustardBottle")
    cid = env.client_id
    p = env._p
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=cid)
    p.resetDebugVisualizerCamera(2.0, 55, -30, env.obj_pos.tolist(), physicsClientId=cid)

    t, q, reachable, _ = reachability_for(env)
    idx = np.where(reachable)[0]
    if len(idx) == 0:
        print("no reachable candidates")
        return

    # Draw the reachable shell
    for i in idx[:: len(idx) // 36 + 1]:
        p.addUserDebugLine(t[i].tolist(), env.obj_pos.tolist(), [0, 1, 0], 0.3, physicsClientId=cid)

    cam = np.array(p.getLinkState(env.robot_id, env.camera_link, physicsClientId=cid)[0])
    order = idx[np.argsort(np.linalg.norm(t[idx] - cam, axis=1))]

    for attempt, i in enumerate(order[:5]):
        targ = t[i]
        p.addUserDebugText(f"> {attempt+1}", targ.tolist(), [1, 1, 0], 1.2, physicsClientId=cid)
        ok, t_achieved = move_camera_to(env, targ, q[i].tolist())
        status = "REACHED" if ok else f"FAILED ({np.linalg.norm(t_achieved - targ):.3f}m)"
        print(f"  {'[PASS]' if ok else '[FAIL]'} candidate {attempt+1}: {status}")
        p.removeAllUserDebugItems(physicsClientId=cid)
        if ok:
            break

    print("holding window: close it to end")
    while p.isConnected(cid):
        time.sleep(1 / 60)
    env.close()


if __name__ == "__main__":
    if "--gui" in sys.argv:
        show()
    else:
        for name, fn in sorted(globals().items()):
            if name.startswith("test_"):
                try:
                    fn()
                    print(f"ok    {name}")
                except AssertionError as e:
                    print(f"FAIL  {name}  {e}")
