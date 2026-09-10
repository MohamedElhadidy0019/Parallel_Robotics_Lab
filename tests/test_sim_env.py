"""Scene sanity: everything loaded where we think it did."""

import numpy as np
import pybullet as p

from nbv_core.config import BASE_LINK, EE_LINK
from nbv_core.sim_env import SimEnv


def links(env) -> dict:
    """name -> joint index, for every link on the robot."""
    n = p.getNumJoints(env.robot_id, physicsClientId=env.client_id)
    out = {}
    for i in range(n):
        info = p.getJointInfo(env.robot_id, i, physicsClientId=env.client_id)
        out[info[12].decode()] = i
    return out


def test_robot_loaded_with_ik_links():
    """IK names these links by string -- a rename in vendored shelf_gym breaks IK, not this."""
    env = SimEnv(render=False)
    try:
        have = links(env)
        # The urdf root is a fixed `world` link; base_link hangs off it, so base_link DOES
        # have a joint entry. CuRobo is handed base_link as its chain root, not world.
        for name in (BASE_LINK, EE_LINK):
            assert name in have, f"{name} missing from urdf -- the IK call names it as a string"
        assert len(have) > 10, f"only {len(have)} links -- gripper probably didn't load"
    finally:
        env.close()


def test_table_surface_matches_physics_aabb():
    """The table surface must match the physics engine's AABB top elevation."""
    env = SimEnv(render=False)
    try:
        _, hi = p.getAABB(env.table_id, physicsClientId=env.client_id)
        assert np.isclose(hi[2], env.table_surface_z, atol=1e-4), f"table top at {hi[2]}"
        assert env.table_surface_z > env.base_pose()[0][2], "table should be elevated above Steve base"
    finally:
        env.close()


def test_object_settles_on_the_table():
    """Dropped, not floating and not fallen through -- its base should touch the tabletop."""
    env = SimEnv(render=False)
    try:
        lo, hi = p.getAABB(env.obj_id, physicsClientId=env.client_id)
        assert lo[2] > env.table_top_z - 0.05, f"object bottom at {lo[2]}, table at {env.table_top_z}"
        assert lo[2] < env.table_top_z + 0.05, f"object floating: bottom at {lo[2]}"
        assert hi[2] - lo[2] > 0.03, "object has no height -- wrong mesh?"

        contacts = p.getContactPoints(env.obj_id, env.table_id, physicsClientId=env.client_id)
        assert len(contacts) > 0, "object is not resting on the table"
    finally:
        env.close()


def test_object_did_not_land_on_the_arm():
    env = SimEnv(render=False)
    try:
        hits = p.getContactPoints(env.obj_id, env.robot_id, physicsClientId=env.client_id)
        assert len(hits) == 0, f"object resting on the arm ({len(hits)} contacts)"
    finally:
        env.close()


def test_orbit_shell_stays_within_reach():
    """The whole point of the placement: far-side candidates must still be reachable."""
    env = SimEnv(render=False)
    try:
        base = env.arm_base_pos() if hasattr(env, "arm_base_pos") else env.base_pose()[0]
        reach = env.max_reach()
        d = np.linalg.norm(env.obj_pos[:2] - base[:2])
        r_min, r_max = env.orbit_shell()
        assert r_min < r_max, "empty shell -- no room between object and reach limit"
        assert d + r_max < reach, f"far side sits {d + r_max:.3f}m out, reach is {reach:.3f}"
        assert r_min > env.intrinsics.near, "shell starts inside the camera's near plane"
        # Closer than this and the object overflows the image; those views lose surface.
        assert r_min >= env.framing_distance() - 1e-9, (
            f"shell starts at {r_min:.3f}, object needs {env.framing_distance():.3f} to fit in frame"
        )
    finally:
        env.close()


# --- gui ---------------------------------------------------------------------


def ring(center, radius: float, color, cid: int, n: int = 72, z=None):
    """Horizontal circle drawn as n debug-line segments."""
    a = np.linspace(0, 2 * np.pi, n + 1)
    z = center[2] if z is None else z
    pts = np.stack([center[0] + radius * np.cos(a), center[1] + radius * np.sin(a), np.full(n + 1, z)], -1)
    for i in range(n):
        p.addUserDebugLine(pts[i].tolist(), pts[i + 1].tolist(), color, 1, physicsClientId=cid)


def show():
    """Scene + the reach budget the placement has to satisfy, held until the window closes."""
    import time

    env = SimEnv(render=True)
    cid = env.client_id
    base = env.base_pose()[0]
    obj = env.obj_pos
    reach = env.max_reach()
    r_min, r_max = env.orbit_shell()
    d = np.linalg.norm(obj[:2] - base[:2])

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=cid)
    p.configureDebugVisualizer(rgbBackground=[1, 1, 1], physicsClientId=cid)
    for ax, col in enumerate(([1, 0, 0], [0, 1, 0], [0, 0, 1])):
        tip = base.copy()
        tip[ax] += 0.2
        p.addUserDebugLine(base.tolist(), tip.tolist(), col, 4, physicsClientId=cid)

    ring(base, reach, [1, 0, 0], cid, z=env.table_top_z)  # red: reach limit
    ring(obj, r_min, [0, 0.5, 0.6], cid, z=env.table_top_z)  # teal: orbit shell
    ring(obj, r_max, [0, 0.5, 0.6], cid, z=env.table_top_z)
    p.addUserDebugLine(  # base -> object, the number the placement chooses
        [base[0], base[1], env.table_top_z], [obj[0], obj[1], env.table_top_z], [0.8, 0.5, 0], 3, physicsClientId=cid
    )
    p.addUserDebugText(
        f"base->obj {d:.3f}  shell {r_min:.2f}-{r_max:.2f}  far side {d + r_max:.3f} / {reach:.3f}",
        [obj[0], obj[1], obj[2] + 0.25], [0, 0, 0], 1.0, physicsClientId=cid,
    )
    p.resetDebugVisualizerCamera(2.0, 55, -30, obj.tolist(), physicsClientId=cid)

    while p.isConnected(cid):
        time.sleep(1 / 60)


if __name__ == "__main__":
    import sys

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
