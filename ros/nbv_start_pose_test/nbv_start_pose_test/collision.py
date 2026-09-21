"""Name overlapping self-collision spheres without disabling any collision pairs."""

import numpy as np

from nbv_planner.motion_planning import robot_spheres_world


def self_collision_details(robot, joints):
    cfg = robot.robot_config["robot_cfg"]["kinematics"]
    spheres = robot_spheres_world(joints, np.zeros(3), np.array([0., 0., 0., 1.]))[0]
    names = []
    radii = []
    for name in cfg["collision_link_names"]:
        for sphere in cfg["collision_spheres"][name]:
            names.append(name)
            # cuRobo removes the world inflation when checking self-collision.
            radii.append(sphere["radius"] + cfg.get("self_collision_buffer", {}).get(name, 0.0))
    if len(names) != len(spheres):
        return "sphere/link count mismatch; inspect cuRobo collision geometry"
    ignores = cfg.get("self_collision_ignore", {})
    overlaps = {}
    for i, name in enumerate(names):
        if radii[i] <= 0:
            continue
        for j in range(i + 1, len(names)):
            other = names[j]
            if radii[j] <= 0 or name == other or other in ignores.get(name, []) or name in ignores.get(other, []):
                continue
            penetration = radii[i] + radii[j] - np.linalg.norm(spheres[i, :3] - spheres[j, :3])
            if penetration > 0:
                pair = tuple(sorted([name, other]))
                overlaps[pair] = max(overlaps.get(pair, 0.0), float(penetration))
    return "; ".join(f"{a} / {b}: {depth * 1000:.1f} mm sphere overlap"
                     for (a, b), depth in sorted(overlaps.items(), key=lambda item: -item[1])[:8]) or "no sphere overlaps found"
