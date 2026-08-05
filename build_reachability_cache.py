"""
Step A of the NBV plan: builds and caches a CuRobo IK-reachability map over
a dense hemisphere-shell of candidate camera poses around the object, so
later NBV planning (nbv_planner.py) loads this instead of solving IK live
per candidate every run - the TA's "IK prefilter, precomputed/cached"
instruction.

Uses nbv_core.reachability for the actual sampling + CuRobo batch-IK logic
(validated against the real ur5_robotiq_85 URDF the same way as
curobo_ur5_ik_test.py); this script just wires: get the object's real
settled pose + safe orbit radius from a real (headless) NBVEnv2 instance,
sample candidates around it, then hand off to CuRobo.

Run once (or whenever the object placement / robot URDF changes) via:
    conda run -n rob_env python build_reachability_cache.py
"""
import os

import numpy as np

from nbv_core.reachability import build_and_save_reachability_cache, sample_candidate_camera_poses
from nbv_environment import ASSET_PATH, TABLE_TOP_Z, NBVEnv2

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reachability")
CACHE_PATH = os.path.join(CACHE_DIR, "reachability_cache.npz")

URDF_PATH = os.path.join(ASSET_PATH, "ur5_robotiq_85.urdf")
BASE_LINK = "base_link"
EE_LINK = "dummy_camera_link"

RADIUS_MARGIN_M = 0.30  # extends the fixed single-radius orbit (move_through_orbit) into a shell this wide
N_RADIUS = 2
N_THETA = 36    # azimuth is where distinct views actually come from (self-occlusion geometry changes fastest
                # with azimuth) - dense here, matching move_through_orbit's efficient wide azimuth-only sweep
N_PHI = 3       # radius/elevation mostly just zoom/tilt the same view - sparse here on purpose (see project
                # memory: an earlier 3x24x5 grid wasted most of its 172 reachable candidates on near-redundant
                # radius/elevation variants of the same azimuth, needing 5-8x more views than the fixed orbit
                # for comparable coverage)
PHI_MIN_DEG = 20.0
PHI_MAX_DEG = 55.0  # centered on ~35deg, the fixed orbit's own (well-performing) elevation


def main() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)

    env = NBVEnv2(render=False)
    t_obj_world = env.obj_pos
    r_min = env._compute_safe_orbit_radius()
    t_base_world_tuple, q_base_world_xyzw_tuple = env._p.getBasePositionAndOrientation(env.robot_id)
    t_base_world = np.array(t_base_world_tuple)
    q_base_world_xyzw = np.array(q_base_world_xyzw_tuple)
    env.close()

    r_max = r_min + RADIUS_MARGIN_M
    t_candidates_world, q_candidates_world_xyzw = sample_candidate_camera_poses(
        t_obj_world, r_min, r_max,
        n_radius=N_RADIUS, n_theta=N_THETA,
        phi_min_deg=PHI_MIN_DEG, phi_max_deg=PHI_MAX_DEG, n_phi=N_PHI,
        z_min_world=TABLE_TOP_Z,
    )
    print(f"Sampled {len(t_candidates_world)} candidate camera poses "
          f"(r in [{r_min:.3f}, {r_max:.3f}]m, {N_THETA} azimuths, {N_PHI} elevations)")

    build_and_save_reachability_cache(
        URDF_PATH, BASE_LINK, EE_LINK,
        t_candidates_world, q_candidates_world_xyzw,
        t_base_world, q_base_world_xyzw, CACHE_PATH,
    )


if __name__ == "__main__":
    main()
