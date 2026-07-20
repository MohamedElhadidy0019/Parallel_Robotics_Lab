"""
Diagnostic: checks whether the mustard bottle's pose (position + orientation)
changes while the arm moves through the fixed 8-view orbit.

Why: combine_mustard_pointclouds.py shows "ghosting" (multiple offset copies
of a similar shape) when merging the 8 per-view point clouds, even though it
contains no voxel-grid code at all - it's a plain concatenation of already-
saved point clouds. That means the misalignment must come from upstream: if
the object itself moves/rotates while the arm glides between viewpoints, each
view would be capturing it at a genuinely different pose, which would produce
exactly this kind of ghosting even though each view's own capture math is
correct. This script isolates that question - no capturing/rendering, just
pose-tracking - so it runs fast.

Usage: python check_object_stability.py
"""
import numpy as np

from nbv_environment import NBVEnv2


def main() -> None:
    env = NBVEnv2(render=False)

    initial_pos, initial_orn = env._p.getBasePositionAndOrientation(env.obj_id)
    print(f"Object pose before scan: pos={np.round(initial_pos, 4)} orn={np.round(initial_orn, 4)}\n")

    def check_pose(view_index: int) -> None:
        pos, orn = env._p.getBasePositionAndOrientation(env.obj_id)
        drift_mm = np.linalg.norm(np.array(pos) - np.array(initial_pos)) * 1000
        print(f"View {view_index}: pos={np.round(pos, 4)} orn={np.round(orn, 4)} "
              f"(drifted {drift_mm:.2f} mm from start)")

    env.move_through_orbit(n_views=8, height=1.2, on_stop=check_pose)

    final_pos, final_orn = env._p.getBasePositionAndOrientation(env.obj_id)
    total_drift_mm = np.linalg.norm(np.array(final_pos) - np.array(initial_pos)) * 1000
    print(f"\nTotal drift over the whole scan: {total_drift_mm:.2f} mm")
    if total_drift_mm > 2.0:
        print("-> The object moved a meaningful amount during the scan. This is very "
              "likely the cause of the ghosting - each view saw it in a different pose.")
    else:
        print("-> The object barely moved. Ghosting is probably NOT caused by object "
              "drift - look at per-view camera pose/back-projection instead.")


if __name__ == "__main__":
    main()
