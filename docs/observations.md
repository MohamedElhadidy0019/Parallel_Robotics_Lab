# Notes: cuRobo Trajectory Planning

- **Problem:** `Couldn't find solution with 10 attempts, resetting seeds`
- **Cause:** Low-elevation poses near tabletop trigger table obstacle repulsion in cuRobo SDF, trapping optimizer in local minima.
- **Solution:** Use cuRobo `plan_batch` across top-5 candidates + include table collision in initial IK filtering.
