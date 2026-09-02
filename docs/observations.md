# Research & Development Notes

### Problem 1: cuRobo Trajectory Planning
- **Problem:** `Couldn't find solution with 10 attempts, resetting seeds`
- **Cause:** Low-elevation poses near tabletop trigger table obstacle repulsion in cuRobo SDF, trapping optimizer in local minima.
- **Solution:** Use cuRobo `plan_batch` across top candidates + include table collision in initial IK filtering.

### Problem 2: Max Achievable Coverage Identification
- **Problem:** Target cannot reach 100% coverage because bottom surface is occluded by table.
- **Cause:** Base contact points cannot be seen from any reachable camera pose above the table plane.
- **Solution:** Compute exact GPU reachable upper bound at startup (`torch.any(vis_matrix, dim=0).mean()`) or exclude table-contact base.
