"""
Step D of the NBV plan: ties Step A (cached CuRobo reachability - see
build_reachability_cache.py / nbv_core.reachability), Step B (known-CAD
coverage tracking - nbv_core.coverage) and Step C (GPU ray-cast scoring -
nbv_core.ray_scoring) into the real greedy next-best-view loop.

select_next_view_pose() has the EXACT signature full_pipeline.py's
select_next_view_pose swap-point stub already anticipated -
(env, accumulated_points_so_far, view_index) -> (t_target_world,
q_target_world) | None - so wiring it in (Step E) is a straight
substitution; run_scan_and_reconstruct()'s while-loop shape does not change.

Per-env planner state (loaded reachability cache, coverage tracker, occluder
mesh triangles, which reachable candidates have already been visited) is
built once on the first call for a given env and reused on every later call.
Kept out of the public function's signature via a WeakKeyDictionary keyed by
the env instance, not a plain module global, so multiple envs (e.g. two
scans in the same process, or this project's own validation scripts) don't
stomp on each other's state and a closed/discarded env's planner state is
garbage-collected naturally instead of leaking.
"""
import os
import weakref

import numpy as np

from nbv_core.coverage import (
    CoverageTracker, MUSTARD_MESH_PATH_COARSE, load_object_mesh_world, sample_surface_points_and_normals,
)
from nbv_core.reachability import load_reachability_cache
from nbv_core.ray_scoring import score_candidate_views

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REACHABILITY_CACHE_PATH = os.path.join(PROJECT_ROOT, "reachability", "reachability_cache.npz")

N_SURFACE_SAMPLES = 5000
BASE_EXCLUSION_HEIGHT_M = 0.015  # bottom 15mm of the object (world z) is excluded from the coverage
                                  # denominator entirely - the table-contact face and its immediate
                                  # transition curve, which no above-table camera can ever see (the object
                                  # rests on an opaque table; this isn't a planner/reachability shortfall,
                                  # it's a physical constraint of the scan setup - see project memory,
                                  # user's own framing: "the TA meant 90-95% except the base, the base is
                                  # occluded from the table"). Empirically verified: 15mm excludes 15.9% of
                                  # surface samples (a plausible base-footprint size for a 192mm-tall
                                  # bottle) and turns 83.5% raw coverage into 92.2% over the observable
                                  # region - comfortably clears the 90% target that was previously unreachable
                                  # by construction, not by insufficient scanning.
COVERAGE_STOP_FRACTION = 0.925  # midpoint of the TA's 90-95% target range
MIN_NEW_SCORE_TO_MOVE = 150     # a candidate offering fewer than this many newly-visible surface pts isn't worth a move -
                                 # also the loop's real terminator when the coverage target is physically unreachable (e.g.
                                 # the object's table-contact underside, which no candidate in the cache can ever see).
                                 # Tuned empirically (see project memory): the score is a noisy predictor of true novel
                                 # coverage (coarse-mesh backface/occlusion test vs. the real sensor's edge-artifact
                                 # filtering never match exactly), so a low threshold let the greedy loop burn through all
                                 # 172 reachable candidates chasing marginal, often-unrealized gains - 20 took 172 views
                                 # for 68.4% coverage (worse efficiency than the 20-view fixed-orbit baseline's 69.5%);
                                 # 150 stops once genuinely large predicted gains run out, trading a slightly lower ceiling
                                 # for drastically fewer views, which is the actual TA-relevant efficiency comparison.


class NBVPlanner:
    def __init__(self, env) -> None:
        if not os.path.exists(REACHABILITY_CACHE_PATH):
            raise FileNotFoundError(
                f"{REACHABILITY_CACHE_PATH} not found - run "
                "`conda run -n rob_env python build_reachability_cache.py` first (Step A)."
            )
        cache = load_reachability_cache(REACHABILITY_CACHE_PATH)
        reachable_mask = cache["reachable"].astype(bool)
        self.candidate_positions_world = cache["t_candidates_world"][reachable_mask]
        self.candidate_orientations_world_xyzw = cache["q_candidates_world_xyzw"][reachable_mask]
        self.visited = np.zeros(len(self.candidate_positions_world), dtype=bool)

        t_obj_world, q_obj_world_xyzw = env._p.getBasePositionAndOrientation(env.obj_id)
        # Coverage ground truth uses the detailed scanned mesh (load_object_mesh_world's default) - accuracy
        # matters for what we're actually trying to measure. The occluder mesh for ray_scoring stays the
        # coarse collision proxy on purpose - self-occlusion doesn't need fine surface detail to be roughly
        # right, and the detailed mesh has ~160x more triangles, which would blow up per-candidate
        # ray/triangle intersection cost during the actual scan (see nbv_core/coverage.py's module docstring).
        self.mesh_world = load_object_mesh_world(np.array(t_obj_world), np.array(q_obj_world_xyzw))
        occluder_mesh_world = load_object_mesh_world(
            np.array(t_obj_world), np.array(q_obj_world_xyzw), mesh_path=MUSTARD_MESH_PATH_COARSE,
        )
        self.triangles_world = occluder_mesh_world.vertices[occluder_mesh_world.faces]

        surface_points, surface_normals = sample_surface_points_and_normals(self.mesh_world, n_samples=N_SURFACE_SAMPLES)
        self.base_exclusion_z = float(self.mesh_world.vertices[:, 2].min()) + BASE_EXCLUSION_HEIGHT_M
        is_observable = surface_points[:, 2] >= self.base_exclusion_z
        self.coverage = CoverageTracker(surface_points[is_observable], surface_normals[is_observable])
        self._n_integrated_chunks = 0

    def select_next(self, accumulated_points_so_far: list[np.ndarray]) -> tuple[np.ndarray, list[float]] | None:
        for points_world in accumulated_points_so_far[self._n_integrated_chunks:]:
            self.coverage.update(points_world)
        self._n_integrated_chunks = len(accumulated_points_so_far)

        if self.coverage.coverage_fraction() >= COVERAGE_STOP_FRACTION:
            return None

        unvisited_idx = np.where(~self.visited)[0]
        if len(unvisited_idx) == 0:
            return None  # every reachable candidate already visited, whatever coverage that got us is final

        scores = score_candidate_views(
            self.candidate_positions_world[unvisited_idx],
            self.coverage.get_unseen_points(), self.coverage.get_unseen_normals(),
            self.triangles_world,
        )
        best_local = int(np.argmax(scores))
        if scores[best_local] < MIN_NEW_SCORE_TO_MOVE:
            return None  # nothing worthwhile left to see from any remaining reachable candidate

        best_idx = unvisited_idx[best_local]
        self.visited[best_idx] = True
        return self.candidate_positions_world[best_idx], self.candidate_orientations_world_xyzw[best_idx].tolist()


_PLANNERS: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def get_coverage_fraction(env) -> float | None:
    """Current known-CAD coverage fraction for env's planner, or None if select_next_view_pose(env, ...) hasn't run yet."""
    planner = _PLANNERS.get(env)
    return planner.coverage.coverage_fraction() if planner is not None else None


def get_coverage_tracker(env) -> "CoverageTracker | None":
    """env's CoverageTracker (seen/unseen state over the known mesh's surface samples), or None if not started yet."""
    planner = _PLANNERS.get(env)
    return planner.coverage if planner is not None else None


def get_base_exclusion_z(env) -> float | None:
    """World z below which env's coverage tracker excludes the table-contact base entirely - see
    BASE_EXCLUSION_HEIGHT_M. None if not started yet."""
    planner = _PLANNERS.get(env)
    return planner.base_exclusion_z if planner is not None else None


def get_object_mesh_world(env):
    """env's known-CAD mesh (trimesh.Trimesh, world frame) - the SAME instance used to build the coverage
    tracker's surface samples, for building a coverage-colored visualization of it. None if not started yet."""
    planner = _PLANNERS.get(env)
    return planner.mesh_world if planner is not None else None


def select_next_view_pose(
    env, accumulated_points_so_far: list[np.ndarray], view_index: int,
) -> tuple[np.ndarray, list[float]] | None:
    """
    Real NBV planner - greedy argmax of Step C's GPU visibility score over
    Step A's cached reachable candidates, stopping once Step B's coverage
    hits COVERAGE_STOP_FRACTION (or no reachable candidate has anything
    worthwhile left to offer). Matches full_pipeline.py's swap-point
    signature exactly; view_index is accepted for compatibility with the
    fixed-orbit stub it replaces but unused - this planner's own visited-set
    tracks progress instead of a linear schedule.
    """
    if env not in _PLANNERS:
        _PLANNERS[env] = NBVPlanner(env)
    planner = _PLANNERS[env]
    return planner.select_next(accumulated_points_so_far)
