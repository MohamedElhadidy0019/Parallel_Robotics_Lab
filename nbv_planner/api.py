"""High-level NBV Planner API for autonomous robot inspection.

Provides a unified, simulation-agnostic interface for candidate viewpoint generation,
kinematic reachability filtering, CUDA ray casting gain evaluation, and surface
coverage tracking.
"""

from dataclasses import dataclass
from typing import Sequence, Tuple
import numpy as np

from nbv_planner.config import (
    BASE_EXCLUSION_HEIGHT_M,
    BASE_LINK,
    EE_LINK,
    ELEVATION_DEG,
    N_AZIMUTH,
    N_RADIUS,
    URDF_PATH,
)
from nbv_planner.coverage import CoverageTracker, sample_surface_points_and_normals
from nbv_planner.ray_scoring import (
    DEFAULT_BACKFACE_MARGIN,
    DEFAULT_OCCLUSION_EPSILON_M,
    score_candidate_views,
)
from nbv_planner.reachability import ik_filter, sample_candidate_camera_poses


class NBVPlanner:
    """Autonomous Next-Best-View planning engine."""

    def __init__(
        self,
        mesh_world,
        n_surface_samples: int = 4000,
        base_exclusion_z: float | None = None,
        urdf_path: str = URDF_PATH,
        base_link: str = BASE_LINK,
        ee_link: str = EE_LINK,
    ) -> None:
        self.mesh_world = mesh_world
        self.triangles_world = np.asarray(mesh_world.vertices[mesh_world.faces], dtype=np.float32)
        self.urdf_path = urdf_path
        self.base_link = base_link
        self.ee_link = ee_link

        if base_exclusion_z is None:
            base_exclusion_z = float(mesh_world.vertices[:, 2].min()) + BASE_EXCLUSION_HEIGHT_M

        self.surface_pts, self.surface_nrm = sample_surface_points_and_normals(
            mesh_world, n_samples=n_surface_samples, base_exclusion_z=base_exclusion_z
        )
        self.tracker = CoverageTracker(self.surface_pts, self.surface_nrm)

    def sample_candidates(
        self,
        obj_pos: np.ndarray,
        radius_range: Tuple[float, float],
        z_min: float | None = None,
        elevation_deg: Tuple[float, float, int] = ELEVATION_DEG,
        n_azimuth: int = N_AZIMUTH,
        n_radius: int = N_RADIUS,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate hemisphere shell candidate poses looking at obj_pos."""
        return sample_candidate_camera_poses(
            obj_pos,
            radius=(radius_range[0], radius_range[1], n_radius),
            elevation_deg=elevation_deg,
            n_azimuth=n_azimuth,
            z_min_world=z_min,
        )

    def filter_reachability(
        self,
        t_cand: np.ndarray,
        q_cand: np.ndarray,
        t_base_world: np.ndarray,
        q_base_world_xyzw: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Filter candidate poses using batch GPU IK kinematics."""
        return ik_filter(
            self.urdf_path,
            self.base_link,
            self.ee_link,
            t_cand,
            q_cand,
            t_base_world,
            q_base_world_xyzw,
        )

    def score_unvisited_views(
        self,
        candidate_positions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate candidate viewpoints against remaining unseen surface points via CUDA ray casting."""
        unseen_pts = self.tracker.get_unseen_points()
        unseen_nrm = self.tracker.get_unseen_normals()
        if len(unseen_pts) == 0 or len(candidate_positions) == 0:
            return np.zeros(len(candidate_positions), dtype=np.int32), np.zeros(
                (len(candidate_positions), 0), dtype=np.uint8
            )

        return score_candidate_views(
            candidate_positions,
            unseen_pts,
            unseen_nrm,
            self.triangles_world,
        )

    def update_coverage(self, observed_points_world: np.ndarray) -> Tuple[int, float]:
        """Update coverage tracker with a newly captured 3D point cloud."""
        newly_seen = self.tracker.update(observed_points_world)
        coverage_fraction = self.tracker.coverage_fraction()
        return newly_seen, coverage_fraction

    def get_coverage(self) -> float:
        """Current surface coverage fraction [0, 1]."""
        return self.tracker.coverage_fraction()
