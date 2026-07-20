"""
Own tri-state (unknown/free/occupied) voxel occupancy grid, with ray-marching
integration of depth points.

Deliberately independent of shelf_gym's mapping_utils.py (BEVMapping /
freeSpaceCalculator) - this is the project's own reconstruction/occupancy
representation. Tri-state (not just occupied, like the earlier CUDA-assignment
grid) because online there's no ground truth: a candidate viewpoint has to be
scored later by how much UNKNOWN space it would resolve, not by re-visibility
of already-known occupied voxels.
"""
import numpy as np

UNKNOWN = 0
FREE = 1
OCCUPIED = 2


class OccupancyVoxelGrid:
    origin: np.ndarray
    size: tuple[int, int, int]
    voxel_size: float
    grid: np.ndarray

    def __init__(self, origin: np.ndarray, size: tuple[int, int, int], voxel_size: float) -> None:
        """
        origin: (3,) world-frame position of the grid's (0,0,0) corner (min corner)
        size: (nx, ny, nz) number of voxels along each axis
        voxel_size: edge length of a voxel, in meters
        """
        self.origin = np.asarray(origin, dtype=np.float64)
        self.size = tuple(int(s) for s in size)
        self.voxel_size = float(voxel_size)
        self.grid = np.zeros(self.size, dtype=np.int8)

    @classmethod
    def from_aabb(
        cls, aabb_min: np.ndarray, aabb_max: np.ndarray, margin: float, voxel_size: float
    ) -> "OccupancyVoxelGrid":
        """Build a grid sized to cover an object's AABB plus a margin on each side."""
        aabb_min = np.asarray(aabb_min, dtype=np.float64) - margin
        aabb_max = np.asarray(aabb_max, dtype=np.float64) + margin
        size = np.maximum(np.ceil((aabb_max - aabb_min) / voxel_size).astype(int), 1)
        return cls(origin=aabb_min, size=size, voxel_size=voxel_size)

    def world_to_index(self, points_world: np.ndarray) -> np.ndarray:
        """(..., 3) world points -> (..., 3) integer indices. Not bounds-checked."""
        points_world = np.asarray(points_world, dtype=np.float64)
        return np.floor((points_world - self.origin) / self.voxel_size).astype(np.int64)

    def index_to_world(self, indices: np.ndarray) -> np.ndarray:
        """(..., 3) integer indices -> (..., 3) voxel-center world coordinates."""
        indices = np.asarray(indices, dtype=np.float64)
        return self.origin + (indices + 0.5) * self.voxel_size

    def in_bounds(self, indices: np.ndarray) -> np.ndarray:
        size = np.asarray(self.size)
        return np.all((indices >= 0) & (indices < size), axis=-1)

    def integrate_points(
        self, points_world: np.ndarray, cam_origin_world: np.ndarray, max_range: float = 2.0
    ) -> None:
        """
        Ray-march from cam_origin_world to each point in points_world (N, 3),
        marking traversed voxels FREE and each ray's endpoint voxel OCCUPIED.
        OCCUPIED is sticky - never downgraded back to FREE/UNKNOWN. Endpoints
        are de-duplicated to voxel resolution before marching (many pixels
        land in the same voxel), same stepping convention (voxel_size / 2)
        as the earlier CUDA-assignment ray caster.
        """
        points_world = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        if points_world.shape[0] == 0:
            return

        cam_origin_world = np.asarray(cam_origin_world, dtype=np.float64)

        end_idx = self.world_to_index(points_world)
        end_idx = end_idx[self.in_bounds(end_idx)]
        if end_idx.shape[0] == 0:
            return
        end_idx = np.unique(end_idx, axis=0)

        end_world = self.index_to_world(end_idx)
        deltas = end_world - cam_origin_world
        distances = np.clip(np.linalg.norm(deltas, axis=-1), 1e-6, max_range)
        directions = deltas / distances[:, None]

        step = self.voxel_size / 2.0
        n_steps = int(np.ceil(distances.max() / step)) + 1
        t = np.arange(n_steps) * step

        ray_points = cam_origin_world + directions[:, None, :] * t[None, :, None]  # (n_rays, n_steps, 3)
        ray_valid = t[None, :] < distances[:, None]  # stop marking free before the endpoint itself

        free_idx = self.world_to_index(ray_points[ray_valid])
        free_idx = free_idx[self.in_bounds(free_idx)]
        if free_idx.shape[0] > 0:
            free_idx = np.unique(free_idx, axis=0)
            fx, fy, fz = free_idx[:, 0], free_idx[:, 1], free_idx[:, 2]
            still_unknown = self.grid[fx, fy, fz] == UNKNOWN
            self.grid[fx[still_unknown], fy[still_unknown], fz[still_unknown]] = FREE

        ex, ey, ez = end_idx[:, 0], end_idx[:, 1], end_idx[:, 2]
        self.grid[ex, ey, ez] = OCCUPIED

    def get_occupied_points(self) -> np.ndarray:
        idx = np.argwhere(self.grid == OCCUPIED)
        return self.index_to_world(idx)

    def get_unknown_count(self) -> int:
        return int(np.sum(self.grid == UNKNOWN))

    def get_occupied_count(self) -> int:
        return int(np.sum(self.grid == OCCUPIED))
