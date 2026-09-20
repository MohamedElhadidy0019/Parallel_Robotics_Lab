"""Detect a single object on a table from one RGB-D observation."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.spatial import cKDTree

from nbv_planner.observations import Observation

MAX_TABLE_TILT_RAD = np.radians(10.0)


class Segmenter(Protocol):
    def segment(self, rgb: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray: ...


@dataclass(frozen=True)
class TablePlane:
    normal: np.ndarray
    offset: float
    inliers_world: np.ndarray

    def height(self, points: np.ndarray) -> np.ndarray:
        return points @ self.normal + self.offset

    def z_at(self, xy: np.ndarray) -> float:
        n = self.normal
        return float(-(n[0] * xy[0] + n[1] * xy[1] + self.offset) / n[2])


@dataclass(frozen=True)
class TableBox:
    center: np.ndarray
    size: np.ndarray
    yaw: float

    def corners(self) -> np.ndarray:
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
        rotation = np.eye(3)
        rotation[:2, :2] = rotation_2d(self.yaw)
        return (signs * self.size / 2.0) @ rotation.T + self.center

    def contains(self, points: np.ndarray, margin: float = 0.0) -> np.ndarray:
        local = points - self.center
        local[:, :2] = local[:, :2] @ rotation_2d(self.yaw)
        return np.all(np.abs(local) <= self.size / 2.0 + margin, axis=1)


@dataclass(frozen=True)
class Detection:
    mask: np.ndarray
    prompt_box: tuple[int, int, int, int]
    points_world: np.ndarray
    box: TableBox
    table: TablePlane


def pixels_to_world(observation: Observation, pixel_mask: np.ndarray) -> np.ndarray:
    v, u = np.nonzero(pixel_mask)
    z = observation.depth_m[v, u].astype(np.float64)
    intr = observation.intrinsics
    points_camera = np.stack([(u - intr.cx) * z / intr.fx, (v - intr.cy) * z / intr.fy, z], axis=-1)
    transform = observation.world_from_camera
    return points_camera @ transform[:3, :3].T + transform[:3, 3]


def valid_depth(observation: Observation, max_depth: float) -> np.ndarray:
    depth = observation.depth_m
    return np.isfinite(depth) & (depth > observation.intrinsics.near) & (depth < max_depth)


def fit_table_plane(points_world: np.ndarray, distance_threshold: float = 0.005, max_planes: int = 3,
                    sample_size: int = 30_000, seed: int = 0) -> TablePlane:
    import open3d as o3d

    rng = np.random.default_rng(seed)
    remaining = points_world[rng.permutation(len(points_world))[:sample_size]]
    for _ in range(max_planes):
        if len(remaining) < 100:
            break
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(remaining))
        (a, b, c, d), inliers = cloud.segment_plane(distance_threshold, ransac_n=3, num_iterations=500)
        normal = np.array([a, b, c])
        norm = np.linalg.norm(normal)
        normal, d = normal / norm, d / norm
        if normal[2] < 0:
            normal, d = -normal, -d
        if np.arccos(np.clip(normal[2], -1.0, 1.0)) <= MAX_TABLE_TILT_RAD:
            return TablePlane(normal, float(d), remaining[inliers])
        remaining = np.delete(remaining, inliers, axis=0)
    raise ValueError("No horizontal table plane found in the observation")


def largest_cluster(points: np.ndarray, eps: float = 0.01, min_points: int = 20,
                    sample_size: int = 20_000, seed: int = 0) -> np.ndarray:
    """Boolean mask over points that belong to the largest DBSCAN cluster (clustered on a subsample)."""
    import open3d as o3d

    rng = np.random.default_rng(seed)
    sample = points[rng.permutation(len(points))[:sample_size]]
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sample))
    labels = np.asarray(cloud.cluster_dbscan(eps=eps, min_points=min_points))
    if labels.max() < 0:
        raise ValueError("No point cluster found above the table")
    cluster = sample[labels == np.bincount(labels[labels >= 0]).argmax()]
    distances, _ = cKDTree(cluster).query(points)
    return distances <= eps


def over_table(points: np.ndarray, table: TablePlane, margin: float = 0.02) -> np.ndarray:
    lo = table.inliers_world[:, :2].min(axis=0) - margin
    hi = table.inliers_world[:, :2].max(axis=0) + margin
    return np.all((points[:, :2] >= lo) & (points[:, :2] <= hi), axis=1)


def object_pixel_mask(observation: Observation, table: TablePlane, max_depth: float, min_height: float,
                      max_height: float, prior: TableBox | None, prior_margin: float) -> np.ndarray:
    candidates = valid_depth(observation, max_depth)
    points = pixels_to_world(observation, candidates)
    height = table.height(points)
    keep = (height > min_height) & (height < max_height) & over_table(points, table)
    if prior is not None:
        keep &= prior.contains(points, prior_margin)
    if keep.sum() < 50:
        raise ValueError("No object points above the table")
    keep[keep] = largest_cluster(points[keep])

    mask = np.zeros_like(candidates)
    v, u = np.nonzero(candidates)
    mask[v[keep], u[keep]] = True
    return mask


def bounding_pixel_box(mask: np.ndarray, pad: int) -> tuple[int, int, int, int]:
    v, u = np.nonzero(mask)
    h, w = mask.shape
    return (max(int(u.min()) - pad, 0), max(int(v.min()) - pad, 0),
            min(int(u.max()) + pad, w - 1), min(int(v.max()) + pad, h - 1))


def rotation_2d(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def footprint_extents(xy: np.ndarray, yaw: float) -> tuple[np.ndarray, np.ndarray]:
    local = xy @ rotation_2d(yaw)
    return local.min(axis=0), local.max(axis=0)


def table_aligned_box(points_world: np.ndarray, table: TablePlane) -> TableBox:
    """Box resting on the table, rotated about vertical to the minimum-area rectangle of the footprint."""
    from scipy.spatial import ConvexHull

    hull = points_world[ConvexHull(points_world[:, :2], qhull_options="QJ").vertices, :2]
    edges = np.roll(hull, -1, axis=0) - hull
    edge_angles = np.unique(np.arctan2(edges[:, 1], edges[:, 0]) % (np.pi / 2))
    areas = [np.prod(np.diff(footprint_extents(hull, angle), axis=0)) for angle in edge_angles]
    yaw = float(edge_angles[int(np.argmin(areas))])

    lo, hi = footprint_extents(hull, yaw)
    if hi[1] - lo[1] > hi[0] - lo[0]:
        yaw += np.pi / 2
    yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2

    lo, hi = footprint_extents(hull, yaw)
    center_xy = rotation_2d(yaw) @ ((lo + hi) / 2.0)
    bottom = table.z_at(center_xy)
    top = float(points_world[:, 2].max())
    return TableBox(
        center=np.array([center_xy[0], center_xy[1], (bottom + top) / 2.0]),
        size=np.array([hi[0] - lo[0], hi[1] - lo[1], top - bottom]),
        yaw=yaw,
    )


def detect_object(observation: Observation, segmenter: Segmenter | None = None, prior: TableBox | None = None,
                  prior_margin: float = 0.10, max_depth: float = 1.5, min_height: float = 0.01,
                  max_height: float = 0.5, prompt_pad: int = 10) -> Detection:
    """Find the object resting on the table, near the prior box if given. Without a segmenter, the depth cluster is the mask."""
    table = fit_table_plane(pixels_to_world(observation, valid_depth(observation, max_depth)))
    cluster_mask = object_pixel_mask(observation, table, max_depth, min_height, max_height, prior, prior_margin)
    prompt_box = bounding_pixel_box(cluster_mask, prompt_pad)

    mask = cluster_mask if segmenter is None else segmenter.segment(observation.rgb, prompt_box)
    mask = mask & valid_depth(observation, max_depth)
    points = pixels_to_world(observation, mask)
    points = points[table.height(points) > min_height]
    if prior is not None:
        points = points[prior.contains(points, prior_margin)]
    if len(points) < 50:
        raise ValueError("Segmented object has too few points above the table")
    points = points[largest_cluster(points)]

    return Detection(mask, prompt_box, points, table_aligned_box(points, table), table)
