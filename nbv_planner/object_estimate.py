"""Object box and point cloud refined by merging detections from several views."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from nbv_planner.config import ALIGN_MAX_CENTER_ERROR_M, ALIGN_MAX_POINT_DISTANCE_M
from nbv_planner.detection import Detection, TableBox, TablePlane, largest_cluster, table_aligned_box


class ObjectEstimate:
    def __init__(self, voxel_size: float = 0.003, converged_shift: float = 0.003) -> None:
        self.voxel_size = voxel_size
        self.converged_shift = converged_shift
        self.points = np.zeros((0, 3))
        self.table: TablePlane | None = None
        self.boxes: list[TableBox] = []

    @property
    def box(self) -> TableBox | None:
        return self.boxes[-1] if self.boxes else None

    @property
    def center_shift(self) -> float:
        if len(self.boxes) < 2:
            return float("inf")
        return float(np.linalg.norm(self.boxes[-1].center - self.boxes[-2].center))

    @property
    def converged(self) -> bool:
        return self.center_shift < self.converged_shift

    def add(self, detection: Detection) -> TableBox:
        import open3d as o3d

        self.table = self.table or detection.table
        merged = np.vstack([self.points, detection.points_world])
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(merged))
        merged = np.asarray(cloud.voxel_down_sample(self.voxel_size).points)
        self.points = merged[largest_cluster(merged)]
        self.boxes.append(table_aligned_box(self.points, self.table))
        return self.box

    def surface_mesh(self, poisson_depth: int = 8, density_quantile: float = 0.02, crop_margin: float = 0.01):
        """Poisson surface of the merged points, trimmed to the box and cut at the table."""
        import open3d as o3d
        import trimesh

        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.points))
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
        normals = np.asarray(cloud.normals)
        pointing_inward = np.einsum("ij,ij->i", normals, self.points - self.box.center) < 0
        normals[pointing_inward] *= -1
        cloud.normals = o3d.utility.Vector3dVector(normals)

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud, depth=poisson_depth)
        densities = np.asarray(densities)
        mesh.remove_vertices_by_mask(densities < np.quantile(densities, density_quantile))

        corners = self.box.corners()
        lo, hi = corners.min(axis=0) - crop_margin, corners.max(axis=0) + crop_margin
        lo[2] = self.box.center[2] - self.box.size[2] / 2.0
        mesh = mesh.crop(o3d.geometry.AxisAlignedBoundingBox(lo, hi))
        mesh.remove_unreferenced_vertices()
        if len(mesh.triangles) == 0:
            raise ValueError("Surface reconstruction produced no triangles")
        return trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles), process=False)


@dataclass(frozen=True)
class MeshAlignment:
    center_error: float
    point_distance_mean: float
    point_distance_p95: float

    def aligned(self, max_center_error: float = ALIGN_MAX_CENTER_ERROR_M,
                max_point_distance: float = ALIGN_MAX_POINT_DISTANCE_M) -> bool:
        return self.center_error <= max_center_error and self.point_distance_p95 <= max_point_distance


def mesh_alignment(estimate: ObjectEstimate, mesh_world, samples: int = 100_000) -> MeshAlignment:
    """How well a world-placed mesh agrees with the scanned points and box."""
    surface = np.asarray(mesh_world.sample(samples))
    distances, _ = cKDTree(surface).query(estimate.points)
    mesh_box = table_aligned_box(surface, estimate.table)
    return MeshAlignment(
        center_error=float(np.linalg.norm(estimate.box.center - mesh_box.center)),
        point_distance_mean=float(distances.mean()),
        point_distance_p95=float(np.percentile(distances, 95)),
    )
