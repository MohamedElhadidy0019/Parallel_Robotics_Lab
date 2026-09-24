"""Surface coverage tracking on CPU using spatial KDTree."""

import os
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh

from nbv_planner.config import YCB_ROOT

DEFAULT_SEEN_DISTANCE_THRESHOLD_M = 0.005
DEFAULT_NORMAL_CONSISTENCY_MIN = -0.2
DEPTH_NOISE_FLOOR_M = 0.005
DEFAULT_N_SURFACE_SAMPLES = 4000
DEFAULT_BASE_EXCLUSION_MARGIN_M = 0.003


def ycb_mesh_path(name: str) -> str:
    """Resolve visual OBJ mesh path for a YCB object from URDF, with fallback."""
    obj_dir = os.path.join(YCB_ROOT, name)
    urdf_path = os.path.join(obj_dir, "model.urdf")
    if os.path.isfile(urdf_path):
        import xml.etree.ElementTree as ET
        tree = ET.parse(urdf_path)
        mesh_elem = tree.getroot().find(".//visual/geometry/mesh")
        if mesh_elem is not None and "filename" in mesh_elem.attrib:
            fname = mesh_elem.attrib["filename"]
            p = os.path.join(obj_dir, fname) if not os.path.isabs(fname) else fname
            if os.path.isfile(p):
                return p
    for cand in ("textured_simple_reoriented.obj", "textured.obj"):
        p = os.path.join(obj_dir, cand)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"No visual OBJ found for {name} in {obj_dir}")


def load_ycb_mesh(name: str) -> trimesh.Trimesh:
    """Load visual OBJ mesh without coordinate modifications."""
    path = ycb_mesh_path(name)
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        return loaded.dump(concatenate=True)
    return loaded.copy()


def transform_mesh(
    mesh: trimesh.Trimesh,
    pos: np.ndarray,
    orn_xyzw: np.ndarray,
    obj_name: str | None = None,
    is_inertial_frame: bool = False,
) -> trimesh.Trimesh:
    """Transform visual mesh into world frame taking URDF scale, inertial, and visual origins into account."""
    world_mesh = mesh.copy()

    T_link_ine = np.eye(4, dtype=np.float64)
    T_link_vis = np.eye(4, dtype=np.float64)

    if obj_name is not None:
        urdf_path = os.path.join(YCB_ROOT, obj_name, "model.urdf")
        if os.path.isfile(urdf_path):
            import xml.etree.ElementTree as ET
            tree = ET.parse(urdf_path)

            mesh_elem = tree.getroot().find(".//visual/geometry/mesh")
            if mesh_elem is not None and "scale" in mesh_elem.attrib:
                scale = [float(x) for x in mesh_elem.attrib["scale"].split()]
                world_mesh.apply_scale(scale)

            if is_inertial_frame:
                ine = tree.getroot().find(".//inertial/origin")
                if ine is not None:
                    rpy = [float(x) for x in ine.get("rpy", "0 0 0").split()]
                    xyz = [float(x) for x in ine.get("xyz", "0 0 0").split()]
                    T_link_ine[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
                    T_link_ine[:3, 3] = xyz

            vis = tree.getroot().find(".//visual/origin")
            if vis is not None:
                rpy = [float(x) for x in vis.get("rpy", "0 0 0").split()]
                xyz = [float(x) for x in vis.get("xyz", "0 0 0").split()]
                T_link_vis[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
                T_link_vis[:3, 3] = xyz

    T_world_link = np.eye(4, dtype=np.float64)
    T_world_link[:3, :3] = Rotation.from_quat(orn_xyzw).as_matrix()
    T_world_link[:3, 3] = pos

    if is_inertial_frame:
        T_world_vis = T_world_link @ np.linalg.inv(T_link_ine) @ T_link_vis
    else:
        T_world_vis = T_world_link @ T_link_vis

    world_mesh.apply_transform(T_world_vis)
    return world_mesh


def sample_surface_points_and_normals(
    mesh_world: trimesh.Trimesh,
    n_samples: int = DEFAULT_N_SURFACE_SAMPLES,
    base_exclusion_z: float | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly sample surface points and face normals from world mesh.

    Optionally drops points below base_exclusion_z (table-contact bottom).
    """
    np.random.seed(seed)
    points, face_indices = trimesh.sample.sample_surface(mesh_world, n_samples)
    normals = mesh_world.face_normals[face_indices]

    if base_exclusion_z is not None:
        mask = points[:, 2] >= base_exclusion_z
        points = points[mask]
        normals = normals[mask]

    return points.astype(np.float32), normals.astype(np.float32)


class CoverageTracker:
    """CPU coverage tracker using scipy cKDTree."""

    def __init__(
        self,
        surface_points_world: np.ndarray,
        surface_normals_world: np.ndarray,
        seen_distance_threshold_m: float = DEFAULT_SEEN_DISTANCE_THRESHOLD_M,
        normal_consistency_min: float = DEFAULT_NORMAL_CONSISTENCY_MIN,
    ) -> None:
        self.surface_points = np.asarray(surface_points_world, dtype=np.float32)
        self.surface_normals = np.asarray(surface_normals_world, dtype=np.float32)
        self.seen_distance_threshold_m = seen_distance_threshold_m
        self.normal_consistency_min = normal_consistency_min

        self.seen = np.zeros(len(self.surface_points), dtype=bool)
        self._tree = cKDTree(self.surface_points)

    def update(self, captured_points_world: np.ndarray) -> int:
        """Mark surface samples seen by captured cloud within distance/normal threshold."""
        captured = np.asarray(captured_points_world, dtype=np.float32).reshape(-1, 3)
        if len(captured) == 0 or len(self.surface_points) == 0:
            return 0

        # Ask, per surface sample, whether the capture reached it. Querying the other way round
        # only marks samples that win a nearest-neighbour contest, so samples closer together
        # than the threshold shadow each other and stay unseen however well they were imaged.
        distances, nearest_idx = cKDTree(captured).query(self.surface_points)
        within_dist = distances < self.seen_distance_threshold_m

        if not np.any(within_dist):
            return 0

        # Normal consistency check
        surf_pts = self.surface_points[within_dist]
        surf_normals = self.surface_normals[within_dist]
        offsets = captured[nearest_idx[within_dist]] - surf_pts
        norms = np.linalg.norm(offsets, axis=-1, keepdims=True)
        norms_safe = np.maximum(norms, 1e-6)

        # Allow points very close to surface regardless of offset direction noise
        dot = np.sum((offsets / norms_safe) * surf_normals, axis=-1)
        valid_mask = (distances[within_dist] < DEPTH_NOISE_FLOOR_M) | (dot >= self.normal_consistency_min)

        new_seen_idx = np.flatnonzero(within_dist)[valid_mask]
        before = self.seen.sum()
        self.seen[new_seen_idx] = True
        return int(self.seen.sum() - before)

    def coverage_fraction(self) -> float:
        """Return seen surface fraction in range [0.0, 1.0]."""
        return float(self.seen.mean()) if len(self.seen) > 0 else 0.0

    def get_unseen_points(self) -> np.ndarray:
        """Return coordinates of remaining unseen surface samples."""
        return self.surface_points[~self.seen]

    def get_unseen_normals(self) -> np.ndarray:
        """Return surface normals of remaining unseen surface samples."""
        return self.surface_normals[~self.seen]

    def get_seen_points(self) -> np.ndarray:
        """Return coordinates of seen surface samples."""
        return self.surface_points[self.seen]

    def reset(self) -> None:
        """Clear all seen marks."""
        self.seen.fill(False)


def build_coverage_colored_mesh(
    mesh_world: trimesh.Trimesh,
    tracker: CoverageTracker,
    base_exclusion_z: float | None = None,
    seen_color: tuple[float, float, float] = (0.2, 0.8, 0.2),
    unseen_color: tuple[float, float, float] = (0.65, 0.65, 0.65),
    excluded_color: tuple[float, float, float] = (0.15, 0.15, 0.15),
):
    """Build Open3D TriangleMesh colored by coverage state."""
    import open3d as o3d

    vertices = np.asarray(mesh_world.vertices, dtype=np.float32)
    _, nearest_idx = tracker._tree.query(vertices)
    is_seen = tracker.seen[nearest_idx]

    colors = np.where(is_seen[:, None], np.array(seen_color), np.array(unseen_color))
    if base_exclusion_z is not None:
        is_ex = vertices[:, 2] < base_exclusion_z
        colors = np.where(is_ex[:, None], np.array(excluded_color), colors)

    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh_world.vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh_world.faces)
    mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh_o3d.compute_vertex_normals()
    return mesh_o3d
