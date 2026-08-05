"""
Known-CAD coverage tracking (Step B of the NBV plan). Two independent
halves:

  - load_object_mesh_world() / sample_surface_points_and_normals(): turn the
    object's ground-truth collision mesh into a dense set of world-frame
    surface points + normals, given the object's LIVE settled pose (no
    dependency on a prior scan's saved object_pose.npz - unlike
    compare_pointcloud_to_mesh.py, this runs during a live NBVEnv2 session,
    before any capture has happened). Same mesh + same inertial-frame
    correction as compare_pointcloud_to_mesh.py (see its docstring for why
    the inertial offset matters - PyBullet's getBasePositionAndOrientation
    returns the INERTIAL frame pose, not the URDF link/visual origin), just
    re-derived here against a live pose instead of a saved one.

  - CoverageTracker: seen/unseen state over those surface samples, updated
    by nearest-neighbor + normal-consistency reprojection of each new
    capture's point cloud, per the plan's Step B design. Coverage = fraction
    of surface samples marked seen; nbv_planner.py's stopping criterion
    (90-95%) reads coverage_fraction() directly.

Mesh path + inertial-offset constants below are specific to YcbMustardBottle
- the only object this project's mustard-only scope (see project plan) ever
loads. Two mesh variants, both usable via load_object_mesh_world's
mesh_path arg (same R_MESH_BASELINK offset applies to both - model.urdf
uses the same <origin rpy="0 0 1.57"> for both its visual and collision
geometry):
  - MUSTARD_MESH_PATH_DETAILED (textured_simple_reoriented.obj, ~15.7k
    triangles) - the real YCB scan, now what nbv_environment.py's
    model.urdf actually renders/captures. Use this for coverage-tracking's
    surface SAMPLES (the "ground truth" points coverage is measured
    against) - accuracy matters there.
  - MUSTARD_MESH_PATH_COARSE (collision_vhacd.obj, ~98 triangles, the old
    default) - kept for nbv_core.ray_scoring's OCCLUDER mesh specifically:
    self-occlusion doesn't need fine surface detail to be roughly right,
    and ray/triangle intersection cost scales with triangle count -
    the detailed mesh would be ~160x more ray-triangle tests per candidate,
    turning a ~1min scan-time scoring cost into over an hour.
"""
import os

import numpy as np
import pybullet as pb
import trimesh
from scipy.spatial import cKDTree

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MUSTARD_DIR = os.path.join(
    PROJECT_ROOT, "third_party/shelf_gym_repo/shelf_gym/meshes/urdf/ycb_objects/YcbMustardBottle",
)
MUSTARD_MESH_PATH_DETAILED = os.path.join(_MUSTARD_DIR, "textured_simple_reoriented.obj")
MUSTARD_MESH_PATH_COARSE = os.path.join(_MUSTARD_DIR, "collision_vhacd.obj")
MUSTARD_MESH_PATH = MUSTARD_MESH_PATH_DETAILED  # load_object_mesh_world's default
# From model.urdf's <visual>/<collision> <origin rpy="0 0 1.57"> (same value for both meshes).
R_MESH_BASELINK = np.array(pb.getMatrixFromQuaternion(pb.getQuaternionFromEuler([0, 0, 1.57]))).reshape(3, 3)
# model.urdf's <inertial><origin rpy="0 0 0.1" xyz="0.005 0.005 -0.015"/> (same value
# model_textureless.urdf used too) - see compare_pointcloud_to_mesh.py's
# R_LINK_INERTIAL/T_LINK_INERTIAL docstring for the full story.
R_LINK_INERTIAL = np.array(pb.getMatrixFromQuaternion(pb.getQuaternionFromEuler([0, 0, 0.1]))).reshape(3, 3)
T_LINK_INERTIAL = np.array([0.005, 0.005, -0.015])

DEFAULT_SEEN_DISTANCE_THRESHOLD_M = 0.006  # matches this project's observed sub-mm-to-few-mm scan noise floor
DEFAULT_NORMAL_CONSISTENCY_MIN = -0.2      # lenient secondary sanity check, distance threshold does the real filtering


def load_object_mesh_world(
    t_obj_world: np.ndarray, q_obj_world_xyzw: np.ndarray, mesh_path: str = MUSTARD_MESH_PATH,
) -> trimesh.Trimesh:
    """
    t_obj_world/q_obj_world_xyzw: the object's LIVE inertial-frame pose,
    straight from env._p.getBasePositionAndOrientation(env.obj_id) - undoes
    the inertial offset the same way compare_pointcloud_to_mesh.py does,
    then composes with the mesh's own local-frame offset to place every
    mesh vertex in world coordinates.
    """
    R_inertial_world = np.array(pb.getMatrixFromQuaternion(q_obj_world_xyzw)).reshape(3, 3)
    R_baselink_world = R_inertial_world @ R_LINK_INERTIAL.T
    t_baselink_world = R_inertial_world @ (-R_LINK_INERTIAL.T @ T_LINK_INERTIAL) + t_obj_world

    tm = trimesh.load(mesh_path, process=False)
    if isinstance(tm, trimesh.Scene):
        tm = trimesh.util.concatenate(list(tm.geometry.values()))
    v_mesh = np.asarray(tm.vertices)
    v_world = (R_baselink_world @ (R_MESH_BASELINK @ v_mesh.T)).T + t_baselink_world

    mesh_world = trimesh.Trimesh(vertices=v_world, faces=np.asarray(tm.faces), process=False)
    return mesh_world


def sample_surface_points_and_normals(
    mesh_world: trimesh.Trimesh, n_samples: int = 5000, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform-area surface sampling (trimesh.sample.sample_surface) -> (points (N,3), outward normals (N,3))."""
    points, face_indices = trimesh.sample.sample_surface(mesh_world, n_samples, seed=seed)
    normals = mesh_world.face_normals[face_indices]
    return np.asarray(points), np.asarray(normals)


DEFAULT_MAX_INCIDENCE_DEG = 75.0  # empirically tuned (see project memory): points captured near this
                                   # grazing an angle relative to the known true surface normal showed
                                   # ~10mm noise std vs ~1mm for near-head-on views - the dominant source
                                   # of visible surface roughness/"ribbing", far larger than depth-buffer
                                   # quantization ever was


def incidence_angle_mask(
    points_world: np.ndarray,
    cam_pos_world: np.ndarray,
    surface_points_world: np.ndarray,
    surface_normals_world: np.ndarray,
    max_incidence_deg: float = DEFAULT_MAX_INCIDENCE_DEG,
) -> np.ndarray:
    """
    True where a captured point was viewed from close enough to head-on relative to the KNOWN
    true surface's local normal to be trustworthy; False at grazing/oblique incidence.

    This is a capture-time quality filter, not a coverage/tracking concern - a standard technique
    in real 3D scanning pipelines (real depth sensors are also noisiest at grazing incidence).
    Uses the known-CAD mesh (already this project's core assumption elsewhere - coverage
    tracking, ray-scoring) as ground truth for "which way does the real surface face here,"
    independent of whichever direction the capturing camera happened to be looking from - the
    same nearest-surface-sample lookup CoverageTracker.update() already does for its
    normal-consistency check, just applied here to decide whether to keep a point at all rather
    than whether it counts as "seen."
    """
    tree = cKDTree(surface_points_world)
    _, nearest_idx = tree.query(points_world)
    nearest_normal = surface_normals_world[nearest_idx]

    to_camera = cam_pos_world[None, :] - points_world
    to_camera_dir = to_camera / np.maximum(np.linalg.norm(to_camera, axis=-1, keepdims=True), 1e-9)
    cos_incidence = np.einsum("ij,ij->i", to_camera_dir, nearest_normal)
    return cos_incidence > np.cos(np.radians(max_incidence_deg))


class CoverageTracker:
    """
    Seen/unseen state over a fixed set of known-CAD surface samples.
    update() reprojects a newly captured point cloud onto the nearest
    surface sample and marks it seen if both the distance is within
    seen_distance_threshold_m AND the capture point isn't clearly on the
    wrong side of the surface (normal-consistency check) - the two checks
    the Step B design calls for.
    """

    def __init__(
        self,
        surface_points_world: np.ndarray,
        surface_normals_world: np.ndarray,
        seen_distance_threshold_m: float = DEFAULT_SEEN_DISTANCE_THRESHOLD_M,
        normal_consistency_min: float = DEFAULT_NORMAL_CONSISTENCY_MIN,
    ) -> None:
        self.surface_points_world = np.asarray(surface_points_world, dtype=np.float64)
        self.surface_normals_world = np.asarray(surface_normals_world, dtype=np.float64)
        self.seen_distance_threshold_m = seen_distance_threshold_m
        self.normal_consistency_min = normal_consistency_min
        self.seen = np.zeros(len(self.surface_points_world), dtype=bool)
        self._tree = cKDTree(self.surface_points_world)

    def update(self, captured_points_world: np.ndarray) -> int:
        """Marks newly-seen surface samples from one capture's points. Returns how many became newly seen."""
        captured_points_world = np.asarray(captured_points_world, dtype=np.float64).reshape(-1, 3)
        if captured_points_world.shape[0] == 0:
            return 0

        distances, nearest_idx = self._tree.query(captured_points_world)
        within_distance = distances < self.seen_distance_threshold_m

        surface_pts = self.surface_points_world[nearest_idx]
        surface_normals = self.surface_normals_world[nearest_idx]
        offsets = captured_points_world - surface_pts
        offset_norms = np.maximum(np.linalg.norm(offsets, axis=-1), 1e-9)
        normal_dot = np.einsum("ij,ij->i", offsets / offset_norms[:, None], surface_normals)
        normal_consistent = normal_dot > self.normal_consistency_min

        newly_confirmed_idx = nearest_idx[within_distance & normal_consistent]
        before = self.seen.sum()
        self.seen[newly_confirmed_idx] = True
        return int(self.seen.sum() - before)

    def coverage_fraction(self) -> float:
        return float(self.seen.mean()) if len(self.seen) > 0 else 0.0

    def get_unseen_points(self) -> np.ndarray:
        return self.surface_points_world[~self.seen]

    def get_unseen_normals(self) -> np.ndarray:
        return self.surface_normals_world[~self.seen]

    def get_seen_points(self) -> np.ndarray:
        return self.surface_points_world[self.seen]
