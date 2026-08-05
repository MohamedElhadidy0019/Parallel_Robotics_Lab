"""
GPU/CUDA occlusion-aware ray casting for candidate-viewpoint scoring (Step C
of the NBV plan). Own batched Möller-Trumbore ray/triangle intersection in
PyTorch (runs on CUDA when available) - not a port of
CUDA_Lab_Assignments/Assignment04_startup's C++/CUDA kernels, just the same
algorithmic shape those reference (dense candidates -> ray-cast visibility
-> greedy marginal-coverage selection, with Step D owning the greedy-pick
part) reimplemented fresh against this project's known-CAD surface samples
instead of a dense occupancy grid.

For one candidate camera position, a surface point is scored "visible" if:
  1. it's front-facing - the camera sits roughly on the outward-normal side
     of the surface, not looking through the object at its own far wall, and
  2. the straight ray from the camera to that point isn't blocked by any
     closer triangle of the object's own mesh (self-occlusion) - the actual
     ray/triangle intersection test.
Score for a candidate = count of currently-unseen surface points (per
CoverageTracker) that would newly become visible from there.
"""
import numpy as np
import torch

DEFAULT_OCCLUSION_EPSILON_M = 0.003  # safety margin so a target point's own triangle isn't flagged as self-blocking
DEFAULT_BACKFACE_MARGIN = 0.35       # camera must be at least this far onto the outward-normal side to count as front-facing -
                                      # empirically tuned (see project memory): the real capture pipeline's
                                      # edge_discontinuity_mask strips near-grazing/silhouette pixels as untrustworthy, so a
                                      # lenient margin (e.g. 0.05) systematically overcounts points the real sensor never
                                      # confirms; 0.35 tracked real per-view coverage gains far better in closed-loop testing


def _resolve_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _nearest_ray_triangle_hit(
    ray_origin: torch.Tensor, ray_dirs: torch.Tensor, triangles: torch.Tensor, t_min: float = 1e-5,
) -> torch.Tensor:
    """
    Batched Moller-Trumbore: one shared ray_origin (3,), ray_dirs (N, 3)
    (need not be normalized - the returned t is in units of ray_dirs'
    length), triangles (T, 3, 3) vertex-per-row. Returns (N,) nearest
    positive intersection parameter t per ray (torch.inf where no triangle
    is hit) - a point at ray_origin + t * ray_dirs[n] lies exactly on the
    nearest blocking triangle.
    """
    v0, v1, v2 = triangles[:, 0, :], triangles[:, 1, :], triangles[:, 2, :]  # (T, 3) each
    edge1 = v1 - v0  # (T, 3)
    edge2 = v2 - v0  # (T, 3)
    s = (ray_origin[None, :] - v0)  # (T, 3) - origin is shared, so this depends only on triangle
    q = torch.cross(s, edge1, dim=-1)  # (T, 3) - also origin/triangle only, independent of ray direction

    h = torch.cross(ray_dirs[:, None, :], edge2[None, :, :], dim=-1)  # (N, T, 3)
    a = torch.sum(edge1[None, :, :] * h, dim=-1)  # (N, T)
    nonparallel = a.abs() > 1e-9
    f = torch.where(nonparallel, 1.0 / torch.where(nonparallel, a, torch.ones_like(a)), torch.zeros_like(a))

    u = f * torch.sum(s[None, :, :] * h, dim=-1)  # (N, T)
    v = f * torch.sum(ray_dirs[:, None, :] * q[None, :, :], dim=-1)  # (N, T)
    t_hit = f * torch.sum(edge2[None, :, :] * q[None, :, :], dim=-1)  # (N, T)

    valid = nonparallel & (u >= 0) & (u <= 1) & (v >= 0) & (u + v <= 1) & (t_hit > t_min)
    t_hit = torch.where(valid, t_hit, torch.full_like(t_hit, torch.inf))
    return t_hit.min(dim=-1).values  # (N,)


def is_visible_from(
    camera_pos_world: np.ndarray,
    target_points_world: np.ndarray,
    target_normals_world: np.ndarray,
    triangles_world: np.ndarray,
    occlusion_epsilon_m: float = DEFAULT_OCCLUSION_EPSILON_M,
    backface_margin: float = DEFAULT_BACKFACE_MARGIN,
    device: str | None = None,
) -> np.ndarray:
    """
    camera_pos_world (3,), target_points_world/target_normals_world (N, 3),
    triangles_world (T, 3, 3) - the object mesh's own triangles, the only
    occluder this project's mustard-only scope needs (no shelf/other
    objects in the way). Returns (N,) bool: front-facing AND not
    self-occluded by the mesh.
    """
    dev = _resolve_device(device)
    camera_pos = torch.as_tensor(camera_pos_world, dtype=torch.float32, device=dev)
    targets = torch.as_tensor(target_points_world, dtype=torch.float32, device=dev)
    normals = torch.as_tensor(target_normals_world, dtype=torch.float32, device=dev)
    triangles = torch.as_tensor(triangles_world, dtype=torch.float32, device=dev)

    to_camera = camera_pos[None, :] - targets  # (N, 3)
    distances = torch.linalg.norm(to_camera, dim=-1)
    distances_safe = torch.clamp(distances, min=1e-9)
    to_camera_dir = to_camera / distances_safe[:, None]
    front_facing = torch.sum(to_camera_dir * normals, dim=-1) > backface_margin

    ray_dirs = -to_camera  # from camera_pos toward each target, length = distances
    nearest_t = _nearest_ray_triangle_hit(camera_pos, ray_dirs, triangles)
    not_occluded = nearest_t >= (1.0 - occlusion_epsilon_m / distances_safe.clamp(min=1e-6))

    return (front_facing & not_occluded).cpu().numpy()


def score_candidate_views(
    candidate_positions_world: np.ndarray,
    unseen_points_world: np.ndarray,
    unseen_normals_world: np.ndarray,
    triangles_world: np.ndarray,
    occlusion_epsilon_m: float = DEFAULT_OCCLUSION_EPSILON_M,
    backface_margin: float = DEFAULT_BACKFACE_MARGIN,
    device: str | None = None,
) -> np.ndarray:
    """
    candidate_positions_world (M, 3) - score[m] = how many currently-unseen
    surface points (per CoverageTracker.get_unseen_points()/get_unseen_normals())
    would become newly visible from candidate m. Returns (M,) int array.
    """
    if unseen_points_world.shape[0] == 0:
        return np.zeros(len(candidate_positions_world), dtype=np.int64)

    scores = np.empty(len(candidate_positions_world), dtype=np.int64)
    for i, cam_pos in enumerate(candidate_positions_world):
        visible = is_visible_from(
            cam_pos, unseen_points_world, unseen_normals_world, triangles_world,
            occlusion_epsilon_m=occlusion_epsilon_m, backface_margin=backface_margin, device=device,
        )
        scores[i] = int(visible.sum())
    return scores
