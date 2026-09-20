"""Occlusion-aware ray casting and candidate viewpoint scoring using custom CUDA kernel."""

import os
import sys

# Ensure CUDA and ninja paths are configured before importing PyTorch cpp_extension
local_bin = os.path.expanduser("~/.local/bin")
if os.path.isdir(local_bin) and local_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{local_bin}:{os.environ.get('PATH', '')}"

if "CUDA_HOME" not in os.environ:
    for cand in ("/usr/local/cuda", "/usr/local/cuda-12.1", "/usr/local/cuda-12"):
        if os.path.isdir(cand):
            os.environ["CUDA_HOME"] = cand
            break

cuda_bin = os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda"), "bin")
if os.path.isdir(cuda_bin) and cuda_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{cuda_bin}:{os.environ.get('PATH', '')}"

if "TORCH_CUDA_ARCH_LIST" not in os.environ:
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"

import numpy as np
import torch
from torch.utils.cpp_extension import load

from nbv_planner.config import DEFAULT_CAMERA_NEAR

DEFAULT_BACKFACE_MARGIN = 0.35
DEFAULT_OCCLUSION_EPSILON_M = 0.003

_EXT_MODULE = None


def _get_cuda_module():
    """JIT compile and load ray scoring C++/CUDA extension once."""
    global _EXT_MODULE
    if _EXT_MODULE is not None:
        return _EXT_MODULE

    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
    cpp_src = os.path.join(src_dir, "ray_scoring.cpp")
    cuda_src = os.path.join(src_dir, "ray_scoring_kernel.cu")

    build_dir = os.path.join(src_dir, "build")
    os.makedirs(build_dir, exist_ok=True)

    extra_cuda_cflags = ["-O3", "--use_fast_math"]
    extra_cflags = ["-O3"]

    _EXT_MODULE = load(
        name="ray_scoring_cuda",
        sources=[cpp_src, cuda_src],
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        build_directory=build_dir,
        verbose=False,
    )
    return _EXT_MODULE


def sphere_blocked_mask(
    camera_positions: np.ndarray,
    points: np.ndarray,
    spheres: np.ndarray,
    min_distance: float = DEFAULT_CAMERA_NEAR,
    device: str = "cuda",
    chunk: int = 8,
) -> np.ndarray:
    """Which camera-to-point lines are blocked by that camera's own spheres.

    Spheres holding the camera itself cannot occlude it, so any sphere containing the camera is skipped.

    Args:
        camera_positions: (M, 3) camera positions in world coordinates.
        points: (N, 3) target points in world coordinates.
        spheres: (M, S, 4) spheres per camera as (x, y, z, radius); radius <= 0 is ignored.
        min_distance: Nearest distance along the ray that can block, the camera near plane.

    Returns:
        blocked: (M, N) boolean mask.
    """
    cameras = torch.as_tensor(camera_positions, dtype=torch.float32, device=device)
    targets = torch.as_tensor(points, dtype=torch.float32, device=device)
    all_spheres = torch.as_tensor(spheres, dtype=torch.float32, device=device)
    blocked = torch.zeros((len(cameras), len(targets)), dtype=torch.bool, device=device)

    for start in range(0, len(cameras), chunk):
        eyes = cameras[start:start + chunk, None, :]
        rays = targets[None, :, :] - eyes
        lengths = torch.linalg.norm(rays, dim=-1)
        directions = rays / lengths[..., None]

        centers, radii = all_spheres[start:start + chunk, :, :3], all_spheres[start:start + chunk, :, 3]
        to_centers = centers[:, None, :, :] - eyes[..., None, :]
        holds_camera = (to_centers[:, 0] ** 2).sum(-1) <= radii ** 2
        along_ray = torch.einsum("mnsd,mnd->mns", to_centers, directions)
        nearest = along_ray.clamp(min=min_distance).minimum(lengths[..., None])
        square_distance = (to_centers ** 2).sum(-1) - 2.0 * nearest * along_ray + nearest ** 2
        usable = (radii > 0) & ~holds_camera
        blocked[start:start + chunk] = ((square_distance < radii[:, None, :] ** 2) & usable[:, None, :]).any(-1)

    return blocked.cpu().numpy()


def score_candidate_views(
    candidate_positions: np.ndarray,
    unseen_points: np.ndarray,
    unseen_normals: np.ndarray,
    triangles: np.ndarray,
    backface_margin: float = DEFAULT_BACKFACE_MARGIN,
    occlusion_epsilon_m: float = DEFAULT_OCCLUSION_EPSILON_M,
    device: str = "cuda",
    blocker_spheres: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Score candidate camera positions against unseen surface points.

    Args:
        candidate_positions: (M, 3) camera positions in world coordinates.
        unseen_points: (N, 3) surface points to test visibility for.
        unseen_normals: (N, 3) outward-facing surface normals.
        triangles: (T, 3, 3) object mesh triangles in world frame.
        backface_margin: Cosine threshold for front-facing test.
        occlusion_epsilon_m: Small offset margin (meters) to avoid self-occlusion.
        device: CUDA device identifier.
        blocker_spheres: Optional (M, S, 4) spheres per candidate, typically the arm holding the camera.

    Returns:
        scores: (M,) int32 array of visible unseen points per candidate view.
        visibility_mask: (M, N) uint8 boolean mask (1 = visible, 0 = occluded/backface).
    """
    if len(candidate_positions) == 0:
        return np.zeros(0, dtype=np.int32), np.zeros((0, len(unseen_points)), dtype=np.uint8)

    if len(unseen_points) == 0:
        return np.zeros(len(candidate_positions), dtype=np.int32), np.zeros(
            (len(candidate_positions), 0), dtype=np.uint8
        )

    module = _get_cuda_module()

    # Move tensors to GPU
    t_cams = torch.as_tensor(candidate_positions, dtype=torch.float32, device=device)
    t_pts = torch.as_tensor(unseen_points, dtype=torch.float32, device=device)
    t_nrm = torch.as_tensor(unseen_normals, dtype=torch.float32, device=device)
    t_tri = torch.as_tensor(triangles, dtype=torch.float32, device=device)

    vis_mask, scores = module.score_candidate_views_cuda(
        t_cams, t_pts, t_nrm, t_tri, float(backface_margin), float(occlusion_epsilon_m)
    )

    if blocker_spheres is None:
        return scores.cpu().numpy(), vis_mask.cpu().numpy()

    visible = vis_mask.cpu().numpy().astype(bool)
    visible &= ~sphere_blocked_mask(candidate_positions, unseen_points, blocker_spheres, device=device)
    return visible.sum(axis=1).astype(np.int32), visible.astype(np.uint8)
