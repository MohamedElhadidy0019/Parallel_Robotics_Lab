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


def score_candidate_views(
    candidate_positions: np.ndarray,
    unseen_points: np.ndarray,
    unseen_normals: np.ndarray,
    triangles: np.ndarray,
    backface_margin: float = DEFAULT_BACKFACE_MARGIN,
    occlusion_epsilon_m: float = DEFAULT_OCCLUSION_EPSILON_M,
    device: str = "cuda",
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

    return scores.cpu().numpy(), vis_mask.cpu().numpy()
