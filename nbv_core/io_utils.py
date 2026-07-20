"""Small save/inspection helpers shared by the debug/test scripts."""
import numpy as np
from PIL import Image
import matplotlib.cm as cm


def _colorize_depth(depth_mm: np.ndarray, near_m: float, far_m: float) -> np.ndarray:
    """(H, W) depth in millimeters -> (H, W, 3) uint8 viridis image, normalized to [near_m, far_m]."""
    depth_m = depth_mm.astype(np.float64) / 1000.0
    normalized_depth = np.clip((depth_m - near_m) / (far_m - near_m), 0.0, 1.0)
    return (cm.viridis(normalized_depth)[:, :, :3] * 255).astype(np.uint8)


def save_depth_image(depth_mm: np.ndarray, path: str, near_m: float = 0.07, far_m: float = 1.5) -> None:
    """Save a depth image as a colorized PNG (viridis: dark = near, bright = far)."""
    Image.fromarray(_colorize_depth(depth_mm, near_m, far_m)).save(path)


def save_rgb_depth_composite(
    rgb: np.ndarray,
    depth_mm: np.ndarray,
    path: str,
    near_m: float = 0.07,
    far_m: float = 1.5,
) -> None:
    """
    Save a single PNG with the RGB frame on the left and a colorized depth
    map on the right - lets you visually check the raw sensor output
    directly, independent of any back-projection/reconstruction math.
    """
    depth_image = _colorize_depth(depth_mm, near_m, far_m)
    rgb_image = np.asarray(rgb)[:, :, :3].astype(np.uint8)
    composite = np.concatenate([rgb_image, depth_image], axis=1)
    Image.fromarray(composite).save(path)
