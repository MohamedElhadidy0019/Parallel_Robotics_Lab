"""Backwards-compatible sim_env module pointing to sim.env.SteveSimEnv."""

from nbv_core.config import ASSET_PATH, DEFAULT_YCB_OBJECT, PROJECT_ROOT, YCB_ROOT
from sim.env import SteveSimEnv, ycb_names, ycb_urdf

# Backwards-compatible alias
SimEnv = SteveSimEnv

__all__ = [
    "ASSET_PATH",
    "DEFAULT_YCB_OBJECT",
    "PROJECT_ROOT",
    "SimEnv",
    "SteveSimEnv",
    "YCB_ROOT",
    "ycb_names",
    "ycb_urdf",
]
