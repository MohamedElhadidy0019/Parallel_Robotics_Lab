"""Autonomous Next-Best-View scanning in PyBullet.

Usage:
    python main.py YcbMustardBottle --views 8 --viz
    python main.py YcbMustardBottle --mode scan --segmenter sam
    python main.py YcbMustardBottle --start-pos 0.5 0.0 1.15 --look-at 0.785 0.0 0.85
"""

import argparse
import io
import os
import sys
import warnings


def _silence_c_output():
    """Route low-level C/C++ stdout and stderr (PyBullet/OpenGL threads) to /dev/null while keeping Python sys.stdout / sys.stderr intact."""
    if getattr(sys, "_c_output_silenced", False):
        return
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        real_out = os.dup(1)
        real_err = os.dup(2)
        sys.stdout = io.TextIOWrapper(open(real_out, "wb", buffering=0), encoding="utf-8", write_through=True)
        sys.stderr = io.TextIOWrapper(open(real_err, "wb", buffering=0), encoding="utf-8", write_through=True)
        null_fd = os.open(os.devnull, os.O_RDWR)
        os.dup2(null_fd, 1)
        os.dup2(null_fd, 2)
        os.close(null_fd)
        sys._c_output_silenced = True
    except Exception:
        pass


_silence_c_output()

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5;8.0;8.6;8.9;9.0")
os.environ.setdefault("RERUN_LOG", "warn")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", module="gymnasium")
warnings.filterwarnings("ignore", module="torch")
warnings.filterwarnings("ignore", module="rerun")

from nbv_planner.config import (
    DEFAULT_YCB_OBJECT,
    START_CAMERA_POSITION_BASE,
    START_LOOK_AT_BASE,
    ycb_names,
)
from nbv_planner.pipeline import run_inspection
from nbv_planner.viz import NBVVisualizer
from sim.env import SteveSimEnv


def run_nbv_scan(obj_name: str = DEFAULT_YCB_OBJECT, max_views: int = 8, target_coverage: float = 0.95,
                 n_surface_samples: int = 4000, gui: bool = False, viz: bool = False,
                 start_position_base=START_CAMERA_POSITION_BASE, look_at_base=START_LOOK_AT_BASE,
                 mode: str = "cad", segmenter_name: str = "sam", scan_views: int = 8) -> dict:
    """Run the inspection pipeline against the PyBullet robot."""
    env = SteveSimEnv(render=gui, ycb_object=obj_name)
    try:
        return run_inspection(
            env,
            NBVVisualizer(obj_name, enabled=viz, mode=mode),
            object_name=obj_name,
            mode=mode,
            max_views=max_views,
            target_coverage=target_coverage,
            n_surface_samples=n_surface_samples,
            start_position_base=start_position_base,
            look_at_base=look_at_base,
            segmenter_name=segmenter_name,
            scan_views=scan_views,
        )
    finally:
        if gui:
            try:
                input("[GUI] Scan complete. Press Enter to close PyBullet window...")
            except (EOFError, KeyboardInterrupt):
                pass
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("object", nargs="?", default=DEFAULT_YCB_OBJECT, choices=ycb_names())
    parser.add_argument("--views", "--frames", type=int, default=8, help="Maximum number of scan viewpoints")
    parser.add_argument("--target-cov", type=float, default=0.95, help="Target coverage fraction (0-1)")
    parser.add_argument("--samples", type=int, default=4000, help="Surface sampling resolution")
    parser.add_argument("--gui", action="store_true", help="Enable PyBullet live simulation window")
    parser.add_argument("--rerun-viz", "--viz", dest="viz", action="store_true", help="Enable live Rerun 3D viewer")
    parser.add_argument(
        "--start-pos", nargs=3, type=float, metavar=("X", "Y", "Z"), default=START_CAMERA_POSITION_BASE,
        help="Start pose camera position, robot base frame (m)",
    )
    parser.add_argument(
        "--look-at", nargs=3, type=float, metavar=("X", "Y", "Z"), default=START_LOOK_AT_BASE,
        help="Point the camera faces at the start pose (roughly the object), robot base frame (m)",
    )
    parser.add_argument(
        "--mode", choices=["cad", "scan", "both"], default="cad",
        help="cad: CAD mesh only; scan: no CAD, mesh built from segmented views; both: scan, check the CAD against it, then use the CAD",
    )
    parser.add_argument("--segmenter", choices=["sam", "depth"], default="sam", help="Segmentation for scan and both modes")
    parser.add_argument("--scan-views", type=int, default=8, help="Ring views for scan and both modes")
    args = parser.parse_args()

    run_nbv_scan(
        obj_name=args.object,
        max_views=args.views,
        target_coverage=args.target_cov,
        n_surface_samples=args.samples,
        gui=args.gui,
        viz=args.viz,
        start_position_base=args.start_pos,
        look_at_base=args.look_at,
        mode=args.mode,
        segmenter_name=args.segmenter,
        scan_views=args.scan_views,
    )
