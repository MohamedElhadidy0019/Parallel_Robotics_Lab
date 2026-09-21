"""Estimate an unknown object from the start pose, refine it over a ring of views, and compare with the CAD mesh.

Usage:
    python scripts/scan_object.py YcbMustardBottle --sam
    python scripts/scan_object.py YcbMustardBottle --teleport
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nbv_planner.config import (
    BASE_EXCLUSION_HEIGHT_M,
    DEFAULT_YCB_OBJECT,
    START_CAMERA_POSITION_BASE,
    START_LOOK_AT_BASE,
    START_SAFETY_RADIUS_M,
    ycb_names,
)
from nbv_planner.motion_planning import move_camera
from nbv_planner.object_estimate import ObjectEstimate
from nbv_planner.object_scan import box_world_config, obstacle_world_config, scan_object
from nbv_planner.start_pose import start_camera_pose_world
from scripts.detect_start_frame import footprint, ground_truth_box, overlay
from sim.env import SteveSimEnv

COMPLETENESS_DISTANCE_M = 0.005


def go_to(env: SteveSimEnv, position, quaternion, world_config, teleport: bool) -> bool:
    if teleport:
        try:
            env.teleport_camera(position, quaternion)
            return True
        except ValueError:
            return False
    return move_camera(env, position, quaternion, world_config)


class MeshComparison:
    def __init__(self, env: SteveSimEnv, samples: int = 200_000) -> None:
        mesh = env.object_mesh_world()
        surface = np.asarray(mesh.sample(samples))
        self.truth = ground_truth_box(env)
        self.surface_tree = cKDTree(surface)
        visible = surface[:, 2] >= env.table_surface_z + BASE_EXCLUSION_HEIGHT_M
        self.visible_surface = surface[visible][::10]

    def row(self, estimate: ObjectEstimate) -> dict:
        box, truth = estimate.box, self.truth
        to_surface, _ = self.surface_tree.query(estimate.points)
        to_points, _ = cKDTree(estimate.points).query(self.visible_surface)
        return {
            "center_err_mm": float(np.linalg.norm(box.center - truth.center) * 1000),
            "size_mm": (np.r_[footprint(box), box.size[2]] * 1000).round(1).tolist(),
            "true_size_mm": (np.r_[footprint(truth), truth.size[2]] * 1000).round(1).tolist(),
            "point_to_mesh_mean_mm": float(to_surface.mean() * 1000),
            "point_to_mesh_p95_mm": float(np.percentile(to_surface, 95) * 1000),
            "completeness": float((to_points <= COMPLETENESS_DISTANCE_M).mean()),
        }


def print_row(label: str, row: dict, shift: float) -> None:
    length, width, height = row["size_mm"]
    true_length, true_width, true_height = row["true_size_mm"]
    shift_text = "   -" if np.isinf(shift) else f"{shift * 1000:4.1f}"
    print(f"{label:<14} {row['center_err_mm']:6.1f} {shift_text:>6} "
          f"{length:4.0f}/{true_length:<4.0f} {width:4.0f}/{true_width:<4.0f} {height:4.0f}/{true_height:<4.0f} "
          f"{row['point_to_mesh_mean_mm']:5.1f} {row['point_to_mesh_p95_mm']:5.1f} {row['completeness'] * 100:7.1f}%")


def run_scan(env: SteveSimEnv, segmenter, n_views: int, teleport: bool, out_dir: str | None = None):
    """Scan from the start pose and compare every frame with the CAD mesh. Returns (estimate, per-frame rows)."""
    comparison = MeshComparison(env)
    base_position, base_quaternion = env.base_pose()
    start_position, start_quaternion, look_at = start_camera_pose_world(
        START_CAMERA_POSITION_BASE, START_LOOK_AT_BASE, base_position, base_quaternion
    )
    start_obstacle = obstacle_world_config(env, look_at, 0.0, np.full(3, 2.0 * START_SAFETY_RADIUS_M))
    if not go_to(env, start_position, start_quaternion, start_obstacle, teleport):
        raise RuntimeError("Could not reach the start pose")

    rows = []

    def record(label, observation, detection, estimate):
        if out_dir is not None:
            overlay(observation, detection, comparison.truth).save(os.path.join(out_dir, f"frame_{label.replace(' ', '_')}.png"))
        rows.append(comparison.row(estimate))
        print_row(label, rows[-1], estimate.center_shift)

    print(f"{'frame':<14} {'ctr mm':>6} {'shift':>6} {'long':>9} {'short':>9} {'height':>9} "
          f"{'pt-mesh':>5} {'p95':>5} {'complete':>8}")
    estimate = scan_object(env, segmenter, lambda p, q, w: go_to(env, p, q, w, teleport), n_views, on_frame=record)

    if not go_to(env, start_position, start_quaternion, box_world_config(env, estimate.box), teleport):
        print("Warning: could not return to the start pose")
    return estimate, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("object", nargs="?", default=DEFAULT_YCB_OBJECT, choices=ycb_names())
    parser.add_argument("--sam", action="store_true", help="Segment with SAM2-tiny instead of the depth cluster")
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--teleport", action="store_true", help="Set joints by IK instead of cuRobo motion")
    parser.add_argument("--out", default="captures/scan")
    args = parser.parse_args()

    segmenter = None
    if args.sam:
        from nbv_planner.segmentation import Sam2Segmenter
        segmenter = Sam2Segmenter()

    out_dir = os.path.join(args.out, args.object)
    os.makedirs(out_dir, exist_ok=True)
    env = SteveSimEnv(render=False, ycb_object=args.object)
    try:
        estimate, rows = run_scan(env, segmenter, args.views, args.teleport, out_dir)
    finally:
        env.close()

    import open3d as o3d

    o3d.io.write_point_cloud(os.path.join(out_dir, "points.ply"),
                             o3d.geometry.PointCloud(o3d.utility.Vector3dVector(estimate.points)))
    box = estimate.box
    with open(os.path.join(out_dir, "box.json"), "w") as f:
        json.dump({"center": box.center.tolist(), "size": box.size.tolist(), "yaw": box.yaw, "frames": rows}, f, indent=2)
    print(f"Saved {out_dir}/points.ply, box.json, frame overlays")


if __name__ == "__main__":
    main()
