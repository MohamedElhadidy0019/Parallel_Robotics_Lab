"""Detect the object in the start pose frame and compare it with the ground-truth CAD box.

Usage:
    python scripts/detect_start_frame.py YcbMustardBottle YcbCrackerBox --sam
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nbv_planner.config import DEFAULT_YCB_OBJECT, START_CAMERA_POSITION_BASE, START_LOOK_AT_BASE, ycb_names
from nbv_planner.detection import Detection, TableBox, TablePlane, detect_object, table_aligned_box
from nbv_planner.observations import Observation
from nbv_planner.start_pose import start_camera_pose_world
from sim.env import SteveSimEnv

BOX_EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1]


def capture_start_frame(env: SteveSimEnv) -> Observation:
    base_position, base_quaternion = env.base_pose()
    position, quaternion, _ = start_camera_pose_world(
        START_CAMERA_POSITION_BASE, START_LOOK_AT_BASE, base_position, base_quaternion
    )
    env.teleport_camera(position, quaternion)
    return env.capture_observation()


def ground_truth_box(env: SteveSimEnv) -> TableBox:
    surface = np.asarray(env.object_mesh_world().sample(20_000))
    table = TablePlane(np.array([0.0, 0.0, 1.0]), -env.table_surface_z, np.zeros((0, 3)))
    return table_aligned_box(surface, table)


def project(points_world: np.ndarray, observation: Observation) -> np.ndarray:
    transform, intr = observation.world_from_camera, observation.intrinsics
    camera = (points_world - transform[:3, 3]) @ transform[:3, :3]
    return np.c_[intr.fx * camera[:, 0] / camera[:, 2] + intr.cx, intr.fy * camera[:, 1] / camera[:, 2] + intr.cy]


def overlay(observation: Observation, detection: Detection, truth: TableBox) -> Image.Image:
    rgb = observation.rgb.astype(float)
    rgb[detection.mask] = 0.5 * rgb[detection.mask] + 0.5 * np.array([40, 220, 110])
    image = Image.fromarray(rgb.astype(np.uint8))
    draw = ImageDraw.Draw(image)
    draw.rectangle(detection.prompt_box, outline=(255, 200, 0), width=2)
    for box, color in ((truth, (230, 40, 40)), (detection.box, (40, 120, 255))):
        pixels = project(box.corners(), observation)
        for a, b in BOX_EDGES:
            draw.line([tuple(pixels[a]), tuple(pixels[b])], fill=color, width=2)
    return image


def footprint(box: TableBox) -> np.ndarray:
    return np.sort(box.size[:2])[::-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("objects", nargs="*", default=[DEFAULT_YCB_OBJECT], choices=ycb_names())
    parser.add_argument("--sam", action="store_true", help="Segment with SAM2-tiny instead of the depth cluster")
    parser.add_argument("--out", default="captures/detection")
    args = parser.parse_args()

    segmenter = None
    if args.sam:
        from nbv_planner.segmentation import Sam2Segmenter
        segmenter = Sam2Segmenter()

    os.makedirs(args.out, exist_ok=True)
    print(f"{'object':<20} {'center err mm':>13} {'height mm':>14} {'long side mm':>15} {'short side mm':>15} {'yaw err deg':>11}")
    for name in args.objects:
        env = SteveSimEnv(render=False, ycb_object=name)
        try:
            observation = capture_start_frame(env)
            detection = detect_object(observation, segmenter)
            truth = ground_truth_box(env)
        finally:
            env.close()

        box = detection.box
        center_error = np.linalg.norm(box.center - truth.center) * 1000
        yaw_error = np.degrees(abs((box.yaw - truth.yaw + np.pi / 2) % np.pi - np.pi / 2))
        est_fp, true_fp = footprint(box) * 1000, footprint(truth) * 1000
        print(f"{name:<20} {center_error:13.1f} {box.size[2] * 1000:6.0f} ({truth.size[2] * 1000:3.0f}) "
              f"{est_fp[0]:7.0f} ({true_fp[0]:3.0f}) {est_fp[1]:7.0f} ({true_fp[1]:3.0f}) {yaw_error:11.1f}")

        path = os.path.join(args.out, f"{name}_{'sam' if args.sam else 'depth'}.png")
        overlay(observation, detection, truth).save(path)
    print(f"Overlays: {args.out}  (yellow = SAM prompt, blue = detected box, red = ground truth)")


if __name__ == "__main__":
    main()
