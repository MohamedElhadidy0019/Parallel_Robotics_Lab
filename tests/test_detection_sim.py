"""Detection on real start-pose frames rendered in PyBullet, checked against the CAD box. Needs a GPU for IK."""

import os

import numpy as np
import pytest

from nbv_planner.detection import detect_object
from nbv_planner.segmentation import SAM2_TINY_CHECKPOINT
from scripts.detect_start_frame import capture_start_frame, footprint, ground_truth_box
from sim.env import SteveSimEnv

OBJECTS = ["YcbMustardBottle", "YcbCrackerBox", "YcbChipsCan", "YcbBleachCleanser", "YcbTomatoSoupCan", "YcbGelatinBox"]


@pytest.fixture(scope="module", params=["depth", "sam"])
def segmenter(request):
    if request.param == "depth":
        return None
    if not os.path.isfile(SAM2_TINY_CHECKPOINT):
        pytest.skip(f"SAM2 checkpoint missing: {SAM2_TINY_CHECKPOINT}")
    from nbv_planner.segmentation import Sam2Segmenter
    return Sam2Segmenter()


@pytest.mark.parametrize("name", OBJECTS)
def test_start_frame_detection_matches_cad_box(name, segmenter):
    env = SteveSimEnv(render=False, ycb_object=name)
    try:
        detection = detect_object(capture_start_frame(env), segmenter)
        truth = ground_truth_box(env)
    finally:
        env.close()

    box = detection.box
    estimated_footprint, true_footprint = footprint(box), footprint(truth)
    assert np.linalg.norm(box.center - truth.center) < 0.015
    assert box.size[2] == pytest.approx(truth.size[2], abs=0.005)
    assert estimated_footprint[0] == pytest.approx(true_footprint[0], abs=0.012)
    # The far side is never seen from one view, so the short side can only be underestimated.
    assert 0.6 * true_footprint[1] <= estimated_footprint[1] <= true_footprint[1] + 0.005


def test_ring_scan_refines_box_to_cad():
    from scripts.scan_object import run_scan

    env = SteveSimEnv(render=False, ycb_object="YcbMustardBottle")
    try:
        estimate, rows = run_scan(env, segmenter=None, n_views=8, teleport=True)
    finally:
        env.close()

    first, last = rows[0], rows[-1]
    assert len(rows) >= 5
    assert last["center_err_mm"] < 2.0 < first["center_err_mm"]
    assert abs(last["size_mm"][1] - last["true_size_mm"][1]) < 3.0
    assert last["completeness"] > 0.95
