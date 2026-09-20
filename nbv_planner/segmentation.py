"""Box-prompted image segmentation."""

import os

import numpy as np

from nbv_planner.config import PROJECT_ROOT

SAM2_TINY_CHECKPOINT = os.path.join(PROJECT_ROOT, "checkpoints/sam2_hiera_tiny.pt")
SAM2_TINY_CONFIG = "sam2_hiera_t.yaml"


class Sam2Segmenter:
    def __init__(self, checkpoint: str = SAM2_TINY_CHECKPOINT, config: str = SAM2_TINY_CONFIG, device: str = "cuda"):
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint}")
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self._predictor = SAM2ImagePredictor(build_sam2(config, checkpoint, device=device))

    def segment(self, rgb: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        import torch

        with torch.inference_mode():
            self._predictor.set_image(rgb)
            masks, _, _ = self._predictor.predict(box=np.asarray(box, dtype=np.float32), multimask_output=False)
        return masks[0].astype(bool)
