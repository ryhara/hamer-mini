# Original code: https://github.com/geopavlakos/hamer/blob/main/vitpose_model.py
# Modified to take the checkpoint path as an argument and to load the vendored
# ViTPose config shipped with this package; visualization helpers are removed.
from __future__ import annotations

import contextlib
import io
import os
import warnings

import numpy as np
import torch

# The old mmcv/mmpose (ViTPose fork) versions pinned by this package emit noisy
# deprecation warnings ('pkg_resources is deprecated', 'Fail to import
# MultiScaleDeformableAttention', 'timm.models.layers is deprecated') and print
# 'apex is not installed' (stdout) and, on CPU-only machines, 'No CUDA runtime
# is found' (stderr, via torch.utils.cpp_extension) at import time; silence them.
with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), \
        contextlib.redirect_stderr(io.StringIO()):
    warnings.simplefilter("ignore")
    from mmpose.apis import inference_top_down_pose_model, init_pose_model, process_mmdet_results

# ViTPose+-G (multi-task train, COCO wholebody head)
VITPOSE_CONFIG = os.path.join(os.path.dirname(__file__),
                              "configs", "vitpose", "ViTPose_huge_wholebody_256x192.py")


class ViTPoseModel(object):
    """Wholebody 2D keypoint detector used to derive the hand bounding boxes."""

    def __init__(self, checkpoint_path: str, device: str | torch.device):
        self.device = torch.device(device)
        # mmcv prints 'Use load_from_local loader' and a long 'unexpected key'
        # dump to stdout while loading: the wholebody checkpoint is a ViTPose+
        # MoE checkpoint whose expert weights are absent from (and not needed by)
        # the plain ViT backbone used here, exactly as in the original HaMeR demo.
        with contextlib.redirect_stdout(io.StringIO()):
            self.model = init_pose_model(VITPOSE_CONFIG, checkpoint_path, device=self.device)

    def predict_pose(
            self,
            image: np.ndarray,
            det_results: list[np.ndarray],
            box_score_threshold: float = 0.5) -> list[dict[str, np.ndarray]]:
        """
        Args:
            image (np.ndarray): Input RGB image of shape (H, W, 3).
            det_results (list): Person detections [(N, 5) array of x1, y1, x2, y2, score].
            box_score_threshold (float): Minimum person detection score.
        Returns:
            list: One dict per person with 'bbox' (5,) and 'keypoints' (133, 3)
                wholebody keypoints [x, y, score].
        """
        image = image[:, :, ::-1]  # RGB -> BGR
        person_results = process_mmdet_results(det_results, 1)
        out, _ = inference_top_down_pose_model(self.model,
                                               image,
                                               person_results=person_results,
                                               bbox_thr=box_score_threshold,
                                               format='xyxy')
        return out
