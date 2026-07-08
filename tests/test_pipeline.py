# Smoke test for the hamer-mini pipeline.
# Run with: python tests/test_pipeline.py [image_path]
# Note: downloads ~9 GB of weights on first run (HaMeR, ViTPose, ViTDet).
import sys

import cv2
import numpy as np
import torch

from hamer_mini import HaMeRHandPose3dEstimationPipeline


def test_pipeline(image_path: str = "example_data/test1.jpg"):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = HaMeRHandPose3dEstimationPipeline(device=device, dtype=dtype)

    image = cv2.imread(image_path)
    assert image is not None, f"could not read {image_path}"
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    outputs = pipe.predict(image, return_vertices_2d=True)
    assert len(outputs) > 0, "expected at least one hand in the test image"

    for ret in outputs:
        assert len(ret["hand_bbox"]) == 4
        assert 0.0 <= ret["hand_score"] <= 1.0
        assert ret["is_right"] in (0.0, 1.0)
        preds = ret["hamer_preds"]
        assert preds["global_orient"].shape == (1, 1, 3)
        assert preds["hand_pose"].shape == (1, 15, 3)
        assert preds["betas"].shape == (1, 10)
        assert preds["pred_cam"].shape == (1, 3)
        assert preds["pred_cam_t_full"].shape == (1, 3)
        assert preds["pred_keypoints_3d"].shape == (1, 21, 3)
        assert preds["pred_vertices"].shape == (1, 778, 3)
        assert preds["pred_keypoints_2d"].shape == (1, 21, 2)
        assert preds["pred_vertices_2d"].shape == (1, 778, 2)
        assert np.isfinite(preds["pred_keypoints_2d"]).all()

    assert pipe.mano_faces.shape[1] == 3
    print(f"OK: {len(outputs)} hand(s) detected, all output shapes verified")


if __name__ == "__main__":
    test_pipeline(*sys.argv[1:2])
