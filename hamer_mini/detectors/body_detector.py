# Original code: https://github.com/geopavlakos/hamer/blob/main/demo.py (detector setup)
import os

import torch

from .utils_detectron2 import DefaultPredictor_Lazy

VITDET_CHECKPOINT_URL = ("https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
                         "cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl")


def build_body_detector(name: str, device: torch.device) -> DefaultPredictor_Lazy:
    """
    Build the person detector used to find people before hand keypoint detection.
    Args:
        name (str): 'vitdet' (ViTDet-H Cascade Mask R-CNN, more accurate) or
            'regnety' (RegNetY-4GF Mask R-CNN, lighter).
        device (torch.device): Device to run the detector on.
    Returns:
        DefaultPredictor_Lazy: Predictor taking a BGR image and returning detectron2 instances.
    """
    if name == "vitdet":
        from detectron2.config import LazyConfig
        cfg_path = os.path.join(os.path.dirname(__file__), "configs",
                                "cascade_mask_rcnn_vitdet_h_75ep.py")
        detectron2_cfg = LazyConfig.load(cfg_path)
        detectron2_cfg.train.init_checkpoint = VITDET_CHECKPOINT_URL
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        return DefaultPredictor_Lazy(detectron2_cfg, device=device)
    elif name == "regnety":
        from detectron2 import model_zoo
        detectron2_cfg = model_zoo.get_config(
            "new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py", trained=True)
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
        return DefaultPredictor_Lazy(detectron2_cfg, device=device)
    else:
        raise ValueError(f"Unknown body detector: {name} (expected 'vitdet' or 'regnety')")
