# Original code: https://github.com/geopavlakos/hamer/blob/main/demo.py
# Pipeline structure follows: https://github.com/warmshao/WiLoR-mini/blob/main/wilor_mini/pipelines/wilor_hand_pose3d_estimation_pipeline.py
# Hand detection is the same two-stage approach as the original HaMeR demo:
# a detectron2 person detector followed by the ViTPose wholebody keypoint model,
# from which the hand bounding boxes and handedness are derived.
import logging
import os
import warnings

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from skimage.filters import gaussian

from ..detectors.body_detector import build_body_detector
from ..detectors.vitpose_model import ViTPoseModel
from ..models.hamer import HAMER
from ..utils import utils
from ..utils.logger import get_logger

# Repository hosting the official HaMeR demo data (checkpoint, MANO mean parameters,
# ViTPose wholebody checkpoint)
HAMER_REPO_ID = "geopavlakos/HaMeR"

# Aspect ratio (width, height) of the bounding box expected by the ViT backbone
BBOX_SHAPE = [192, 256]

# A hand is kept when more than this many of its 21 keypoints are confident
MIN_VALID_HAND_KEYPOINTS = 3


class HaMeRHandPose3dEstimationPipeline:
    """3D hand pose estimation pipeline based on HaMeR.

    Following the original HaMeR demo, people are detected with a detectron2 detector,
    their wholebody keypoints are estimated with ViTPose, hand bounding boxes are
    derived from the hand keypoints, and the MANO hand mesh is regressed for each
    hand with the HaMeR model.
    """

    def __init__(self, **kwargs):
        """
        Keyword Args:
            device (torch.device): Device to run on. Defaults to CPU.
            dtype (torch.dtype): HaMeR model dtype (e.g. torch.float16 on GPU).
                Defaults to torch.float32. The detectors always run in float32.
            verbose (bool): Whether to log progress messages. Defaults to False.
            body_detector (str): Person detector, 'vitdet' (default) or 'regnety' (lighter).
            body_conf (float): Default person detection score threshold. Defaults to 0.5.
            hand_conf (float): Default hand keypoint confidence threshold used to
                derive the hand bounding boxes. Defaults to 0.5.
            focal_length (float): Focal length used by HaMeR, in crop pixels. Defaults to 5000.
            pretrained_dir (str): Directory to store the downloaded weights. Defaults to
                the HAMER_MINI_PRETRAINED_DIR environment variable or ~/.cache/hamer_mini.
            mano_model_path (str): Path to the MANO right hand model file (MANO_RIGHT.pkl).
                Defaults to <pretrained_dir>/mano/MANO_RIGHT.pkl. This file is NOT
                downloaded automatically; obtain it from https://mano.is.tue.mpg.de after
                accepting the MANO license and place it there yourself.
            hamer_repo_id (str): Hugging Face Space hosting the HaMeR demo data.
        """
        self.verbose = kwargs.get("verbose", False)
        self.logger = get_logger(self.__class__.__name__,
                                 lv=logging.INFO if self.verbose else logging.ERROR)
        self.device = kwargs.get("device", torch.device("cpu"))
        self.dtype = kwargs.get("dtype", torch.float32)
        self.body_conf = kwargs.get("body_conf", 0.5)
        self.hand_conf = kwargs.get("hand_conf", 0.5)
        self.FOCAL_LENGTH = kwargs.get("focal_length", 5000)
        self.IMAGE_SIZE = 256
        self.init_models(**kwargs)

    def init_models(self, **kwargs):
        hamer_repo_id = kwargs.get("hamer_repo_id", HAMER_REPO_ID)
        pretrained_dir = kwargs.get(
            "pretrained_dir",
            os.environ.get("HAMER_MINI_PRETRAINED_DIR",
                           os.path.join(os.path.expanduser("~"), ".cache", "hamer_mini")))
        os.makedirs(pretrained_dir, exist_ok=True)

        # The MANO hand model is licensed separately (https://mano.is.tue.mpg.de) and is
        # therefore not downloaded automatically; the user must place it manually.
        mano_model_path = kwargs.get("mano_model_path",
                                     os.path.join(pretrained_dir, "mano", "MANO_RIGHT.pkl"))
        if not os.path.exists(mano_model_path):
            raise FileNotFoundError(
                f"MANO model not found: {mano_model_path}\n"
                "The MANO hand model cannot be redistributed with hamer-mini. Please:\n"
                "  1. Register at https://mano.is.tue.mpg.de and accept the MANO license\n"
                "  2. Download 'Models & Code' (mano_v1_2.zip)\n"
                f"  3. Copy mano_v1_2/models/MANO_RIGHT.pkl to {mano_model_path}\n"
                "or pass its location with the 'mano_model_path' keyword argument.")

        # File layout of the official HaMeR demo data inside the Hugging Face Space
        mano_mean_params_path = os.path.join(pretrained_dir, "_DATA", "data", "mano_mean_params.npz")
        hamer_ckpt_path = os.path.join(pretrained_dir, "_DATA", "hamer_ckpts", "checkpoints", "hamer.ckpt")
        vitpose_ckpt_path = os.path.join(pretrained_dir, "_DATA", "vitpose_ckpts",
                                         "vitpose+_huge", "wholebody.pth")
        for filename, path in [("_DATA/data/mano_mean_params.npz", mano_mean_params_path),
                               ("_DATA/hamer_ckpts/checkpoints/hamer.ckpt", hamer_ckpt_path),
                               ("_DATA/vitpose_ckpts/vitpose+_huge/wholebody.pth", vitpose_ckpt_path)]:
            if not os.path.exists(path):
                self.logger.info(f"downloading {filename} from Hugging Face Space {hamer_repo_id}")
                hf_hub_download(repo_id=hamer_repo_id, repo_type="space", filename=filename,
                                local_dir=pretrained_dir)

        self.logger.info("loading HaMeR model >>> ")
        self.hamer_model = HAMER(mano_model_path=mano_model_path,
                                 mano_mean_params_path=mano_mean_params_path,
                                 focal_length=self.FOCAL_LENGTH,
                                 image_size=self.IMAGE_SIZE)
        # The checkpoint is a pytorch-lightning checkpoint that stores the training
        # config as pickled objects, so weights_only cannot be used.
        state_dict = torch.load(hamer_ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
        missing, unexpected = self.hamer_model.load_state_dict(state_dict, strict=False)
        if missing:
            self.logger.warning(f"missing keys when loading the HaMeR checkpoint: {missing}")
        self.hamer_model.eval()
        self.hamer_model.to(self.device, dtype=self.dtype)
        # The MANO layer always runs in float32 (HAMER.forward casts its inputs),
        # so undo the dtype conversion for it
        self.hamer_model.mano.float()

        self.logger.info("loading person detection model >>> ")
        self.body_detector = build_body_detector(kwargs.get("body_detector", "vitdet"), self.device)

        self.logger.info("loading ViTPose keypoint detection model >>> ")
        self.vitpose_model = ViTPoseModel(vitpose_ckpt_path, self.device)

    @property
    def mano_faces(self) -> np.ndarray:
        """Triangle faces of the MANO right hand mesh, shape (1538, 3).

        For left hands (is_right == 0) flip the winding order with faces[:, [0, 2, 1]].
        """
        return self.hamer_model.mano.faces

    @torch.no_grad()
    def predict(self, image: np.ndarray, body_conf: float = None, hand_conf: float = None,
                rescale_factor: float = 2.0, return_vertices_2d: bool = False) -> list:
        """
        Detect hands and estimate their 3D pose.
        Args:
            image (np.ndarray): Input RGB image of shape (H, W, 3).
            body_conf (float): Person detection score threshold. Defaults to the
                value passed to the constructor (0.5 if not set).
            hand_conf (float): Hand keypoint confidence threshold used to derive the
                hand bounding boxes. Defaults to the value passed to the constructor
                (0.5 if not set).
            rescale_factor (float): Padding factor applied to the derived hand boxes.
            return_vertices_2d (bool): Whether to also project the 778 mesh vertices
                onto the image ('pred_vertices_2d').
        Returns:
            list: One dict per detected hand:
                - 'hand_bbox' (list): Detected bounding box [x1, y1, x2, y2] in pixels
                  (tight box around the confident hand keypoints).
                - 'crop_bbox' (list): Square crop region [x1, y1, x2, y2] in pixels that
                  was actually fed to HaMeR (hand_bbox padded by rescale_factor and
                  expanded to the model aspect ratio; may extend beyond the image).
                - 'hand_score' (float): Mean confidence of the confident hand keypoints, in [0, 1].
                - 'is_right' (float): Handedness, 1.0 for right and 0.0 for left.
                - 'hamer_preds' (dict): HaMeR predictions, see _estimate().
        """
        if body_conf is None:
            body_conf = self.body_conf
        if hand_conf is None:
            hand_conf = self.hand_conf

        self.logger.info("start person detection >>> ")
        with warnings.catch_warnings():
            # detectron2 still calls torch.meshgrid without the indexing argument
            warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")
            det_out = self.body_detector(image[:, :, ::-1])  # the detector takes BGR input
        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > body_conf)
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()
        if len(pred_bboxes) == 0:
            self.logger.warning("No person detected!")
            return []

        self.logger.info("start wholebody keypoint detection >>> ")
        vitposes_out = self.vitpose_model.predict_pose(
            image,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        # Derive hand bounding boxes from the hand keypoints of each person.
        # The last 42 wholebody keypoints are the left and right hand keypoints.
        bboxes = []
        is_rights = []
        scores = []
        for vitposes in vitposes_out:
            for keyp, right in [(vitposes['keypoints'][-42:-21], 0),
                                (vitposes['keypoints'][-21:], 1)]:
                # Rejecting not confident detections
                valid = keyp[:, 2] > hand_conf
                if sum(valid) > MIN_VALID_HAND_KEYPOINTS:
                    bboxes.append([keyp[valid, 0].min(), keyp[valid, 1].min(),
                                   keyp[valid, 0].max(), keyp[valid, 1].max()])
                    is_rights.append(float(right))
                    scores.append(float(keyp[valid, 2].mean()))

        if len(bboxes) == 0:
            self.logger.warning("No hand detected!")
            return []

        return self._estimate(image, np.stack(bboxes), is_rights, scores,
                              rescale_factor=rescale_factor,
                              return_vertices_2d=return_vertices_2d)

    @torch.no_grad()
    def predict_with_bboxes(self, image: np.ndarray, bboxes: np.ndarray, is_rights,
                            rescale_factor: float = 2.0, return_vertices_2d: bool = False) -> list:
        """
        Estimate the 3D pose of hands with externally supplied bounding boxes,
        skipping the detection stage.
        Args:
            image (np.ndarray): Input RGB image of shape (H, W, 3).
            bboxes (np.ndarray): Hand bounding boxes of shape (N, 4) as [x1, y1, x2, y2].
            is_rights: Handedness per box, 1 for right and 0 for left.
            rescale_factor (float): Padding factor applied to the hand boxes.
            return_vertices_2d (bool): Whether to also project the 778 mesh vertices
                onto the image ('pred_vertices_2d').
        Returns:
            list: Same structure as predict(); 'hand_score' is None.
        """
        if len(bboxes) == 0:
            self.logger.warning("No hand bounding box given!")
            return []
        bboxes = np.asarray(bboxes, dtype=np.float32)
        return self._estimate(image, bboxes, list(is_rights), [None] * len(bboxes),
                              rescale_factor=rescale_factor,
                              return_vertices_2d=return_vertices_2d)

    def _estimate(self, image: np.ndarray, bboxes: np.ndarray, is_rights: list, scores: list,
                  rescale_factor: float, return_vertices_2d: bool) -> list:
        """
        Run HaMeR on the given hand boxes and post-process the outputs.

        The 'hamer_preds' dict of each returned entry contains:
            - 'global_orient' (1, 1, 3): Global hand orientation as axis-angle.
            - 'hand_pose' (1, 15, 3): MANO hand pose as axis-angle.
            - 'betas' (1, 10): MANO shape parameters.
            - 'pred_cam' (1, 3): Weak-perspective camera in the crop frame.
            - 'pred_cam_t_full' (1, 3): Camera translation in the full image frame.
            - 'scaled_focal_length' (float): Focal length scaled to the full image.
            - 'pred_keypoints_3d' (1, 21, 3): 3D joints (OpenPose ordering) in meters,
              relative to the camera after adding 'pred_cam_t_full'.
            - 'pred_vertices' (1, 778, 3): 3D MANO mesh vertices.
            - 'pred_keypoints_2d' (1, 21, 2): 2D joints in full image pixels.
            - 'pred_vertices_2d' (1, 778, 2): 2D mesh vertices in full image pixels
              (only when return_vertices_2d=True).
        """
        detect_rets = [{"hand_bbox": bboxes[i, :4].tolist(),
                        "hand_score": scores[i],
                        "is_right": is_rights[i]} for i in range(len(bboxes))]

        center = (bboxes[:, 2:4] + bboxes[:, 0:2]) / 2.0
        # Expand each padded box to the 192:256 aspect ratio expected by the model
        scale = rescale_factor * (bboxes[:, 2:4] - bboxes[:, 0:2])
        bbox_sizes = np.array([utils.expand_to_aspect_ratio(s, BBOX_SHAPE).max() for s in scale])
        # Square crop region actually fed to the model, in full image pixels
        for i in range(len(detect_rets)):
            half = bbox_sizes[i] / 2.0
            detect_rets[i]["crop_bbox"] = [float(center[i, 0] - half), float(center[i, 1] - half),
                                           float(center[i, 0] + half), float(center[i, 1] + half)]

        self.logger.info(f"start hand 3d pose estimation for {len(bboxes)} hand(s) >>> ")
        img_size = np.array([image.shape[1], image.shape[0]])
        patch_width = patch_height = self.IMAGE_SIZE
        img_patches = []
        for i in range(len(bboxes)):
            bbox_size = bbox_sizes[i]
            # Left hand crops are mirrored, since the model is right-hand only
            flip = is_rights[i] == 0

            cvimg = image.copy()
            # Blur image to avoid aliasing artifacts when downsampling large crops
            downsampling_factor = (bbox_size / patch_width) / 2.0
            if downsampling_factor > 1.1:
                cvimg = gaussian(cvimg, sigma=(downsampling_factor - 1) / 2,
                                 channel_axis=2, preserve_range=True)

            img_patch, _ = utils.generate_image_patch_cv2(cvimg,
                                                          center[i, 0], center[i, 1],
                                                          bbox_size, bbox_size,
                                                          patch_width, patch_height,
                                                          flip, 1.0, 0,
                                                          border_mode=cv2.BORDER_CONSTANT)
            img_patches.append(img_patch)

        img_patches = np.stack(img_patches).astype(np.float32)
        img_patches = torch.from_numpy(img_patches).to(device=self.device, dtype=self.dtype)
        hamer_output = self.hamer_model(img_patches)
        hamer_output = {k: v.cpu().float().numpy() for k, v in hamer_output.items()}

        scaled_focal_length = self.FOCAL_LENGTH / self.IMAGE_SIZE * img_size.max()
        for i in range(len(detect_rets)):
            preds = {key: val[[i]] for key, val in hamer_output.items()}
            pred_cam = preds["pred_cam"]
            right = is_rights[i]
            # Undo the horizontal mirroring applied to left hand crops
            multiplier = 2 * right - 1
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            if right == 0:
                preds["pred_keypoints_3d"][:, :, 0] = -preds["pred_keypoints_3d"][:, :, 0]
                preds["pred_vertices"][:, :, 0] = -preds["pred_vertices"][:, :, 0]
                preds["global_orient"] = np.concatenate(
                    (preds["global_orient"][:, :, 0:1], -preds["global_orient"][:, :, 1:3]), axis=-1)
                preds["hand_pose"] = np.concatenate(
                    (preds["hand_pose"][:, :, 0:1], -preds["hand_pose"][:, :, 1:3]), axis=-1)

            pred_cam_t_full = utils.cam_crop_to_full(pred_cam, center[[i]], bbox_sizes[i],
                                                     img_size[None], scaled_focal_length)
            preds["pred_cam_t_full"] = pred_cam_t_full
            preds["scaled_focal_length"] = scaled_focal_length

            # Project the 3D points to full image pixel coordinates
            focal = np.array([[scaled_focal_length, scaled_focal_length]])
            camera_center = img_size[None] / 2
            preds["pred_keypoints_2d"] = utils.perspective_projection(
                preds["pred_keypoints_3d"], translation=pred_cam_t_full,
                focal_length=focal, camera_center=camera_center)
            if return_vertices_2d:
                preds["pred_vertices_2d"] = utils.perspective_projection(
                    preds["pred_vertices"], translation=pred_cam_t_full,
                    focal_length=focal, camera_center=camera_center)
            detect_rets[i]["hamer_preds"] = preds

        self.logger.info("finish hand 3d pose estimation!")
        return detect_rets
