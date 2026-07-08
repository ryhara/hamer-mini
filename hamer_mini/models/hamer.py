# Original code: https://github.com/geopavlakos/hamer/blob/main/hamer/models/hamer.py
# Librarization follows: https://github.com/warmshao/WiLoR-mini/blob/main/wilor_mini/models/wilor.py
# The pytorch-lightning LightningModule is converted to a plain nn.Module and the
# training members (losses, discriminator, renderers) are removed.
import numpy as np
import roma
import torch
from torch import nn

from .mano_head import MANOTransformerDecoderHead
from .mano_wrapper import MANO
from .vit import vit


class HAMER(nn.Module):
    """HaMeR hand mesh recovery model (inference only)."""

    def __init__(self, mano_model_path: str, mano_mean_params_path: str,
                 focal_length: float = 5000., image_size: int = 256):
        """
        Args:
            mano_model_path (str): Path to the MANO right hand model file (MANO_RIGHT.pkl).
            mano_mean_params_path (str): Path to the MANO mean parameters file (mano_mean_params.npz).
            focal_length (float): Focal length used during training, in crop pixels.
            image_size (int): Input crop size the model was trained with.
        """
        super().__init__()
        self.FOCAL_LENGTH = focal_length
        self.IMAGE_SIZE = image_size
        # ImageNet mean/std, laid out for (B, H, W, 3) inputs
        self.register_buffer('IMAGE_MEAN', torch.tensor([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3), persistent=False)
        self.register_buffer('IMAGE_STD', torch.tensor([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3), persistent=False)

        # Create ViT-H backbone feature extractor
        self.backbone = vit()

        # Create MANO transformer decoder head
        self.mano_head = MANOTransformerDecoderHead(mano_mean_params_path)

        # Instantiate the MANO right hand model layer
        self.mano = MANO(model_path=mano_model_path, create_body_pose=False)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Run a forward pass on a batch of hand crops.
        Args:
            x (torch.Tensor): RGB image patches of shape (B, IMAGE_SIZE, IMAGE_SIZE, 3),
                with values in [0, 255]. Left hand crops must be flipped horizontally
                beforehand (the model is right-hand only).
        Returns:
            dict: Dictionary containing:
                - 'pred_cam' (B, 3): Weak-perspective camera [s, tx, ty] in the crop frame.
                - 'global_orient' (B, 1, 3): Global orientation as axis-angle.
                - 'hand_pose' (B, 15, 3): MANO hand pose as axis-angle.
                - 'betas' (B, 10): MANO shape parameters.
                - 'pred_keypoints_3d' (B, 21, 3): 3D hand joints in the OpenPose ordering.
                - 'pred_vertices' (B, 778, 3): 3D MANO mesh vertices.
        """
        x = x / 255.0
        x = (x - self.IMAGE_MEAN.to(dtype=x.dtype)) / self.IMAGE_STD.to(dtype=x.dtype)
        x = x.permute(0, 3, 1, 2)
        batch_size = x.shape[0]

        # Compute conditioning features using the backbone. The ViT backbone takes a
        # 256x192 aspect ratio, so the square crop is trimmed at the sides.
        conditioning_feats = self.backbone(x[:, :, :, 32:-32])

        pred_mano_params, pred_cam = self.mano_head(conditioning_feats)

        # Compute model vertices and joints
        pred_mano_params['global_orient'] = pred_mano_params['global_orient'].reshape(batch_size, -1, 3, 3)
        pred_mano_params['hand_pose'] = pred_mano_params['hand_pose'].reshape(batch_size, -1, 3, 3)
        pred_mano_params['betas'] = pred_mano_params['betas'].reshape(batch_size, -1)
        mano_output = self.mano(**{k: v.float() for k, v in pred_mano_params.items()}, pose2rot=False)

        output = {
            'pred_cam': pred_cam,
            'global_orient': roma.rotmat_to_rotvec(pred_mano_params['global_orient']),
            'hand_pose': roma.rotmat_to_rotvec(pred_mano_params['hand_pose']),
            'betas': pred_mano_params['betas'],
            'pred_keypoints_3d': mano_output.joints.reshape(batch_size, -1, 3),
            'pred_vertices': mano_output.vertices.reshape(batch_size, -1, 3),
        }
        return output
