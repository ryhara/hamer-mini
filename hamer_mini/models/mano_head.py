# Original code: https://github.com/geopavlakos/hamer/blob/main/hamer/models/heads/mano_head.py
# The yacs config is removed; hyperparameters are hardcoded to the values of the
# released HaMeR checkpoint (_DATA/hamer_ckpts/model_config.yaml).
import einops
import numpy as np
import torch
import torch.nn as nn

from ..utils.geometry import rot6d_to_rotmat
from .pose_transformer import TransformerDecoder


class MANOTransformerDecoderHead(nn.Module):
    """Cross-attention based MANO transformer decoder head."""

    def __init__(self, mano_mean_params_path: str, num_hand_joints: int = 15, ief_iters: int = 1):
        """
        Args:
            mano_mean_params_path (str): Path to the MANO mean parameters file (mano_mean_params.npz).
            num_hand_joints (int): Number of MANO hand joints (excluding the global orientation).
            ief_iters (int): Number of iterative error feedback refinement iterations.
        """
        super().__init__()
        self.joint_rep_dim = 6  # 6D rotation representation
        npose = self.joint_rep_dim * (num_hand_joints + 1)
        self.npose = npose
        self.num_hand_joints = num_hand_joints
        self.ief_iters = ief_iters
        self.transformer = TransformerDecoder(
            num_tokens=1,
            token_dim=1,
            dim=1024,
            depth=6,
            heads=8,
            mlp_dim=1024,
            dim_head=64,
            dropout=0.0,
            emb_dropout=0.0,
            norm='layer',
            context_dim=1280,
        )
        dim = 1024
        self.decpose = nn.Linear(dim, npose)
        self.decshape = nn.Linear(dim, 10)
        self.deccam = nn.Linear(dim, 3)

        mean_params = np.load(mano_mean_params_path)
        init_hand_pose = torch.from_numpy(mean_params['pose'].astype(np.float32)).unsqueeze(0)
        init_betas = torch.from_numpy(mean_params['shape'].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam'].astype(np.float32)).unsqueeze(0)
        self.register_buffer('init_hand_pose', init_hand_pose)
        self.register_buffer('init_betas', init_betas)
        self.register_buffer('init_cam', init_cam)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Backbone feature map of shape (B, C, H, W).
        Returns:
            pred_mano_params (dict): Dictionary with 'global_orient' (B, 1, 3, 3),
                'hand_pose' (B, 15, 3, 3) and 'betas' (B, 10).
            pred_cam (torch.Tensor): Weak-perspective camera of shape (B, 3).
        """
        batch_size = x.shape[0]
        # vit pretrained backbone is channel-first. Change to token-first
        x = einops.rearrange(x, 'b c h w -> b (h w) c')

        pred_hand_pose = self.init_hand_pose.expand(batch_size, -1)
        pred_betas = self.init_betas.expand(batch_size, -1)
        pred_cam = self.init_cam.expand(batch_size, -1)
        for _ in range(self.ief_iters):
            # Input token to transformer is zero token
            token = torch.zeros(batch_size, 1, 1, device=x.device, dtype=x.dtype)

            # Pass through transformer
            token_out = self.transformer(token, context=x)
            token_out = token_out.squeeze(1)  # (B, C)

            # Readout from token_out
            pred_hand_pose = self.decpose(token_out) + pred_hand_pose
            pred_betas = self.decshape(token_out) + pred_betas
            pred_cam = self.deccam(token_out) + pred_cam

        # Convert the 6D rotation representation to rotation matrices
        pred_hand_pose = rot6d_to_rotmat(pred_hand_pose).view(batch_size, self.num_hand_joints + 1, 3, 3)

        pred_mano_params = {'global_orient': pred_hand_pose[:, [0]],
                            'hand_pose': pred_hand_pose[:, 1:],
                            'betas': pred_betas}
        return pred_mano_params, pred_cam
