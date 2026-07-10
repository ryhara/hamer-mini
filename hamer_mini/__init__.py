"""hamer-mini: a minimal, pip-installable inference library for HaMeR hand mesh recovery.

HaMeR: https://github.com/geopavlakos/hamer
Librarization inspired by WiLoR-mini: https://github.com/warmshao/WiLoR-mini
"""
from .pipelines.hamer_hand_pose3d_estimation_pipeline import HaMeRHandPose3dEstimationPipeline
from .utils.rotations import (
    FINGER_NAMES,
    FINGERTIP_KEYPOINTS,
    FINGERTIP_PARENT_JOINTS,
    MANO_PARENTS,
    MANO_TO_KEYPOINT,
    axis_angle_to_euler,
    axis_angle_to_matrix,
    cumulative_joint_rotations,
    fingertip_rotations,
    matrix_to_euler,
)

__version__ = "0.1.0"
__all__ = [
    "HaMeRHandPose3dEstimationPipeline",
    "axis_angle_to_euler",
    "axis_angle_to_matrix",
    "matrix_to_euler",
    "cumulative_joint_rotations",
    "fingertip_rotations",
    "MANO_PARENTS",
    "MANO_TO_KEYPOINT",
    "FINGERTIP_PARENT_JOINTS",
    "FINGERTIP_KEYPOINTS",
    "FINGER_NAMES",
    "__version__",
]
