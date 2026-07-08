"""hamer-mini: a minimal, pip-installable inference library for HaMeR hand mesh recovery.

HaMeR: https://github.com/geopavlakos/hamer
Librarization inspired by WiLoR-mini: https://github.com/warmshao/WiLoR-mini
"""
from .pipelines.hamer_hand_pose3d_estimation_pipeline import HaMeRHandPose3dEstimationPipeline

__version__ = "0.1.0"
__all__ = ["HaMeRHandPose3dEstimationPipeline", "__version__"]
