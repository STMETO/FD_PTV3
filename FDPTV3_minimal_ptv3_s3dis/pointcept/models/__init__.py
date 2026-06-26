"""Model registry surface kept by the minimal project.

Only the PTv3 backbone family, prompt-driven normalization helper and the two
segmentation wrappers are intentionally re-exported here.
"""

from .builder import build_model
from .default import DefaultSegmentor, DefaultSegmentorV2
from .modules import PointModule, PointModel
from .point_transformer_v3 import *
from .point_prompt_training import *
