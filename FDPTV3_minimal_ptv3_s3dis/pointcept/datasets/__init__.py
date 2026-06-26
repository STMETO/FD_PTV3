"""Dataset registry surface for the minimal project.

The extracted project keeps only S3DIS-related dataset code and the generic
dataset utilities that Pointcept's loaders expect.
"""

from .defaults import DefaultDataset, ConcatDataset
from .builder import build_dataset
from .utils import point_collate_fn, collate_fn

from .s3dis import S3DISDataset
