"""Loss registry surface for the minimal semantic-segmentation project."""

from .builder import build_criteria, LOSSES

from .misc import CrossEntropyLoss
from .lovasz import LovaszLoss
