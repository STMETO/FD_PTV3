"""Minimal task heads used by the extracted PTv3 + S3DIS project.

This file intentionally keeps only the segmentation heads that are required by
the federated S3DIS training chain. The original repository also contains
classification and DINO-enhanced variants, but they are not needed here and
would pull in extra dependencies.
"""

import torch
import torch.nn as nn

from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from .builder import MODELS, build_model


@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    """Legacy segmentation wrapper.

    This class is kept for compatibility with upstream Pointcept configs. The
    minimal federated S3DIS pipeline uses DefaultSegmentorV2 below.
    """

    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        # Point Prompt Training models may attach a batch-level condition.
        if "condition" in input_dict.keys():
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.backbone(input_dict)
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    """Main semantic segmentation head used by the minimal federated project.

    PTv3 backbones return a Point object carrying sparse hierarchy metadata.
    During decoding we walk back through the stored pooling chain so the head
    always sees per-point features at the original point resolution.
    """

    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_dict, return_point=False):
        # Wrap the raw batch dictionary into Pointcept's Point structure so the
        # backbone can access coordinates, offsets and sparse metadata.
        point = Point(input_dict)
        point = self.backbone(point)

        # PTv3 returns a Point object after sparse encoding/decoding. When the
        # decoder stores pooling ancestry, we propagate features back to the
        # finest resolution before applying the segmentation head.
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point

        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            # Evaluators that need coordinates/features can read the recovered
            # Point object from the returned payload.
            return_dict["point"] = point
        if self.training:
            return_dict["loss"] = self.criteria(seg_logits, input_dict["segment"])
        elif "segment" in input_dict.keys():
            return_dict["loss"] = self.criteria(seg_logits, input_dict["segment"])
            return_dict["seg_logits"] = seg_logits
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict
