"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Modules to compute the matching cost and solve the corresponding LSAP.

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""
"""
MIT License
Copyright (c) 2024 AmadFat
https://github.com/HUSTLYRM/ly-rtdetr-ov
"""


import torch
import torch.nn as nn
import torch.nn.functional as F 

from scipy.optimize import linear_sum_assignment
from typing import Dict 

from .box_ops import generalized_box_iou, quad_to_bbox

from ...core import register


@register()
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    __share__ = ['use_focal_loss', ]

    def __init__(self, weight_dict, use_focal_loss=False, alpha=0.25, gamma=2.0):
        """Creates the matcher

        Params:
            cost_bar: This is the relative weight of the bar classification error in the matching cost
            cost_pattern: This is the relative weight of the pattern classification error in the matching cost
            cost_quad: This is the relative weight of the L1 error of the quad coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_bar = weight_dict['cost_bar']
        self.cost_pat = weight_dict['cost_pat']
        self.cost_quad = weight_dict['cost_quad']
        self.cost_giou = weight_dict['cost_giou']

        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

        assert all(_ for _ in [self.cost_bar, self.cost_pat, self.cost_quad, self.cost_giou]), "all 0 costs"

    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_bar_logits": Tensor of dim [batch_size, num_queries, num_bar_classes] with the bar classification logits
                 "pred_pat_logits": Tensor of dim [batch_size, num_queries, num_pat_classes] with the pattern classification logits
                 "pred_quads": Tensor of dim [batch_size, num_queries, 8] with the predicted quad coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "bars": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                        objects in the target) containing the bar class labels
                 "patterns": Tensor of dim [num_target_boxes] containing the pattern class labels
                 "quads": Tensor of dim [num_target_boxes, 8] containing the target quad coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_bar_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        if self.use_focal_loss:
            # out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))
            out_bar_prob = F.sigmoid(outputs["pred_bar_logits"].flatten(0, 1))
            out_pat_prob = F.sigmoid(outputs["pred_pat_logits"].flatten(0, 1))
        else:
            # out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
            out_bar_prob = outputs["pred_bar_logits"].flatten(0, 1).softmax(-1)
            out_pat_prob = outputs["pred_pat_logits"].flatten(0, 1).softmax(-1)

        # out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        out_quad = outputs["pred_quads"].flatten(0, 1)  # [batch_size * num_queries, 8]

        # Also concat the target bars, target patterns and quads
        tgt_bar = torch.cat([v["bars"] for v in targets])
        tgt_pat = torch.cat([v["patterns"] for v in targets])
        tgt_quad = torch.cat([v["quads"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        if self.use_focal_loss:
            # out_prob = out_prob[:, tgt_ids]
            # neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * (-(1 - out_prob + 1e-8).log())
            # pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
            # cost_class = pos_cost_class - neg_cost_class        
            out_bar_prob = out_bar_prob[:, tgt_bar]
            out_pat_prob = out_pat_prob[:, tgt_pat]
            neg_cost_bar = (1 - self.alpha) * (out_bar_prob ** self.gamma) * (-(1 - out_bar_prob + 1e-8).log())
            neg_cost_pat = (1 - self.alpha) * (out_pat_prob ** self.gamma) * (-(1 - out_pat_prob + 1e-8).log())
            pos_cost_bar = self.alpha * ((1 - out_bar_prob) ** self.gamma) * (-(out_bar_prob + 1e-8).log())
            pos_cost_pat = self.alpha * ((1 - out_pat_prob) ** self.gamma) * (-(out_pat_prob + 1e-8).log())
            cost_bar = pos_cost_bar - neg_cost_bar
            cost_pat = pos_cost_pat - neg_cost_pat
        else:
            # cost_class = -out_prob[:, tgt_ids]
            cost_bar = -out_bar_prob[:, tgt_bar]
            cost_pat = -out_pat_prob[:, tgt_pat]

        # Compute the L1 cost between boxes
        # cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_quad = torch.cdist(out_quad, tgt_quad, p=1)

        # Compute the giou cost betwen boxes
        # cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        cost_giou = -generalized_box_iou(quad_to_bbox(tgt_quad), quad_to_bbox(tgt_quad))
        
        # Final cost matrix
        # C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        # C = C.view(bs, num_queries, -1).cpu()
        C = self.cost_quad * cost_quad + self.cost_bar * cost_bar + self.cost_pat * cost_pat + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["quads"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

        return {'indices': indices}
        