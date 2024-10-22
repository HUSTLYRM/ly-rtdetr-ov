"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import torchvision

from ...core import register


__all__ = ['RTDETRPostProcessor']


def mod(a, b):
    return a - a // b * b


@register()
class RTDETRPostProcessor(nn.Module):
    __share__ = [
        'num_classes', 
        'use_focal_loss', 
        'num_top_queries', 
        'remap_mscoco_category',
        'remap_rm_category',
    ]
    
    def __init__(
        self,
        # num_classes=80,
        num_bar_classes=80,
        num_pat_classes=80,
        use_focal_loss=True,
        num_top_queries=300,
        remap_mscoco_category=False,
        
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        # self.num_classes = int(num_classes)
        self.num_bar_classes = int(num_bar_classes)
        self.num_pat_classes = int(num_pat_classes)
        self.remap_mscoco_category = remap_mscoco_category 
        self.deploy_mode = False 

    def extra_repr(self) -> str:
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'
    
    # def forward(self, outputs, orig_target_sizes):
    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        # logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        bar_logits, pat_logits, quad = outputs['pred_bar_logits'], outputs['pred_pat_logits'], outputs['pred_quads']
        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)        

        # bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        # bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)
        quads_pred = quad * orig_target_sizes.tile(1, 4).unsqueeze(1)

        if self.use_focal_loss:
            # scores = F.sigmoid(logits)
            # scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            bar_scores = F.sigmoid(bar_logits)
            bar_scores, bar_index = torch.topk(bar_scores.flatten(1), self.num_top_queries)
            bar_labels = mod(bar_index, self.num_bar_classes)
            index = index // self.num_bar_classes
            # TODO for older tensorrt
            # labels = index % self.num_classes
            bar_labels = mod(bar_index, self.num_bar_classes)
            bar_index = bar_index // self.num_bar_classes
            # boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))
            quads = quads_pred.gather(1, index.unsqueeze(-1).repeat(1, 1, quads_pred.shape[-1]))
            
        else:
            # scores = F.softmax(logits)[:, :, :-1]
            bar_scores = F.softmax(bar_logits)[:, :, :-1]
            pat_scores = F.softmax(pat_logits)[:, :, :-1]
            # scores, labels = scores.max(dim=-1)
            bar_scores, bar_labels = bar_scores.max(dim=-1)
            pat_scores, pat_labels = pat_scores.max(dim=-1)
            # if scores.shape[1] > self.num_top_queries:
            if bar_scores.shape[1] > self.num_top_queries and pat_scores.shape[1] > self.num_top_queries:
                # scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                bar_scores, bar_index = torch.topk(bar_scores, self.num_top_queries)
                pat_scores, pat_index = torch.topk(pat_scores, self.num_top_queries)
                # labels = torch.gather(labels, dim=1, index=index)
                bar_labels = torch.gather(bar_labels, 1, index)
                pat_labels = torch.gather(pat_labels, 1, index)
                # boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1]))
                # TODO how to define quads?
        
        # TODO for onnx export
        if self.deploy_mode:
            # return labels, boxes, scores
            return quads, bar_labels, bar_scores, pat_labels, pat_scores

        # TODO
        if self.remap_mscoco_category:
            from ...data.dataset import rm_label2category_v2
            bar_labels = torch.tensor([rm_label2category_v2['bar'][int(x.item())] for x in bar_labels.flatten()]) \
                .to(quads.device).reshape(bar_labels.shape)
            pat_labels = torch.tensor([rm_label2category_v2['pattern'][int(x.item())] for x in pat_labels.flatten()]) \
                .to(quads.device).reshape(pat_labels.shape)

        results = []
        # for lab, box, sco in zip(labels, boxes, scores):
        #     result = dict(labels=lab, boxes=box, scores=sco)
        #     results.append(result)
        for qua, bar_lab, bar_sco, pat_lab, pat_sco in zip(quads, bar_labels, bar_scores, pat_labels, pat_scores):
            results.append({
                'quads': quad,
                'bar_labels': bar_lab,
                'bar_scores': bar_sco,
                'pat_labels': pat_lab,
                'pat_scores': pat_sco,
            })
        return results
        
    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self 
