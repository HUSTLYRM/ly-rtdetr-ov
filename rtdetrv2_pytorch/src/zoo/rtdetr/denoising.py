"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 

from .utils import inverse_sigmoid
# from .box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh


def get_contrastive_denoising_training_group(targets,
                                            #  num_classes,
                                             num_bar_classes,
                                             num_pat_classes,
                                             num_queries,
                                            #  class_embed,
                                             bar_class_embed,
                                             pat_class_embed,
                                             num_denoising=100,
                                            #  label_noise_ratio=0.5,
                                             bar_noise_ratio = 0.5,
                                             pat_noise_ratio = 0.5,
                                            #  box_noise_scale=1.0,):
                                             quad_noise_scale = 1.):
    """cnd"""
    if num_denoising <= 0:
        return None, None, None, None

    num_gts = [len(t['bars']) for t in targets]
    device = targets[0]['bars'].device
    
    max_gt_num = max(num_gts)
    if max_gt_num == 0:
        return None, None, None, None

    num_group = num_denoising // max_gt_num
    num_group = 1 if num_group == 0 else num_group
    # pad gt to max_num of a batch
    bs = len(num_gts)

    # input_query_class = torch.full([bs, max_gt_num], num_classes, dtype=torch.int32, device=device)
    input_query_bar = torch.full([bs, max_gt_num], num_bar_classes, dtype=torch.int32, device=device)
    input_query_pat = torch.full([bs, max_gt_num], num_pat_classes, dtype=torch.int32, device=device)
    # input_query_bbox = torch.zeros([bs, max_gt_num, 4], device=device)
    input_query_quad = torch.zeros([bs, max_gt_num, 8], device=device)
    pad_gt_mask = torch.zeros([bs, max_gt_num], dtype=torch.bool, device=device)

    for i in range(bs):
        num_gt = num_gts[i]
        if num_gt > 0:
            # input_query_class[i, :num_gt] = targets[i]['labels']
            input_query_bar[i, :num_gt] = targets[i]['bars']
            input_query_pat[i, :num_gt] = targets[i]['patterns']
            # input_query_bbox[i, :num_gt] = targets[i]['boxes']
            input_query_quad[i, :num_gt] = targets[i]['quads']
            pad_gt_mask[i, :num_gt] = 1

    # each group has positive and negative queries.
    # input_query_class = input_query_class.tile([1, 2 * num_group])
    input_query_bar = input_query_bar.tile([1, 2 * num_group])
    input_query_pat = input_query_pat.tile([1, 2 * num_group])
    # input_query_bbox = input_query_bbox.tile([1, 2 * num_group, 1])
    input_query_quad = input_query_quad.tile([1, 2 * num_group, 1])
    pad_gt_mask = pad_gt_mask.tile([1, 2 * num_group])

    # positive and negative mask
    negative_gt_mask = torch.zeros([bs, max_gt_num * 2, 1], device=device)
    negative_gt_mask[:, max_gt_num:] = 1
    negative_gt_mask = negative_gt_mask.tile([1, num_group, 1])
    positive_gt_mask = 1 - negative_gt_mask

    # contrastive denoising training positive index
    positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask
    dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]
    dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in num_gts])

    # total denoising queries
    num_denoising = int(max_gt_num * 2 * num_group)

    if bar_noise_ratio > 0:
        # mask = torch.rand_like(input_query_class, dtype=torch.float) < (label_noise_ratio * 0.5)
        mask_bar = torch.rand_like(input_query_bar, dtype=torch.float32) < (bar_noise_ratio * 0.5)
        # randomly put a new one here
        # new_label = torch.randint_like(mask, 0, num_classes, dtype=input_query_class.dtype)
        new_bar = torch.randint_like(mask_bar, 0, num_bar_classes, dtype=input_query_bar.dtype)
        # input_query_class = torch.where(mask & pad_gt_mask, new_label, input_query_class)
        input_query_bar = torch.where(mask_bar & pad_gt_mask, new_bar, input_query_bar)

    if pat_noise_ratio > 0:
        mask_pat = torch.rand_like(input_query_pat, dtype=torch.float32) < (pat_noise_ratio * 0.5)
        new_pat = torch.randint_like(mask_pat, 0, num_pat_classes, dtype=input_query_pat.dtype)
        input_query_pat = torch.where(mask_pat & pad_gt_mask, new_pat, input_query_pat)

    # if box_noise_scale > 0:
    #     known_bbox = box_cxcywh_to_xyxy(input_query_bbox)
    #     diff = torch.tile(input_query_bbox[..., 2:] * 0.5, [1, 1, 2]) * box_noise_scale
    #     rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0
    #     rand_part = torch.rand_like(input_query_bbox)
    #     rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (1 - negative_gt_mask)
    #     known_bbox += (rand_sign * rand_part * diff)
    #     known_bbox = torch.clip(known_bbox, min=0.0, max=1.0)
    #     input_query_bbox = box_xyxy_to_cxcywh(known_bbox)
    #     input_query_bbox_unact = inverse_sigmoid(input_query_bbox)

    if quad_noise_scale > 0:
        known_quad = input_query_quad
        diff = torch.tile(input_query_quad[..., [2, 3, 6, 7]], [1, 1, 2]) * quad_noise_scale * 0.5 ## TODO ?
        rand_sign = torch.randint_like(input_query_quad, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(input_query_quad)
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (1 - negative_gt_mask)
        known_quad += (rand_sign * rand_part * diff)
        known_quad += torch.clamp(known_quad, min=0, max=1)
        input_query_quad = known_quad
        input_query_quad_unact = inverse_sigmoid(input_query_quad)

    # input_query_logits = class_embed(input_query_class)
    input_query_bar_logits = bar_class_embed(input_query_bar)
    input_query_pat_logits = pat_class_embed(input_query_pat)

    tgt_size = num_denoising + num_queries
    attn_mask = torch.full([tgt_size, tgt_size], False, dtype=torch.bool, device=device)
    # match query cannot see the reconstruction
    attn_mask[num_denoising:, :num_denoising] = True
    
    # reconstruct cannot see each other
    for i in range(num_group):
        if i == 0:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1): num_denoising] = True
        if i == num_group - 1:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), :max_gt_num * i * 2] = True
        else:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1): num_denoising] = True
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), :max_gt_num * 2 * i] = True
        
    dn_meta = {
        "dn_positive_idx": dn_positive_idx,
        "dn_num_group": num_group,
        "dn_num_split": [num_denoising, num_queries]
    }

    # print(input_query_class.shape) # torch.Size([4, 196, 256])
    # print(input_query_bbox.shape) # torch.Size([4, 196, 4])
    # print(attn_mask.shape) # torch.Size([496, 496])
    
    # return input_query_logits, input_query_bbox_unact, attn_mask, dn_meta
    return input_query_bar_logits, input_query_pat_logits, input_query_quad_unact, attn_mask, dn_meta
