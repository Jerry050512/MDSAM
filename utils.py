import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import OrderedDict
from matplotlib.backends.backend_pdf import PdfPages

def reshapePos(pos_embed, img_size):
    token_size = int(img_size // 16)
    if pos_embed.shape[1] != token_size:
        # resize pos embedding
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
        pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
    return pos_embed

def reshapeRel(k, rel_pos_params, img_size):
    if not ('2' in k or '5' in  k or '8' in k or '11' in k):
        return rel_pos_params
    
    token_size = int(img_size // 16)
    h, w = rel_pos_params.shape
    rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
    rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
    return rel_pos_params[0, 0, ...]

def load_sam(net, sam_pre, img_size):
    state_dict = torch.load(sam_pre)
    dict_filtered = OrderedDict()

    for k, v in state_dict.items():
        if 'pe_layer' in k:
            dict_filtered[k[15:]] = v
            continue
        if 'image_encoder' in k:
            if 'pos_embed' in k:
                dict_filtered[k] = reshapePos(v, img_size)
            elif 'rel_pos' in k:
                dict_filtered[k] = reshapeRel(k, v, img_size)
            else:
                dict_filtered[k] = v
            continue
        if 'mask_decoder' in k:
            if any(x in k for x in ['transformer', 'iou_token', 'output_upscaling']):
                dict_filtered[k] = v
            else:
                continue
    load_info = net.load_state_dict(dict_filtered, strict=False)

    return load_info

def LossFunc(pred, mask):
    mask=mask.float()
    bce = F.binary_cross_entropy(pred, mask, reduce=None)

    inter = ((pred * mask)).sum(dim=(2, 3))
    union = ((pred + mask)).sum(dim=(2, 3))
    aiou = 1 - (inter + 1) / (union - inter + 1)

    mae = F.l1_loss(pred, mask, reduce=None)

    return (bce + aiou + mae).mean()

