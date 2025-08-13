import torch
import torch.nn.functional as F
from collections import OrderedDict

def reshapePos(pos_embed, img_size):
    """
    Reshape the positional embedding for a different image size.
    """
    token_size = int(img_size // 16)
    if pos_embed.shape[1] != token_size:
        # resize pos embedding
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
        pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
    return pos_embed

def reshapeRel(k, rel_pos_params, img_size):
    """
    Reshape the relative positional parameters for a different image size.
    """
    if not ('2' in k or '5' in  k or '8' in k or '11' in k):
        return rel_pos_params
    
    token_size = int(img_size // 16)
    h, w = rel_pos_params.shape
    rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
    rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
    return rel_pos_params[0, 0, ...]

def load_sam(net, sam_pre, img_size):
    """
    Load weights from a pretrained SAM model into the DefectSAM model.
    This function handles reshaping of positional embeddings.
    """
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
