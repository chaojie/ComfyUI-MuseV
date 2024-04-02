from typing import Callable

import numpy as np
import torch
from torch import nn
import cv2


def torch_wrap(img, flow, mode='bilinear', padding_mode='zeros', align_corners=True, outlier_func: Callable=None):
    """
    reffers:
        1. https://github.com/safwankdb/ReCoNet-PyTorch/blob/master/utilities.py
        2. https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
    warp an image/tensor (img) back to output images, according to the optical flow
    img: [B, C, H, W], torch.FloatTensor, [0, 1] or [0, 255]
    flow: [B, 2, H, W]
    return:
        output: BxCxHxW
    """
    B, C, H, W = img.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy), 1)
    grid = grid.to(img.device, dtype=img.dtype)
    # print(img.shape, grid.shape, flow.shape)
    vgrid = grid + flow
    if outlier_func is not None:
        from ..utils.torch_util import find_outlier
        mask = find_outlier(vgrid).to(img.device)
        mask = mask.unsqueeze(dim=1).repeat(1,C,1,1)
    # scale grid to [-1,1]
    vgrid[:,0,:,:] = 2.0 * vgrid[:,0,:,:] / max(W - 1, 1) - 1.0
    vgrid[:,1,:,:] = 2.0 * vgrid[:,1,:,:] / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(img, vgrid,  mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    if outlier_func is not None:
        outlier = outlier_func(output.shape).to(img.device)
        output = mask * output + (1 - mask) * outlier
    output = output.to(dtype=img.dtype)
    return output


def opencv_wrap(img:np.array, flow: np.array, outlier_func: Callable=None) -> np.array:
    """wrap image with flow to output image

    Args:
        img (np.array): source image, HxWx3, [0-255]
        flow (np.array): flow from source image to output image, HxWx2, [-int, int]

    Returns:
        np.array: output image, HxWx3, 
    """
    from ..utils.vision_util import find_outlier
    h, w, c = flow.shape
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    output = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    if outlier_func is not None:
        outlier = outlier_func(output.shape)
        mask = find_outlier(np.transpose(flow, (1, 2, 0)))
        mask = np.repeat(mask[:, :, np.newaxis], repeats=c, axis=2)
        output = mask * output + (1 - mask) * outlier
    return output
