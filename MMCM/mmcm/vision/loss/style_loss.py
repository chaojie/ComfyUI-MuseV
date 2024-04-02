
from typing import List, Union

import torch
import torch.nn as nn

from .multi_layer_loss import MultiLayerLoss


class StyleLoss(MultiLayerLoss):
    def __init__(self, loss_fn: nn.Module=nn.MSELoss, weights: List[float] = None) -> None:
        super().__init__(loss_fn, weights)

    def cal_single_layer_loss(self, x, y):
        b, c, h, w = x.shape
        loss = self.loss_fn(x, y) / (c * h * w) ** 2 / 4
        return loss
    
    def _get_feature(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = x.view(b, c, h * w)
        gram = x.mul(x, x.transpose(1, 2))
        return gram


