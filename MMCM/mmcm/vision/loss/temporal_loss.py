from typing import List, Union

import torch
import torch.nn as nn
from .multi_layer_loss import MultiLayerLoss

from ..flow.util import torch_wrap


class FlowShortTermLoss(nn.Module):
    def __init__(self, loss_fn: nn.Module = nn.MSELoss) -> None:
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, output, wrap):
        b, c, h, w = output.shape
        loss = self.loss_fn(output, wrap)
        return loss


class FlowLongTermLoss(MultiLayerLoss):
    def __init__(self, loss_fn: nn.Module, weights: List[float] = None) -> None:
        super().__init__(loss_fn, weights)

    def forward(
        self, outputs: List[torch.tensor], wraps: List[torch.tensor]
    ) -> torch.Tensor:
        """_summary_

        Args:
            output (torch.Tensor): b * c * h * w

        Returns:
            torch.Tensor: _description_
        """
        assert len(outputs) == len(
            wraps
        ), f"length should be x({len(outputs)}) == target({len(wraps)})"
        if self.weights is not None:
            assert len(outputs) == len(
                self.weights
            ), f"weights should be None or length of x({len(outputs)}) must be equal to target({len(self.weights)})"

        total_loss = 0
        for i in len(outputs):
            b, c, h, w = output.shape
            output = outputs[i]
            wrap = wraps[i]
            loss = self.loss_fn(output, wrap)  # mseloss reduce=mean
            if self.weights is not None:
                loss *= self.weights[i]
            total_loss += loss
        return total_loss
