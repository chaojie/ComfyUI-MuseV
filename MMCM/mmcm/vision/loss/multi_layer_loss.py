
from typing import List, Union

import torch
import torch.nn as nn


class MultiLayerLoss(nn.Module):
    def __init__(self, loss_fn: nn.Module, weights: List[float]=None) -> None:
        super().__init__()
        self.weights = weights
        self.loss_fn = loss_fn

    def forward(self, output: Union[torch.Tensor, List[torch.tensor]], target: Union[torch.Tensor, List[torch.tensor]]) -> torch.Tensor:
        """_summary_

        Args:
            output (torch.Tensor): b * c * h * w

        Returns:
            torch.Tensor: _description_
        """
        if not isinstance(output, List):
            output = [output]
        if not isinstance(target, list):
            target = [target]
        assert len(output) == len(target), f"length of x({len(output)}) must be equal to target({len(target)})"
        if self.weights is not None:
            assert len(output) == len(self.weights), f"weights should be None or length of x({len(output)}) must be equal to weights({len(self.weights)})"

        total_loss = 0
        for i in range(len(output)):
            x = output[i]
            y = target[i]
            x = self._get_feature(x)
            y = self._get_feature(y)
            loss = self.loss_fn(x, y)
            if self.weights is not None:
                loss *= self.weights[i]
            total_loss += loss
        return total_loss

    def cal_single_layer_loss(self, x, y):
        raise NotImplementedError

    def _get_feature(self, x):
        raise NotImplementedError