
from typing import List, Union, Callable

import torch
import torch.nn as nn

from .multi_layer_loss import MultiLayerLoss


class ContentLoss(nn.Module):
    def __init__(self, model: nn.Module, loss_fn: nn.Module=nn.MSELoss, weights: List[float] = None, transform: Callable=None,) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.weights = weights
        self.transform = transform

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            output (torch.Tensor): b * c * h * w

        Returns:
            torch.Tensor: _description_
        """

        if self.transform is not None:
            output = self.transform(output)
            target = self.transform(target)
        output_feature = self.model(output)
        target_feature = self.model(target)
        assert len(output_feature) == len(target_feature)
        keys = sorted(output_feature.keys())
        total_loss = 0
        for i, k in enumerate(keys):
            loss = self.loss_fn(output_feature[k], target_feature[k])
            print(i, k, loss)
            if self.weights is not None:
                loss *= self.weights[i]
            total_loss += loss
        return total_loss

