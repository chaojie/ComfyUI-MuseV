import torch
import torch.nn as nn


class TVLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): b * c * h * w

        Returns:
            torch.Tensor: _description_
        """
        b, c, h, w = x.shape
        count_h = b * (h - 1) * w * c
        count_w = b * h * (w - 1) * c
        h_tv = (torch.pow((x[:,:,1:,:]-x[:,:,:-1,:]),2) / count_h).sum()
        w_tv = (torch.pow((x[:,:,:,1:]-x[:,:,:,:-1]),2) / count_w).sum()
        loss = 2 * (h_tv  + w_tv)
        return loss
