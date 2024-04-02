import torch


def find_outlier(grid: torch.Tensor) -> torch.Tensor:
    """find outlier coordinary out of grid

    Args:
        grid (torch.Tensor): Bx2xHxW

    Returns:
        mask: ndarray, BxHxW, 1 for coordinary in grid, 0 for outlier
    """
    b, _, h, w = grid.shape
    mask = torch.ones((b, h, w))
    outlier_x_coordinary = (grid[:,0,:,:] >= w).nonzero(as_tuple=True)
    outlier_y_coordinary = (grid[:,1,:,:] >= h).nonzero(as_tuple=True)
    mask[outlier_x_coordinary] = 0
    mask[outlier_y_coordinary] = 0
    return mask
