from typing import List, Union, Any

import torch
from torch import nn
import numpy as np
import h5py


class BaseFeatureExtractor(nn.Module):
    def __init__(self, device: str = "cpu", dtype=torch.float32, name: str = None):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.name = name

    def extract(
        self, data: Any, return_type: Union[str, str] = "numpy"
    ) -> Union[np.ndarray, torch.tensor]:
        raise NotADirectoryError

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.extract(*args, **kwds)

    def save_with_h5py(self, f: Union[h5py.File, str], *args, **kwds):
        raise NotImplementedError

    def forward(self, *args: Any, **kwds: Any) -> Any:
        return self.extract(*args, **kwds)
