from typing import Tuple
import os, random

import numpy as np
import torch


def set_all_seed(seed: int) -> Tuple[torch.Generator, torch.Generator]:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    cpu_generator = torch.Generator("cpu").manual_seed(seed)
    gpu_generator = torch.Generator("cuda").manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    return cpu_generator, gpu_generator
