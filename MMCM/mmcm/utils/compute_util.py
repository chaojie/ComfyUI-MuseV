from typing import List
import numpy as np


def weighted_sum(weights: List[float], datas: List[np.array]) -> np.array:
    """对矩阵列表按照权重列表加权求和

    Args:
        weights (list): 权重列表
        datas (list): 矩阵列表

    Returns:
        np.array: 加权求和后的矩阵
    """
    res = np.zeros(datas[0].shape)
    n_data = len(datas)
    for i in range(n_data):
        res += datas[i] * weights[i] / n_data
    return res
