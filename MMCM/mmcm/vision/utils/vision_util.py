import math
from typing import Union, Tuple


from numpy import ndarray
import numpy as np


def cal_start_end_point(big: float, small: float, center: float = None):
    if center is None:
        center = big / 2
    if center < small / 2:
        center = small / 2
    if center > (big - small / 2):
        center = big - small / 2
    start = center - small / 2
    end = center + small / 2
    return (start, end)


def cal_small_bbox_coord_of_big_bbox(
    bigbbox_width,
    bigbbox_height,
    smallbbox_width,
    smallbbox_height,
    center_width: float = None,
    center_height: float = None,
    need_round2even=False,
):
    """只有宽高信息，按中心crop计算小矩形在大矩形的剪辑坐标

    Args:
        bigbbox_width (float): _description_
        bigbbox_height (float): _description_
        smallbbox_width (float): _description_
        smallbbox_height (float): _description_

    Returns:
        (float, float, float, float): (x1, y1, x2, y2) 在大矩形中的剪辑位置
    """
    x1, x2 = cal_start_end_point(bigbbox_width, smallbbox_width, center=center_width)
    y1, y2 = cal_start_end_point(bigbbox_height, smallbbox_height, center=center_height)
    # x1 = (bigbbox_width - smallbbox_width) / 2
    # y1 = (bigbbox_height - smallbbox_height) / 2
    # x2 = (bigbbox_width + smallbbox_width) / 2
    # y2 = (bigbbox_height + smallbbox_height) / 2
    if need_round2even:
        x1, y1, x2, y2 = round_up_coord_to_even(x1, y1, x2, y2)
    return (x1, y1, x2, y2)


def round_up_to_even(num):
    return math.floor(num / 2.0) * 2


def round_up_coord_to_even(x1, y1, x2, y2):
    x2 = x1 + round_up_to_even(x2 - x1)
    y2 = y1 + round_up_to_even(y2 - y1)
    return (x1, y1, x2, y2)


def cal_crop_coord(
    width,
    height,
    target_width_height_ratio,
    restricted_bbox=None,
    need_round2even=False,
):
    """
    (TODO): 当前只考虑crop，不考虑补全；
    (TODO): 当前只考虑视频尺寸比目标尺寸大

    Args:
        width (float): 原视频的宽
        height (float): 原视频的高
        target_width (float): 目标视频的宽
        target_height (float): 目标视频的高
        restricted_bbox ((float, float, float, float), optional): (x1, y1, x2, y2). Defaults to None.

    Returns:
        (float, float, float, float): (x1, y1, x2, y2) 在原视频中的剪辑位置
    """

    if restricted_bbox is None:
        target_width, target_height = cal_target_width_height_by_ratio(
            width=width,
            height=height,
            target_width_height_ratio=target_width_height_ratio,
        )
        crop_bbox = cal_small_bbox_coord_of_big_bbox(
            width, height, target_width, target_height
        )
    else:
        r_width = restricted_bbox[2] - restricted_bbox[0]
        r_height = restricted_bbox[3] - restricted_bbox[1]
        target_width, target_height = cal_target_width_height_by_ratio(
            width=r_width,
            height=r_height,
            target_width_height_ratio=target_width_height_ratio,
        )
        crop_bbox = cal_small_bbox_coord_of_big_bbox(
            r_width, r_height, target_width, target_height
        )
        crop_bbox = (
            crop_bbox[0] + restricted_bbox[0],
            crop_bbox[1] + restricted_bbox[1],
            crop_bbox[2] + restricted_bbox[0],
            crop_bbox[3] + restricted_bbox[1],
        )
    if need_round2even:
        crop_bbox = round_up_coord_to_even(*crop_bbox)
    return crop_bbox


def cal_target_width_height_by_ratio(
    width, height, target_width_height_ratio, need_round2even=False
):
    """针对原视频的宽、高和目标视频宽高比，计算合适的宽、高

    Args:
        width (float): 原视频的宽
        height (float): 原视频的高
        target_width_height_ratio (float): 目标视频宽高比

    Returns:
        target_width (float)): 目标宽
        target_height (float): 目标高
    """
    width_height_ratio = width / height
    if width_height_ratio >= target_width_height_ratio:
        target_width = height * target_width_height_ratio
        target_height = height
    else:
        target_width = width
        target_height = width / target_width_height_ratio
    if need_round2even:
        target_height = round_up_to_even(target_height)
        target_width = round_up_to_even(target_width)
    return target_width, target_height


def cal_target_width_height(
    target_width=None,
    target_height=None,
    target_width_height_ratio=None,
    need_even=True,
):
    if target_width and not target_height and target_width_height_ratio:
        target_height = int(target_width / target_width_height_ratio)
        if need_even:
            target_height = round_up_to_even(target_height)
    if target_height and not target_width and target_width_height_ratio:
        target_width = int(target_height * target_width_height_ratio)
        if need_even:
            target_width = round_up_to_even(target_width)
    if target_height and target_width:
        target_width_height_ratio = target_width / target_height
    return target_width, target_height, target_width_height_ratio


def find_outlier(grid: ndarray) -> ndarray:
    """find outlier coordinary out of grid

    Args:
        grid (ndarray): 2xHxW

    Returns:
        mask: ndarray, HxW, 1 for coordinary in grid, 0 for outlier
    """
    c, h, w = grid.shape
    mask = np.zeros((h, w))
    for i, j in zip(range(h), range(w)):
        x = int(grid[0, i, j])
        y = int(grid[1, i, j])
        if x < h and y < w:
            mask[x, y] = 1
    return mask
