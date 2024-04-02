import numpy as np

from .vision_object import Object


def cal_shot_size_by_face(frame_width: int, frame_height: int, obj: Object) -> str:
    """根据图像宽高和人脸框的大小判断人脸的景别

    Args:
        frame_width (int): 图像宽
        frame_height (int): 图像高
        obj (Object): 人脸检测框

    Returns:
        str: 根据人脸框信息计算的景别
    """
    obj_area = obj.area
    frame_area = frame_height * frame_width
    area_ratio = obj_area / frame_area
    width_ratio = obj.width / frame_width
    height_ratio = obj.height / frame_height
    if height_ratio >= 0.7 or width_ratio >= 0.8:
        shot_size = "ExtremeCloseUP"
    elif height_ratio >= 0.5 or width_ratio >= 0.7:
        shot_size = "CloseUp"
    elif height_ratio >= 0.2 or width_ratio > 0.4:
        shot_size = "MeiumShot"
    elif height_ratio >= 0.1 or width_ratio >= 0.1:
        shot_size = "FullShot"
    else:
        shot_size = "WideShot"
    return shot_size
