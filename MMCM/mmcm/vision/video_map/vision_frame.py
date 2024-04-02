from __future__ import annotations

from typing import Any, Dict, List
import numpy as np

from ...data import Item, Items
from ...utils.util import convert_class_attr_to_dict

from .vision_object import Objects
from .shot_size import cal_shot_size_by_face


# 结构体定义 VideoMashup/videomashup/data_structure/vision_data_structure.py Frame
class Frame(Item):
    def __init__(
        self,
        frame_idx: int,
        objs: Objects = None,
        scene: str = None,
        caption: str = None,
        shot_size: str = None,
        shot_composition: str = None,
        camera_angle: str = None,
        field_depth: str = None,
        content_width=None,
        content_height=None,
        **kwargs,
    ) -> None:
        """_summary_

        Args:
            frame_idx (int): 帧序号
            objs (Objects, optional): 检测到的物体. Defaults to None.
            scene (str, optional): 场景，天空、机场等. Defaults to None.
            caption (str, optional): 文本描述. Defaults to None.
            shot_size (str, optional): 景别. Defaults to None.
            shot_composition (str, optional): 构图. Defaults to None.
            camera_angle (str, optional): 相机角度. Defaults to None.
            field_depth (str, optional): 景深. Defaults to None.
        """
        self.frame_idx = frame_idx
        self.objs = objs if isinstance(objs, Objects) else Objects(objs)
        self.scene = scene
        self.caption = caption
        self.shot_size = shot_size
        self.shot_composition = shot_composition
        self.camera_angle = camera_angle
        self.field_depth = field_depth
        self.content_height = content_height
        self.content_width = content_width
        self.__dict__.update(**kwargs)
        self.preprocess()

    def preprocess(self):
        if (
            self.shot_size is None
            and self.content_height is not None
            and self.content_width is not None
        ):
            self.shot_size = self.cal_shot_size()

    def cal_shot_size(
        self,
    ):
        """计算当前帧的景别，目前使用人脸信息计算

        Returns:
            str: 景别，参考 VideoMashup/videomashup/data_structure/vision_data_structure.py
        """
        if len(self.objs.objs) > 0:
            obj = self.objs.get_max_bbox_obj()
            shot_size = cal_shot_size_by_face(
                frame_width=self.content_width,
                frame_height=self.content_height,
                obj=obj,
            )
        else:
            shot_size = "ExtremeWideShot"
        return shot_size

    @property
    def timestamp(self):
        timestamp = self.frame_idx / self.fps
        return timestamp

    def to_dct(self, target_keys: List[str] = None, ignored_keys: List[str] = None):
        dct = super().to_dct(target_keys, ignored_keys=["objs"])
        dct["objs"] = self.objs.to_dct()
        return dct


def get_width_center_by_topkrole(
    objs: list,
    coord_offset=None,
) -> float:
    """通过视频镜头中的人物目标信息 计算适合剪辑的横轴中心点

    Args:
        objs (list): 目标信息
        coord_offset (list, optional): 原视频的坐标和检测目标的坐标信息可能存在偏移，如有可使用该偏移矫正. Defaults to None.

    Returns:
        float: 横轴中心点
    """
    if coord_offset is None:
        coord_offset = [0, 0]
    min_roleid = str(min([int(x) for x in objs.keys()]))
    target_role = objs[min_roleid]
    bbox = [target_role["bbox"][x][0] for x in sorted(target_role["bbox"].keys())]
    target_idx = int(len(bbox) // 2)
    target_bbox = bbox[target_idx]
    target_bbox = [
        target_bbox[0] - coord_offset[0],
        target_bbox[1] - coord_offset[1],
        target_bbox[2] - coord_offset[0],
        target_bbox[3] - coord_offset[1],
    ]
    target_center_x = (target_bbox[0] + target_bbox[2]) / 2
    return target_center_x


def get_time_center_by_topkrole(
    objs: list,
) -> float:
    """计算主要目标人物的中心时间戳，适用于从原片段裁剪时序上的子片段，替代默认中间向两边

    Args:
        objs (list): 有时间戳信息的目标人物列表

    Returns:
        float: 中心时间戳
    """
    min_roleid = str(min([int(x) for x in objs.keys()]))
    target_role = objs[min_roleid]
    frame_idxs = [int(x) for x in target_role["bbox"].keys()]
    frame_idx = np.mean(frame_idxs)
    return frame_idx


class FrameSeq(Items):
    def __init__(self, frameseq: Any = None, **kwargs):
        super().__init__(frameseq)
        self.frameseq = self.data
        self.__dict__.update(**kwargs)

    @classmethod
    def from_data(
        cls, datas: List[Frame], frame_kwargs: Dict = None, **kwargs
    ) -> FrameSeq:
        if frame_kwargs is None:
            frame_kwargs = {}
        frameseq = [Frame(data, **frame_kwargs) for data in datas]
        return FrameSeq(frameseq=frameseq, **kwargs)
