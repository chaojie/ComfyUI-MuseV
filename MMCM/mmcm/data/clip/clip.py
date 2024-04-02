from __future__ import annotations
from copy import deepcopy

from typing import Iterable, List, Tuple, Dict, Hashable, Any, Union

import numpy as np

from ...utils.util import convert_class_attr_to_dict


from ..general.items import Items, Item
from .clipid import MatchedClipIds


import logging

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


__all__ = ["Clip", "ClipSeq"]


class Clip(Item):
    """媒体片段, 指转场点与转场点之间的部分"""

    def __init__(
        self,
        time_start: float,
        duration: float,
        clipid: int = None,
        media_type: str = None,
        mediaid: str = None,
        timepoint_type: str = None,
        text: str = None,
        stage: str = None,
        path: str = None,
        duration_num: int = None,
        similar_clipseq: MatchedClipIds = None,
        dynamic: float = None,
        **kwargs,
    ):
        """
        Args:
            time_start (float): 开始时间,秒为单位,对应该媒体文件的, 和media_map.json上的序号一一对应
            duration (_type_): 片段持续时间
            clipid (int, or [int]): 由media_map提供的片段序号, 和media_map.json上的序号一一对应
            media_type (str, optional): music, video,text, Defaults to None.
            mediaid (int): 多媒体id, 当clipid是列表时,表示该片段是个融合片段
            timepoint_type(int, ): 开始点的转场类型. Defaults to None.
            text(str, optional): 该片段的文本描述,音乐可以是歌词,视频可以是台词,甚至可以是弹幕. Defaults to None.
            stage(str, optional): 该片段在整个媒体文件中的结构位置,如音乐的intro、chrous、vesa,视频的片头、片尾、开始、高潮、转场等. Defaults to None.
            path (str, optional): 该媒体文件的路径,用于后续媒体读取、处理. Defaults to None.
            duration_num (_type_, optional): 片段持续帧数, Defaults to None.
            similar_clipseq ([Clip]], optional): 与该片段相似的片段，具体结构待定义. Defaults to None.
        """
        self.media_type = media_type
        self.mediaid = mediaid
        self.time_start = time_start
        self.duration = duration
        self.clipid = clipid
        self.path = path
        self.timepoint_type = timepoint_type
        self.text = text
        self.stage = stage
        self.duration_num = duration_num
        self.similar_clipseq = similar_clipseq
        self.dynamic = dynamic
        self.__dict__.update(**kwargs)

    def preprocess(self):
        pass

    def spread_parameters(self):
        pass

    @property
    def time_end(
        self,
    ) -> float:
        return self.time_start + self.duration

    def get_emb(self, key: str, idx: int) -> np.float:
        return self.emb.get_value(key, idx)


class ClipSeq(Items):
    """媒体片段序列"""

    def __init__(self, items: List[Clip] = None):
        super().__init__(items)
        self.clipseq = self.data

    def preprocess(self):
        pass

    def set_clip_value(self, k: Hashable, v: Any) -> None:
        """给序列中的每一个clip 赋值"""
        for i in range(len(self.clipseq)):
            self.clipseq[i].__setattr__(k, v)

    def __len__(
        self,
    ) -> int:
        return len(self.clipseq)

    @property
    def duration(
        self,
    ) -> float:
        """Clip.duration的和

        Returns:
            float: 序列总时长
        """
        if len(self.clipseq) == 0:
            return 0
        else:
            return sum([c.duration for c in self.clipseq])

    def __getitem__(self, i: Union[int, Iterable]) -> Union[Clip, ClipSeq]:
        """支持索引和切片操作，如果输入是整数则返回Clip，如果是切片，则返回ClipSeq

        Args:
            i (int or slice): 索引

        Raises:
            ValueError: 需要按照给的输入类型索引

        Returns:
            Clip or ClipSeq:
        """
        if "int" in str(type(i)):
            i = int(i)
        if isinstance(i, int):
            clip = self.clipseq[i]
            return clip
        elif isinstance(i, Iterable):
            clipseq = [self.__getitem__(x) for x in i]
            clipseq = ClipSeq(clipseq)
            return clipseq
        elif isinstance(i, slice):
            if i.step is None:
                step = 1
            else:
                step = i.step
            clipseq = [self.__getitem__(x) for x in range(i.start, i.stop, step)]
            clipseq = ClipSeq(clipseq)
            return clipseq
        else:
            raise ValueError(
                "unsupported input, should be int or slice, but given {}, type={}".format(
                    i, type(i)
                )
            )

    @property
    def mvp_clip(self):
        """读取实际的片段数据为moviepy格式

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

    @property
    def duration_seq_emb(
        self,
    ) -> np.array:
        emb = np.array([c.duration for c in self.clipseq])
        return emb

    @property
    def timestamp_seq_emb(self) -> np.array:
        emb = np.array([c.time_start for c in self.clipseq])
        return emb

    @property
    def rela_timestamp_seq_emb(self) -> np.array:
        duration_seq = [c.duration for c in self.clipseq]
        emb = np.cumsum(duration_seq) / self.duration
        return emb

    def get_emb(self, key: str, idx: int) -> np.float:
        clip_start_idx = self.clipseq[0].clipid
        clip_end_idx = self.clipseq[-1].clipid
        # TODO: 待修改为更通用的形式
        if idx is None:
            idx = range(clip_start_idx, clip_end_idx + 1)
        elif isinstance(idx, int):
            idx += clip_start_idx
        elif isinstance(idx, Iterable):
            idx = [x + clip_start_idx for x in idx]
        else:
            raise ValueError(
                f"idx only support None, int, Iterable, but given {idx},type is {type(idx)}"
            )
        return self.emb.get_value(key, idx=idx)
