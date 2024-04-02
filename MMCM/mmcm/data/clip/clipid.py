from __future__ import annotations

from typing import Union, List

__all__ = [
    "ClipIds",
    "ClipIdsSeq",
    "MatchedClipIds",
    "MatchedClipIdsSeq",
]


class ClipIds(object):
    def __init__(
        self,
        clipids: Union[int, List[int]],
    ) -> None:
        """ClipSeq 中的 Clip序号，主要用于多个 Clip 融合后的 Clip, 使用场景如
        1. 一个 MusicClip 可以匹配到多个 VideoClip，VideoClip 的索引便可以使用 ClipIds 定义。

        Args:
            clipids (list or int): ClipSeq 中的序号
        """
        self.clipids = clipids if isinstance(clipids, list) else [clipids]


class ClipIdsSeq(object):
    def __init__(self, clipids_seq: List[ClipIds]) -> None:
        """多个 ClipIds，使用场景可以是
        1. 将MediaClipSeq 进行重组，拆分重组成更粗粒度的ClipSeq；

        Args:
            clipids_seq (list): 组合后的 ClipIds 列表
        """
        self.clipids_seq = (
            clipids_seq if isinstance(clipids_seq, ClipIds) else [clipids_seq]
        )


# TODO: metric后续可能是字典
class MatchedClipIds(object):
    def __init__(
        self, id1: ClipIds, id2: ClipIds, metric: float = None, **kwargs
    ) -> None:
        """两种模态数据的片段匹配对，使用场景 可以是
        1. 音乐片段和视频片段 之间的匹配关系，

        Args:
            id1 (ClipIds): 第一种模态的片段
            id2 (ClipIds): 第二种模态的片段
            metric (float): 匹配度量距离
        """
        self.id1 = id1 if isinstance(id1, ClipIds) else ClipIds(id1)
        self.id2 = id2 if isinstance(id2, ClipIds) else ClipIds(id2)
        self.metric = metric
        self.__dict__.update(**kwargs)


class MatchedClipIdsSeq(object):
    def __init__(self, seq: List[MatchedClipIds], metric: float = None, **kwargs) -> None:
        """两种模态数据的序列匹配对，使用场景可以是
        1. 音乐片段序列和视频片段序列 之间的匹配，每一个元素都是MatchedClipIds:

        Args:
            seq (list): 两种模态数据的序列匹配对列表
            metric (float): 匹配度量距离
        """
        self.seq = seq
        self.metric = metric
        self.__dict__.update(**kwargs)
