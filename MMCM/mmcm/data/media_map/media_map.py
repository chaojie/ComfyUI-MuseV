from __future__ import annotations
import bisect

import logging
from copy import deepcopy

from functools import partial
from typing import Any, Callable, Iterable, List, Union, Tuple, Dict

import numpy as np
from ..clip.clip_process import get_subseq_by_time
from ..clip.clip_stat import stat_clipseq_duration
from ..clip import Clip, ClipSeq, ClipIds, MatchedClipIds, MatchedClipIdsSeq
from .media_map_process import get_sub_mediamap_by_time
from ..emb import MediaMapEmb, H5pyMediaMapEmb
from ..general.items import Item, Items
from ...utils.data_util import pick_subdct
from ...utils.util import convert_class_attr_to_dict, load_dct_from_file

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


__all__ = ["MetaInfo", "MetaInfoList", "MediaMap", "MediaMapSeq"]


class MetaInfo(Item):
    """歌曲、视频等媒体文件级别的元信息"""

    def __init__(
        self,
        mediaid=None,
        media_name=None,
        media_duration=None,
        signature=None,
        media_path: str = None,
        media_map_path: str = None,
        start: float = None,
        end: float = None,
        ext=None,
        **kwargs,
    ):
        super(MetaInfo).__init__()
        self.mediaid = mediaid
        self.media_name = media_name
        self.media_duration = media_duration
        self.signature = signature
        self.media_path = media_path
        self.media_map_path = media_map_path
        self.start = start
        self.end = end
        self.ext = ext
        self.__dict__.update(**kwargs)
        self.preprocess()

    def preprocess(self):
        self.set_start_end()

    def set_start_end(self):
        if self.start is None:
            self.start = 0
        elif self.start >= 0 and self.start <= 1:
            self.start = self.start * self.media_duration

        if self.end is None:
            self.end = self.media_duration
        elif self.end >= 0 and self.end <= 1:
            self.end = self.end * self.media_duration


class MetaInfoList(Items):
    """媒体元数据列表，主要用于多歌曲、多视频剪辑时存储原单一媒体文件的元信息"""

    def __init__(self, items: Union[MetaInfo, List[MetaInfo]] = None):
        """
        Args:
            meta_info_list (list, optional): MetaInfo 列表. Defaults to None.
        """
        if items is None:
            items = []
        else:
            items = items if isinstance(items, list) else [items]
        super().__init__(items)
        self.meta_info_list = self.items
        if len(self.items) > 1:
            self.reset()

    def __len__(self):
        return len(self.meta_info_list)

    def __getitem__(self, i) -> MetaInfo:
        return self.meta_info_list[i]

    @property
    def groupnum(self) -> int:
        return len(self.meta_info_list)


class MediaMap(object):
    """媒体信息基类，也可以理解为音乐谱面、视觉谱面、音游谱面基类。主要有 MetaInfo、MetaInfoList、ClipSeq 属性。
    不同的媒体信息的 属性 类会有不同，所以在类变量里做定义。如有变化，可以定义自己的属性类。
    """

    def __init__(
        self,
        meta_info: MetaInfo = None,
        clipseq: ClipSeq = None,
        stageseq: ClipSeq = None,
        frameseq: ClipSeq = None,
        emb: H5pyMediaMapEmb = None,
        **kwargs,
    ):
        """用于存储media的相关信息，media_info是json或直接字典

        Args:
            meta_info (MetaInfo): 当sub_meta_info不为None时, meta_info由sub_meta_info整合而成
            sub_meta_info (None or [MetaInfo]): 当多个MediaInfo拼在一起时,用于保留子MediaInfo的信息
            clipseq (ClipSeq): # 按照clipidx排序;
            stageseq (ClipSeq): # 比 clipseq 更高纬度的片段划分，例如clips是镜头分割，stages是scenes分割；clips是关键点分割，stages是结构分割；
            frameseq (ClipSeq): # 比 clipseq 更低纬度的片段划分
            kwargs (dict, optional): 所有相关信息都会作为 meta_info 的补充，赋值到 meta_info 中
        """
        self.meta_info = meta_info
        self.clipseq = clipseq
        self.frameseq = frameseq
        self.stageseq = stageseq
        self.emb = emb
        self.meta_info.__dict__.update(**kwargs)
        self.preprocess()

    def preprocess(
        self,
    ):
        if (self.meta_info.start != 0 and self.meta_info.start is not None) or (
            self.meta_info.end is not None and self.meta_info.end == 1
        ):
            self.drop_head_and_tail()
        self.meta_info.preprocess()
        if self.clipseq is not None:
            self.clipseq.preprocess()
        if self.frameseq is not None:
            self.frameseq.preprocess()
        if self.stageseq is not None:
            self.stageseq.preprocess()
        self.clip_start_idx = self.clipseq[0].clipid
        self.clip_end_idx = self.clipseq[-1].clipid

    def drop_head_and_tail(self) -> MediaMap:
        self.clipseq = get_subseq_by_time(
            self.clipseq,
            start=self.meta_info.start,
            end=self.meta_info.end,
            duration=self.meta_info.media_duration,
        )
        if self.stageseq is not None:
            self.stageseq = get_subseq_by_time(
                self.clipseq,
                start=self.meta_info.start,
                end=self.meta_info.end,
                duration=self.meta_info.media_duration,
            )

    def set_clip_value(self, k, v):
        """为clipseq中的每个clip赋值，

        Args:
            k (str): Clip中字段名
            v (any): Clip中字段值
        """
        self.clipseq.set_clip_value(k, v)

    def spread_metainfo_2_clip(
        self, target_keys: List = None, ignored_keys: List = None
    ) -> None:
        """将metainfo中的信息赋值到clip中，便于clip后面做相关处理。

        Args:
            target_keys ([str]): 待赋值的目标字段
        """
        dst = pick_subdct(
            self.meta_info.__dict__, target_keys=target_keys, ignored_keys=ignored_keys
        )
        for k, v in dst.items():
            self.set_clip_value(k, v)

    def spread_parameters(self, target_keys: list, ignored_keys) -> None:
        """元数据广播，将 media_info 的元数据广播到 clip 中，以及调用 clip 自己的参数传播。"""
        self.spread_metainfo_2_clip(target_keys=target_keys, ignored_keys=ignored_keys)
        for clip in self.clipseq:
            clip.spread_parameters()

    def stat(
        self,
    ):
        """统计 media_info 相关信息，便于了解，目前统计内容有
        1. 片段长度
        """
        self.stat_clipseq_duration()

    def stat_clipseq_duration(
        self,
    ):
        hist, bin_edges = stat_clipseq_duration(self.clipseq)
        print(self.media_name, "bin_edges", bin_edges)
        print(self.media_name, "hist", hist)

    def to_dct(self, target_keys: list = None, ignored_keys: list = None):
        raise NotImplementedError

    @property
    def duration(
        self,
    ):
        return self.clipseq.duration

    @property
    def mediaid(
        self,
    ):
        return self.meta_info.mediaid

    @property
    def media_name(
        self,
    ):
        return self.meta_info.media_name

    @property
    def duration_seq_emb(self):
        return self.clipseq.duration_seq_emb

    @property
    def timestamp_seq_emb(self):
        return self.clipseq.timestamp_seq_emb

    @property
    def rela_timestamp_seq_emb(self):
        return self.clipseq.rela_timestamp_seq_emb

    def get_emb(self, key, idx=None):
        # TODO: 待修改为更通用的形式
        if idx is None:
            idx = range(self.clip_start_idx, self.clip_end_idx + 1)
        elif isinstance(idx, int):
            idx += self.clip_start_idx
        elif isinstance(idx, Iterable):
            idx = [x + self.clip_start_idx for x in idx]
        else:
            raise ValueError(
                f"idx only support None, int, Iterable, but given {idx},type is {type(idx)}"
            )
        return self.emb.get_value(key, idx=idx)

    def get_meta_info_attr(self, key: str) -> Any:
        return getattr(self.meta_info, key)

    @classmethod
    def from_json_path(
        cls, path: Dict, emb_path: str, media_path: str = None, **kwargs
    ) -> MediaMap:
        media_map = load_dct_from_file(path)
        emb = H5pyMediaMapEmb(emb_path)
        return cls.from_data(media_map, emb=emb, media_path=media_path, **kwargs)


class MediaMapSeq(Items):
    def __init__(self, maps: List[MediaMap]) -> None:
        super().__init__(maps)
        self.maps = self.data
        self.preprocess()
        self.each_map_clipseq_num = [len(m.clipseq) for m in self.maps]
        self.each_map_clipseq_num_cumsum = np.cumsum([0] + self.each_map_clipseq_num)

    @property
    def clipseq(self):
        clipseq = []
        for m in self.maps:
            clipseq.extend(m.clipseq.data)
        return type(self.maps[0].clipseq)(clipseq)

    @property
    def stagesseq(self):
        stagesseq = []
        for m in self.maps:
            stagesseq.extend(m.stagesseq.data)
        return type(self.maps[0].stagesseq)(stagesseq)

    @property
    def frameseq(self):
        frameseq = []
        for m in self.maps:
            frameseq.extend(m.frameseq.data)
        return type(self.maps[0].frameseq)(frameseq)

    def preprocess(self):
        for m in self.maps:
            m.preprocess()

    def _combine_str(
        self,
        attrs: List[str],
        sep: str = "|",
        single_maxlen: int = 10,
        total_max_length: int = 60,
    ) -> str:
        return sep.join([str(attr)[:single_maxlen] for attr in attrs])[
            :total_max_length
        ]

    def get_meta_info_attr(self, key: str, func: Callable) -> Any:
        attrs = [m.get_meta_info_attr(key) for m in self.maps]
        return func(attrs)

    @property
    def mediaid(self) -> str:
        return self.get_meta_info_attr(key="mediaid", func=self._combine_str)

    @property
    def media_name(self) -> str:
        return self.get_meta_info_attr(key="media_name", func=self._combine_str)

    @property
    def duration(self) -> float:
        return sum([m.duration for m in self.maps])

    @property
    def media_duration(self) -> float:
        return self.get_meta_info_attr(key="media_duration", func=sum)

    @classmethod
    def from_json_paths(
        cls,
        media_map_class: MediaMap,
        media_paths: str,
        media_map_paths: str,
        emb_paths: str,
        **kwargs,
    ) -> MediaMapSeq:
        map_seq = [
            media_map_class.from_json_path(
                path=media_map_paths[i],
                emb_path=emb_paths[i],
                media_path=media_paths[i],
                **kwargs,
            )
            for i in range(len(media_map_paths))
        ]
        return cls(map_seq)

    # TODO: implement mapseq stat func
    def stat(self):
        for m in self.maps:
            m.stat()

    def _combine_embs(self, embs):
        return np.concatenate(embs, axis=0)

    @property
    def duration_seq_emb(self):
        embs = [m.duration_seq_emb for m in self.maps]
        return self._combine_embs(embs)

    @property
    def timestamp_seq_emb(self):
        embs = [m.timestamp_seq_emb for m in self.maps]
        return self._combine_embs(embs)

    @property
    def rela_timestamp_seq_emb(self):
        embs = [m.rela_timestamp_seq_emb for m in self.maps]
        return self._combine_embs(embs)

    def clip_idx_2_map_idx(self, idx):
        target_map_idx = bisect.bisect_right(self.each_map_clipseq_num_cumsum, idx)
        target_map_idx = min(max(0, target_map_idx - 1), len(self.maps) - 1)
        target_map_clip_idx = idx - self.each_map_clipseq_num_cumsum[target_map_idx]
        return target_map_idx, target_map_clip_idx

    def get_emb(self, key: str, idx: Union[None, int, List[int]] = None) -> np.array:
        if idx is None:
            embs = [m.get_emb(key, idx=idx) for m in self.maps]
        else:
            if not isinstance(idx, list):
                idx = [idx]
            embs = []
            for c_idx in idx:
                target_map_idx, target_map_clip_idx = self.clip_idx_2_map_idx(c_idx)
                embs.append(
                    self.maps[target_map_idx].get_emb(key, int(target_map_clip_idx))
                )
        if len(embs) == 1:
            return embs[0]
        else:
            return self._combine_embs(embs)
