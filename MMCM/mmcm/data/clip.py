from copy import deepcopy
from typing import Iterable
import logging

import numpy as np

from ..utils.util import convert_class_attr_to_dict

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Clip(object, Item):
    """媒体片段, 指转场点与转场点之间的部分"""

    def __init__(
        self,
        time_start,
        duration,
        clipid=None,
        media_type=None,
        mediaid=None,
        timepoint_type=None,
        text=None,
        stage=None,
        path=None,
        duration_num=None,
        group_time_start=0,
        group_clipid=None,
        original_clipid=None,
        emb=None,
        multi_factor=None,
        similar_clipseq=None,
        rythm: float = None,
        **kwargs
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
            path (_type_, optional): 该媒体文件的路径,用于后续媒体读取、处理. Defaults to None.
            duration_num (_type_, optional): 片段持续帧数, Defaults to None.
            group_time_start (int, optional): 当多歌曲、多视频剪辑时,group_time_start 表示该片段所对应的子媒体前所有子媒体的片段时长总和。
                默认0, 表示只有1个媒体文件. Defaults to 0.
            group_clipid (int, optional):  # MediaInfo.sub_meta_info 中的实际序号.
            original_clipid (None or [int], optional): 有些片段由其他片段合并,该字段用于片段来源,id是 media_map.json 中的实际序号. Defaults to None.
            emb (np.array, optional): 片段 综合emb,. Defaults to None.
            multi_factor (MultiFactorFeature), optional): 多维度特征. Defaults to None.
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
        self.group_time_start = group_time_start
        self.group_clipid = group_clipid
        self.duration_num = duration_num
        self.original_clipid = original_clipid if original_clipid is not None else []
        self.emb = emb
        self.multi_factor = multi_factor
        self.similar_clipseq = similar_clipseq
        self.rythm = rythm
        # TODO: 目前谱面中会有一些不必要的中间结果，比较占内存，现在代码里删掉，待后续数据协议确定
        kwargs = {k: v for k, v in kwargs.items()}
        self.__dict__.update(kwargs)
        self.preprocess()

    def preprocess(self):
        pass

    def spread_parameters(self):
        pass

    @property
    def time_end(
        self,
    ):
        return self.time_start + self.duration

    @property
    def mvp_clip(self):
        """读取实际的片段数据为moviepy格式

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError


class ClipSeq(object):
    """媒体片段序列"""

    ClipClass = Clip

    def __init__(self, clips) -> None:
        """_summary_

        Args:
            clips ([Clip]]): 媒体片段序列
        """
        if not isinstance(clips, list):
            clips = [clips]
        if len(clips) == 0:
            self.clips = []
        elif isinstance(clips[0], dict):
            self.clips = [self.ClipClass(**d) for d in clips]
        else:
            self.clips = clips

    def set_clip_value(self, k, v):
        """给序列中的每一个clip 赋值"""
        for i in range(len(self.clips)):
            self.clips[i].__setattr__(k, v)

    def __len__(
        self,
    ):
        return len(self.clips)

    def merge(self, other, group_time_start_delta=None, groupid_delta=None):
        """融合其他ClipSeq。media_info 融合时需要记录 clip 所在的 groupid 和 group_time_start，delta用于表示变化

        Args:
            other (ClipSeq): 待融合的ClipSeq
            group_time_start_delta (float, optional): . Defaults to None.
            groupid_delta (int, optional): _description_. Defaults to None.
        """
        if group_time_start_delta is not None or groupid_delta is not None:
            for i, clip in enumerate(other):
                if group_time_start_delta is not None:
                    clip.group_time_start += group_time_start_delta
                if groupid_delta is not None:
                    clip.groupid += groupid_delta
        self.clips.extend(other.clips)
        for i in range(len(self.clips)):
            self.clips[i].group_clipid = i

    @property
    def duration(
        self,
    ):
        """Clip.duration的和

        Returns:
            float: 序列总时长
        """
        if len(self.clips) == 0:
            return 0
        else:
            return sum([c.duration for c in self.clips])

    def __getitem__(self, i) -> Clip:
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
            clip = self.clips[i]
            return clip
        elif isinstance(i, Iterable):
            clips = [self.__getitem__(x) for x in i]
            clipseq = ClipSeq(clips)
            return clipseq
        elif isinstance(i, slice):
            if i.step is None:
                step = 1
            else:
                step = i.step
            clips = [self.__getitem__(x) for x in range(i.start, i.stop, step)]
            clipseq = ClipSeq(clips)
            return clipseq
        else:
            raise ValueError(
                "unsupported input, should be int or slice, but given {}, type={}".format(
                    i, type(i)
                )
            )

    def insert(self, idx, obj):
        self.clips.insert(idx, obj)

    def append(self, obj):
        self.clips.append(obj)

    def extend(self, objs):
        self.clips.extend(objs)

    @property
    def duration_seq_emb(
        self,
    ):
        emb = np.array([c.duration for c in self.clips])
        return emb

    @property
    def timestamp_seq_emb(self):
        emb = np.array([c.time_start for c in self.clips])
        return emb

    @property
    def rela_timestamp_seq_emb(self):
        emb = self.timestamp_seq_emb / self.duration
        return emb

    def get_factor_seq_emb(self, factor, dim):
        emb = []
        for c in self.clips:
            if factor not in c.multi_factor or c.multi_factor[factor] is None:
                v = np.full(dim, np.inf)
            else:
                v = c.multi_factor[factor]
            emb.append(v)
        emb = np.stack(emb, axis=0)
        return emb

    def semantic_seq_emb(self, dim):
        return self.get_factor_seq_emb(factor="semantics", dim=dim)

    def emotion_seq_emb(self, dim):
        return self.get_factor_seq_emb(factor="emotion", dim=dim)

    def theme_seq_emb(self, dim):
        return self.get_factor_seq_emb(factor="theme", dim=dim)

    def to_dct(
        self,
        target_keys=None,
        ignored_keys=None,
    ):
        if ignored_keys is None:
            ignored_keys = ["kwargs", "audio_path", "lyric_path", "start", "end"]
        clips = [
            clip.to_dct(target_keys=target_keys, ignored_keys=ignored_keys)
            for clip in self.clips
        ]
        return clips

    @property
    def mvp_clip(self):
        """读取实际的片段数据为moviepy格式

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError


class ClipIds(object):
    def __init__(
        self,
        clipids: list or int,
    ) -> None:
        """ClipSeq 中的 Clip序号，主要用于多个 Clip 融合后的 Clip, 使用场景如
        1. 一个 MusicClip 可以匹配到多个 VideoClip，VideoClip 的索引便可以使用 ClipIds 定义。

        Args:
            clipids (list or int): ClipSeq 中的序号
        """
        self.clipids = clipids if isinstance(clipids, list) else [clipids]


class ClipIdsSeq(object):
    def __init__(self, clipids_seq: list) -> None:
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
    def __init__(self, seq: list, metric: float = None, **kwargs) -> None:
        """两种模态数据的序列匹配对，使用场景可以是
        1. 音乐片段序列和视频片段序列 之间的匹配，每一个元素都是MatchedClipIds:

        Args:
            seq (list): 两种模态数据的序列匹配对列表
            metric (float): 匹配度量距离
        """
        self.seq = seq
        self.metric = metric
        self.__dict__.update(**kwargs)
