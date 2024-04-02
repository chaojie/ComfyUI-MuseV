from __future__ import annotations
from typing import Dict, List

from ...data.clip import Clip, ClipSeq


class MusicClip(Clip):
    def __init__(self, time_start: float, duration: float, clipid: int = None, media_type: str = None, mediaid: str = None, timepoint_type: str = None, text: str = None, stage: str = None, path: str = None, duration_num: int = None, similar_clipseq: MatchedClipIds = None, dynamic: float = None, **kwargs):
        super().__init__(time_start, duration, clipid, media_type, mediaid, timepoint_type, text, stage, path, duration_num, similar_clipseq, dynamic, **kwargs)

    @property
    def text_num(self):
        return self._cal_text_num()

    @property
    def original_text_num(self):
        return self._cal_text_num(text_mode=1)

    def _cal_text_num(self, text_mode: int = 0) -> int:
        """计算 文本 字的数量

        Args:
            text_mode (int, optional): 0选text， 其他选original_text. Defaults to 0.

        Returns:
            int: _description_
        """
        if text_mode == 0:
            text = self.text
        else:
            text = self.original_text
        if text is None:
            n_text = 0
        else:
            text = text.strip().split(" ")
            n_text = len(text)
        return n_text

    @property
    def text_num_per_second(self):
        """单位时间内的text数量"""
        return self._cal_text_num_per_second(mode=0)

    @property
    def original_text_num_per_second(self):
        """单位时间内的original_text数量"""
        return self._cal_text_num_per_second(mode=1)

    @property
    def tnps(self):
        """单位时间内的text数量"""
        return self.text_num_per_second

    @property
    def original_tnps(self):
        """单位时间内的original_text数量"""
        return self.original_text_num_per_second

    def _cal_text_num_per_second(self, mode=0):
        """计算单位时间内的文本数量"""
        text_num = self.text_num if mode == 0 else self.original_text_num
        return text_num / self.duration

    @classmethod
    def from_data(cls, data: Dict):
        return MusicClip(**data)


class MusicClipSeq(ClipSeq):

    def __init__(self, items: List[Clip] = None):
        super().__init__(items)
        self.clipseq = self.data

    @classmethod
    def from_data(cls, clipseq: List[Dict]) -> MusicClipSeq:
        new_clipseq = []
        for clip in clipseq:
            video_clip = MusicClip.from_data(clip)
            new_clipseq.append(video_clip)
        video_clipseq = MusicClipSeq(new_clipseq)
        return video_clipseq
    