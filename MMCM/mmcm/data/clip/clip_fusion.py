from typing import List, Union, Callable

from copy import deepcopy

from .clip import ClipSeq
from .clip_process import reset_clipseq_id
import logging

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


# TODO: 不同类型的clip需要不同的融合方式
def fuse_clips(s1: ClipSeq, s2: ClipSeq) -> ClipSeq:
    """合并2个clip

    Args:
        s1 (Clip):
        s2 (Clip):

    Returns:
        Clip: 合并后Clip
    """
    if not isinstance(s2, list):
        s2 = [s2]
    s1 = deepcopy(s1)
    for other_clip in s2:
        s1.duration += other_clip.duration
        if s1.stage is not None and other_clip.stage is not None:
            # TODO：如何保留融合的clip信息
            s1.stage = "{}_{}".format(s1.stage, other_clip.stage)
            s1.origin_clipid.extend(other_clip.origin_clipid)
        if s1.timepoint_type is not None and other_clip.timepoint_type is not None:
            s1.timepoint_type = "{}_{}".format(
                s1.timepoint_type, other_clip.timepoint_type
            )
    return s1


# TODO: 不同的filter和fusion函数不适用同一种流程，待优化
class ClipSeqFusion(object):
    """_summary_

    Args:
        object (_type_): _description_
    """

    def __init__(self, filter: Callable, fuse_func: Callable = None) -> None:
        self.filter = filter
        self.fuse_func = fuse_func

    def __call__(self, clipseq: ClipSeq) -> ClipSeq:
        new_clipseq = []
        n_clipseq = len(clipseq)
        for i in range(n_clipseq):
            clip = clipseq[i]
            if self.filter(clip):
                new_clipseq.append(clip)
        new_clipseq = reset_clipseq_id(new_clipseq)
        logger.debug(
            "ClipSeqFilter: clipseq length before={}, after={}".format(
                n_clipseq, len(new_clipseq)
            )
        )
        return new_clipseq
