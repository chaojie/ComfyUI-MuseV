from functools import partial
from copy import deepcopy
from typing import Iterable, List, Tuple, Union
import bisect
import logging

import numpy as np


from .clip import Clip, ClipSeq
from .clipid import ClipIds, ClipIdsSeq, MatchedClipIds, MatchedClipIdsSeq

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

__all__ = [
    "find_idx_by_rela_time",
    "find_idx_by_time",
    "find_idx_by_clip",
    "get_subseq_by_time",
    "get_subseq_by_idx",
    "clip_is_top",
    "clip_is_middle",
    "clip_is_end",
    "abadon_old_return_new",
    "reset_clipseq_id",
    "insert_endclip",
    "insert_startclip",
    "drop_start_end_by_time",
    "complete_clipseq",
    "complete_gap",
    "get_subseq_by_stages",
    "find_time_by_stage",
]


def find_idx_by_rela_time(clipseq: ClipSeq, timepoint: float) -> int:
    clipseq_duration = clipseq.duration
    timepoint = clipseq_duration * timepoint
    clipseq_times = [c.duration for c in clipseq]
    clipseq_times.insert(0, 0)
    clipseq_times = np.cumsum(clipseq_times)
    idx = bisect.bisect_right(clipseq_times, timepoint)
    idx = min(max(0, idx - 1), len(clipseq) - 1)
    return idx


def find_idx_by_time(clipseq: ClipSeq, timepoint: float) -> int:
    """寻找指定时间timepoint 在 clipseq 中的片段位置

    Args:
        clipseq (ClipSeq): 待寻找的片段序列
        timepoint (float): 指定时间位置

    Returns:
        _type_: _description_
    """
    clipseq_times = [c.time_start for c in clipseq]
    idx = bisect.bisect_right(clipseq_times, timepoint)
    idx = min(max(0, idx - 1), len(clipseq) - 1)
    return idx


def find_idx_by_clip(clipseq: ClipSeq, clip: Clip, eps: float = 1e-4) -> int:
    """通过计算目标clip和clipseq中所有候选clip的交集占比来找最近clip

    Args:
        clipseq (ClipSeq): 候选clip序列
        clip (Clip): 目标clip
        eps (float, optional): 最小交集占比. Defaults to 1e-4.

    Returns:
        int: 目标clip在候选clip序列的位置，若无则为None
    """
    timepoints = np.array([[c.time_start, c.time_start + c.duration] for c in clipseq])
    clip_time_start = clip.time_start
    clip_duraiton = clip.duration
    clip_time_end = clip_time_start + clip_duraiton
    max_time_start = np.maximum(timepoints[:, 0], clip_time_start)
    min_time_end = np.minimum(timepoints[:, 1], clip_time_end)
    intersection = min_time_end - max_time_start
    intersection_ratio = intersection / clip_duraiton
    max_intersection_ratio = np.max(intersection_ratio)
    idx = np.argmax(intersection_ratio) if max_intersection_ratio > eps else None
    return idx


def get_subseq_by_time(
    clipseq: ClipSeq,
    start: float = 0,
    duration: float = None,
    end: float = 1,
    eps: float = 1e-2,
) -> ClipSeq:
    """根据时间对媒体整体做掐头去尾，保留中间部分。，也可以是大于1的数。
        start和end如果是0-1的小数，则认为是是相对时间位置，实际位置会乘以duration；
        start和end如果是大于1的数，则是绝对时间位置。

    Args:
        clipseq (ClipSeq): 待处理的序列
        start (float,): 保留部分的开始，. Defaults to 0.
        duration (float, optional): 媒体文件当前总时长
        end (float, optional): 保留部分的结尾. Defaults to 1.

    Returns:
        ClipSeq: 处理后的序列
    """
    if (start == 0 or start is None) and (end is None or end == 1):
        logger.warning("you should set start or end")
        return clipseq
    if duration is None:
        duration = clipseq.duration
    if start is None or start == 0:
        clip_start_idx = 0
    else:
        if start < 1:
            start = start * duration
        clip_start_idx = find_idx_by_time(clipseq, start)
    if end is None or end == 1 or np.abs(duration - end) < eps:
        clip_end_idx = -1
    else:
        if end < 1:
            end = end * duration
        clip_end_idx = find_idx_by_time(clipseq, end)
    if clip_end_idx != -1 and clip_start_idx >= clip_end_idx:
        logger.error(
            f"clip_end_idx({clip_end_idx}) should be > clip_start_idx({clip_start_idx})"
        )
    subseq = get_subseq_by_idx(clipseq, clip_start_idx, clip_end_idx)
    return subseq


def get_subseq_by_idx(clipseq: ClipSeq, start: int = None, end: int = None) -> ClipSeq:
    """通过指定索引范围，切片子序列

    Args:
        clipseq (ClipSeq):
        start (int, optional): 开始索引. Defaults to None.
        end (int, optional): 结尾索引. Defaults to None.

    Returns:
        _type_: _description_
    """
    if start is None and end is None:
        return clipseq
    if start is None:
        start = 0
    if end is None:
        end = len(clipseq)
    return clipseq[start:end]


def clip_is_top(clip: Clip, total: float, th: float = 0.1) -> bool:
    """判断Clip是否属于开始部分

    Args:
        clip (Clip):
        total (float): 所在ClipSeq总时长
        th (float, optional): 开始范围的截止位置. Defaults to 0.05.

    Returns:
        Bool: 是不是头部Clip
    """
    clip_time = clip.time_start
    if clip_time / total <= th:
        return True
    else:
        return False


def clip_is_end(clip: Clip, total: float, th: float = 0.9) -> bool:
    """判断Clip是否属于结尾部分

    Args:
        clip (Clip):
        total (float): 所在ClipSeq总时长
        th (float, optional): 结尾范围的开始位置. Defaults to 0.9.

    Returns:
        Bool: 是不是尾部Clip
    """
    clip_time = clip.time_start + clip.duration
    if clip_time / total >= th:
        return True
    else:
        return False


def clip_is_middle(
    clip: Clip, total: float, start: float = 0.05, end: float = 0.9
) -> bool:
    """判断Clip是否属于中间部分

    Args:
        clip (Clip):
        total (float): 所在ClipSeq总时长
        start (float, optional): 中间范围的开始位置. Defaults to 0.05.
        start (float, optional): 中间范围的截止位置. Defaults to 0.9.

    Returns:
        Bool: 是不是中间Clip
    """
    if start >= 0 and start < 1:
        start = total * start
    if end > 0 and end <= 1:
        end = total * end
    clip_time_start = clip.time_start
    clip_time_end = clip.time_start + clip.duration
    if (clip_time_start >= start) and (clip_time_end <= end):
        return True
    else:
        return False


def abadon_old_return_new(s1: Clip, s2: Clip) -> Clip:
    """特殊的融合方式
    Args:

        s1 (Clip): 靠前的clip
        s2 (Clip): 靠后的clip

    Returns:
        Clip: 融合后的Clip
    """
    return s2


# TODO：待确认是否要更新clipid，不方便对比着json进行debug
def reset_clipseq_id(clipseq: ClipSeq) -> ClipSeq:
    for i in range(len(clipseq)):
        if isinstance(clipseq[i], dict):
            clipseq[i]["clipid"] = i
        else:
            clipseq[i].clipid = i
    return clipseq


def insert_startclip(clipseq: ClipSeq) -> ClipSeq:
    """给ClipSeq插入一个开始片段。

    Args:
        clipseq (ClipSeq):
        clip_class (Clip, optional): 插入的Clip类型. Defaults to Clip.

    Returns:
        ClipSeq: 插入头部Clip的新ClipSeq
    """
    if clipseq[0].time_start > 0:
        start = clipseq.ClipClass(
            time_start=0, duration=round(clipseq[0].time_start, 3), timepoint_type=0
        )
        clipseq.insert(0, start)
    clipseq = reset_clipseq_id(clipseq)
    return clipseq


def insert_endclip(clipseq: ClipSeq, duration: float) -> ClipSeq:
    """给ClipSeq插入一个尾部片段。

    Args:
        clipseq (ClipSeq):
        duration(float, ): 序列的总时长
        clip_class (Clip, optional): 插入的Clip类型. Defaults to Clip.

    Returns:
        ClipSeq: 插入尾部Clip的新ClipSeq
    """
    clipseq_endtime = clipseq[-1].time_start + clipseq[-1].duration
    if duration - clipseq_endtime > 1:
        end = clipseq.ClipClass(
            time_start=round(clipseq_endtime, 3),
            duration=round(duration - clipseq_endtime, 3),
            timepoint_type=0,
        )
        clipseq.append(end)
    clipseq = reset_clipseq_id(clipseq)
    return clipseq


def drop_start_end_by_time(
    clipseq: ClipSeq, start: float, end: float, duration: float = None
):
    return get_subseq_by_time(clipseq=clipseq, start=start, end=end, duration=duration)


def complete_clipseq(
    clipseq: ClipSeq, duration: float = None, gap_th: float = 2
) -> ClipSeq:
    """绝大多数需要clipseq中的时间信息是连续、完备的，有时候是空的，需要补足的部分。
    如歌词时间戳生成的music_map缺头少尾、中间有空的部分。

    Args:
        clipseq (ClipSeq): 待补集的序列
        duration (float, optional): 整个序列持续时间. Defaults to None.
        gap_th (float, optional): 有时候中间空隙过短就会被融合到上一个片段中. Defaults to 2.

    Returns:
        ClipSeq: 补集后的序列，时间连续、完备。
    """
    if isinstance(clipseq, list):
        clipseq = ClipSeq(clipseq)
        return complete_clipseq(clipseq=clipseq, duration=duration, gap_th=gap_th)
    clipseq = complete_gap(clipseq, th=gap_th)
    clipseq = insert_startclip(clipseq)
    if duration is not None:
        clipseq = insert_endclip(clipseq, duration)
    return clipseq


def complete_gap(clipseq: ClipSeq, th: float = 2) -> ClipSeq:
    """generate blank clip timepoint = 0，如果空白时间过短，则空白附到上一个歌词片段中。


    Args:
        clipseq (ClipSeq): 原始的歌词生成的MusicClipSeq
        th (float, optional): 有时候中间空隙过短就会被融合到上一个片段中. Defaults to 2.

    Returns:
        ClipSeq: 补全后的
    """
    gap_clipseq = []
    clipid = 0
    for i in range(len(clipseq) - 1):
        time_start = clipseq[i].time_start
        duration = clipseq[i].duration
        time_end = time_start + duration
        next_time_start = clipseq[i + 1].time_start
        time_diff = next_time_start - time_end
        if time_diff >= th:
            blank_clip = clipseq.ClipClass(
                time_start=time_end,
                duration=time_diff,
                timepoint_type=0,
                clipid=clipid,
            )
            gap_clipseq.append(blank_clip)
            clipid += 1
        else:
            clipseq[i].duration = next_time_start - time_start
    clipseq.extend(gap_clipseq)
    clipseq.clips = sorted(clipseq.clips, key=lambda clip: clip.time_start)
    reset_clipseq_id(clipseq)
    return clipseq


def find_time_by_stage(
    clipseq: ClipSeq, stages: Union[str, List[str]] = None
) -> Tuple[float, float]:
    if isinstance(stages, list):
        stages = [stages]
    for clip in clipseq:
        if clip.stage in stages:
            return clip.time_start, clip.time_end
    return None, None


def get_subseq_by_stages(clipseq: ClipSeq, stages: Union[str, List[str]]) -> ClipSeq:
    if isinstance(stages, List):
        stages = [stages]
    start, _ = find_time_by_stage(clipseq, stages[0])
    _, end = find_time_by_stage(clipseq, stages[-1])
    if start1 is None:
        start1 = 0
    if end2 is None:
        end2 = clipseq.duration
    subseq = get_subseq_by_time(clipseq=clipseq, start=start, end=end)
    return subseq
