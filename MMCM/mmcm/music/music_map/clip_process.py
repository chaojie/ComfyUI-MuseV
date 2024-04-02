from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List

import numpy as np

from ...data.clip.clip_process import find_idx_by_time, reset_clipseq_id
from ...data.clip.clip_fusion import fuse_clips
from ...utils.util import merge_list_continuous_same_element

if TYPE_CHECKING:
    from .music_clip import MusicClip, MusicClipSeq
    from .music_map import MusicMap, MusicMapSeq


# TODO: 待和clip操作做整合
def music_clip_is_short(clip: MusicClip, th: float = 3) -> bool:
    """判断音乐片段是否过短

    Args:
        clip (MusicClip): 待判断的音乐片段
        th (float, optional): 短篇的参数. Defaults to 3.

    Returns:
        bool: 是或不是 短片段
    """
    if clip.duration < th:
        return False
    else:
        return True


def music_clip_timepoint_is_target(clip: MusicClip, target: list = [-1, 1, 0]) -> bool:
    """音乐片段的关键点类型是否是目标关键点
    关键点类型暂时参考：VideoMashup/videomashup/data_structure/music_data_structure.py
    Args:
        clip (MusicClip): 待判断的音乐片段
        target (list, optional): 目标关键点类别. Defaults to [-1, 1, 0].

    Returns:
        bool: 是还是不是
    """
    timepoint = clip.timepoint_type
    if isinstance(timepoint, int):
        timepoint = {timepoint}
    else:
        timepoint = {int(x) for x in timepoint.split("_")}
    if timepoint & set(target):
        return True
    else:
        return False


def filter_clipseq_target_point(
    clipseq: MusicClipSeq, target: list = [-1, 1, 0]
) -> MusicClipSeq:
    """删除目标关键点之外的点，对相应的片段做融合

    Args:
        clipseq (MusicClipSeq): 待处理的音乐片段序列
        target (list, optional): 保留的目标关键点. Defaults to [-1, 1, 0].

    Returns:
        MusicClipSeq: 处理后的音乐片段序列
    """
    n_clipseq = len(clipseq)
    if n_clipseq == 1:
        return clipseq
    newclipseq = []
    start_clip = clipseq[0]
    if music_clip_timepoint_is_target(start_clip, target=target):
        has_start_clip = True
    else:
        has_start_clip = False
    i = 1
    while i <= n_clipseq - 1:
        clip = clipseq[i]
        start_clip_is_target = music_clip_timepoint_is_target(start_clip, target=target)
        next_clip_is_target = music_clip_timepoint_is_target(clip, target=target)
        # logger.debug("filter_clipseq_target_point: i={},start={}, clip={}".format(i, start_clip["timepoint_type"], clip["timepoint_type"]))
        # logger.debug("start_clip_is_target: {}, next_clip_is_target {}".format(start_clip_is_target, next_clip_is_target))
        if not has_start_clip:
            start_clip = clip
            has_start_clip = next_clip_is_target
        else:
            if start_clip_is_target:
                has_start_clip = True
                if next_clip_is_target:
                    newclipseq.append(start_clip)
                    start_clip = clip
                    if i == n_clipseq - 1:
                        newclipseq.append(clip)
                else:
                    start_clip = fuse_clips(start_clip, clip)
                    if i == n_clipseq - 1:
                        newclipseq.append(start_clip)
                    # logger.debug("filter_clipseq_target_point: fuse {}, {}".format(i, clip["timepoint_type"]))
            else:
                start_clip = clip
        i += 1
    newclipseq = reset_clipseq_id(newclipseq)
    return newclipseq


def merge_musicclip_into_clipseq(
    clip: MusicClipSeq, clipseq: MusicClip, th: float = 1
) -> MusicClipSeq:
    """给clipseq插入一个新的音乐片段，会根据插入后片段是否过短来判断。

    Args:
        clip (MusicClipSeq): 要插入的音乐片段
        clipseq (MusicClip): 待插入的音乐片段序列
        th (float, optional): 插入后如果受影响的片段长度过短，则放弃插入. Defaults to 1.

    Returns:
        MusicClipSeq: _description_
    """
    n_clipseq = len(clipseq)
    clip_time = clip.time_start
    idx = find_idx_by_time(clipseq, clip_time)
    last_clip_time_start = clipseq[idx].time_start
    next_clip_time_start = clipseq[idx].time_start + clipseq[idx].duration
    last_clip_time_delta = clip_time - last_clip_time_start
    clip_duration = next_clip_time_start - clip_time
    # TODO: 副歌片段改变th参数来提升音符密度，暂不使用，等待音游谱面
    # TODO: 待抽离独立的业务逻辑为单独的函数
    # 只针对副歌片段插入关键点
    if clipseq[idx].text is None or (
        clipseq[idx].text is not None
        and clipseq[idx].stage is not None
        and "C" in clipseq[idx].stage
    ):
        if (last_clip_time_delta > th) and (clip_duration > th):
            clip.duration = clip_duration
            clipseq[idx].duration = last_clip_time_delta
            clipseq.insert(idx + 1, clip)
        clipseq = reset_clipseq_id(clipseq)
    return clipseq


def merge_music_clipseq(clipseq1: MusicClipSeq, clipseq2: MusicClipSeq) -> MusicClipSeq:
    """将片段序列clipseq2融合到音乐片段序列clipseq1中。融合过程也会判断新片段长度。

    Args:
        clipseq1 (MusicClipSeq): 要融合的目标音乐片段序列
        clipseq2 (MusicClipSeq): 待融合的音乐片段序列

    Returns:
        MusicClipSeq: 融合后的音乐片段序列
    """
    while len(clipseq2) > 0:
        clip = clipseq2[0]
        clipseq1 = merge_musicclip_into_clipseq(clip, clipseq1)
        del clipseq2[0]
    return clipseq1


def merge_lyricseq_beatseq(
    lyric_clipseq: MusicClipSeq, beat_clipseq: MusicClipSeq
) -> MusicClipSeq:
    """将beat序列融合到歌词序列中

    Args:
        lyric_clipseq (MusicClipSeq): 歌词序列
        beat_clipseq (MusicClipSeq): beat序列

    Returns:
        MusicClipSeq: 融合后的音乐片段序列
    """
    newclipseq = merge_music_clipseq(lyric_clipseq, beat_clipseq)
    # for i, clip in enumerate(newclipseq):
    # logger.debug("i={}, time_start={}, duration={}".format(i, clip.time_start, clip.duration))
    return newclipseq


def get_stageseq_from_clipseq(clipseq: MusicClipSeq) -> List[Dict]:
    """对clip.stage做近邻融合，返回总时间

    Returns:
        List[Dict]: 根据音乐结构进行分割的片段序列
    """
    stages = [clip.stage for clip in clipseq]
    merge_stages_idx = merge_list_continuous_same_element(stages)
    merge_stages = []
    for n, stages_idx in enumerate(merge_stages_idx):
        dct = {
            "clipid": n,
            "time_start": clipseq[stages_idx["start"]].time_start,
            "time_end": clipseq[stages_idx["end"]].time_end,
            "stage": stages_idx["element"],
            "original_clipid": list(
                range(stages_idx["start"], stages_idx["end"] + 1)
            ),  # mss都是左闭、 右闭的方式
        }
        dct["duration"] = dct["time_end"] - dct["time_start"]
        merge_stages.append(dct)
    return merge_stages
