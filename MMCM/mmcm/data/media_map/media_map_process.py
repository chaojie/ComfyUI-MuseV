from __future__ import annotations

from typing import List, Union, TYPE_CHECKING
from ..clip.clip_process import (
    get_subseq_by_time,
    find_time_by_stage, 

)
if TYPE_CHECKING:
    from ..media_map.media_map import MediaMap
    from ..clip import Clip, ClipSeq


__all__ =[
    "get_sub_mediamap_by_clip_idx",
    "get_sub_mediamap_by_stage",
    "get_sub_mediamap_by_time",
]


def get_sub_mediamap_by_time(media_map:MediaMap, start: int=0, end:int=1, eps=1e-2) -> MediaMap:
    """获取子片段序列，同时更新media_map中的相关信息

    Args:
        media_map (MediaInfo): _description_
        start (float): 开始时间
        end (float): 结束时间

    Returns:
        _type_: _description_
    """
    if start < 1:
        start = media_map.duration * start
    if end is None:
        end = media_map.meta_info.media_duration
    elif end <= 1:
        end = media_map.duration * end
    media_map.meta_info.start = start
    media_map.meta_info.end = end
    media_map.clipseq = get_subseq_by_time(
        media_map.clipseq,
        start=start,
        end=end,
    )
    if media_map.stageseq is not None:
        media_map.stageseq = get_subseq_by_time(media_map.stageseq, start=start, end=end)   
    return media_map


def get_sub_mediamap_by_clip_idx(media_map: MediaMap, start: int=None, end: int=None) -> MediaMap:
    """不仅获取子片段序列，还要更新media_map中的相关信息

    Args:
        media_map (_type_): _description_
    """
    if start is None:
        start = 0
    if end is None:
        end = -1
    start = media_map.clipseq[start].time_start
    end = media_map.clipseq[end].time_end
    media_map = get_sub_mediamap_by_time(media_map=media_map, start=start, end=end)      
    return media_map


def get_sub_mediamap_by_stage(media_map: MediaMap, stages: Union[str, List[str]]) -> MediaMap:
    if isinstance(stages, List):
        stages = [stages]
    start, _ = find_time_by_stage(media_map.stageseq, stages[0])
    _, end = find_time_by_stage(media_map.stageseq, stages[-1])
    media_map = get_sub_mediamap_by_time(media_map=media_map, start=start, end=end)
    return media_map
