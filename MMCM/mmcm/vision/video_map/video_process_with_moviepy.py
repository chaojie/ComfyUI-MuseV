import math
from heapq import nsmallest
import logging

import numpy as np
import cv2
from moviepy.editor import (
    VideoFileClip,
    VideoClip,
    concatenate_videoclips,
    vfx,
    TextClip,
    CompositeVideoClip,
)

from ..utils.vision_util import (
    cal_crop_coord,
    round_up_coord_to_even,
    cal_small_bbox_coord_of_big_bbox,
)

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class VideoClipOperator(object):
    def __init__(self, *args, **kwds) -> None:
        pass

    def __call__(self, *args, **kwds):
        pass


def get_subclip_from_clipseq_by_time():
    pass


def get_mvpclip_from_clip_by_time(
    clips, final_duration: float, method: str = None, delta=0
):
    """根据视频长度，对齐到指定长度

    Args:
        clips (VideoClipSeq): 媒体文件片段序列
        final_duration (float): 目标长度
        method (int, optional): how to chang video length. Defaults to `None`.
            speed: chang length by sample
            cut: change length by cut middle length
            None: change length accorrding difference of clip duration and final_duration. Defaults to None.

    Returns:
        VideoClip: 读取、对齐后moviepy VideoClip
    """
    n_clips = len(clips)
    video_clips = []
    for i, clip in enumerate(clips):
        start_delta = 0
        end_delta = 0
        # TODO: 为了解决夹帧问题，视视觉片段长音乐片段一些，便于只取中间部分。
        ## 适用于多个视频源的片段
        ## 适用于同一个视频源的 多个连续片段
        if n_clips > 1:
            if i == 0:
                start_delta = delta
            if i == n_clips - 1:
                end_delta = delta
        else:
            start_delta = delta
            end_delta = delta
        video_clip = clip.get_mvp_clip(start_delta=start_delta, end_delta=end_delta)
        video_clips.append(video_clip)
    video_clips = concatenate_videoclips(clips=video_clips, method="compose")
    video_clips = get_sub_mvpclip_by_time(
        clip=video_clips, final_duration=final_duration, method=method
    )
    return video_clips


def get_sub_mvpclip_by_time(
    clip, final_duration: float, method: str = "speed", center_ratio: float = 0.5
):
    duration = clip.duration
    center = duration * center_ratio
    center = min(max(center, final_duration / 2), duration - final_duration / 2)
    if method == "speed":
        clip = clip.fx(vfx.speedx, final_duration=final_duration)
    elif method == "cut" or method is None:
        if duration >= final_duration:
            t_start = center - final_duration / 2
            t_end = center + final_duration / 2
            clip = clip.subclip(t_start, t_end)
            logger.debug(
                "[cut_clip_time]: change length by cut: t_start={:.3f}, t_end={:.3f},  duration={:.3f}, final_duration={:.3f}".format(
                    t_start, t_end, duration, final_duration
                )
            )
        clip = clip.fx(vfx.speedx, final_duration=final_duration)
    else:
        raise NotImplementedError(
            "var_video_clip_length do not support mode={}".format(clip)
        )
    return clip


def crop_by_ratio(
    clip, target_width_height_ratio, restricted_bbox=None, need_round2even=False
):
    """将原视频中的有效部分剪辑成目标宽高比，有效部分用坐标表示，一般来说是非黑边、非水印位置

    Args:
        clip (VideoClip): moviepy中的视频片段
        target_width_height_ratio (float): 目标宽高比，常见的有2.35, 1.777, 0.75, 1, 0.5625
        restricted_bbox ((float, float, float, float), optional): (x1, y1, x2, y2). Defaults to None.

    Returns:
        VideoClip: 剪辑好的moviepy视频片段
    """
    width = clip.w
    height = clip.h
    target_coord = cal_crop_coord(
        width=width,
        height=height,
        target_width_height_ratio=target_width_height_ratio,
        restricted_bbox=restricted_bbox,
    )
    if need_round2even:
        target_coord = round_up_coord_to_even(*target_coord)
    clip = clip.crop(*target_coord)
    return clip


def crop_by_perception(
    clip,
    target_width_height_ratio: float,
    perception: dict,
    need_round2even: bool = True,
):
    """将原视频中的有效部分剪辑成目标宽高比，有效部分用坐标表示，一般来说是非黑边、非水印位置

    Args:
        clip (VideoClip): moviepy中的视频片段
        target_width_height_ratio (float): 目标宽高比，常见的有2.35, 1.777, 0.75, 1, 0.5625

    Returns:
        VideoClip: 剪辑好的moviepy视频片段
    """

    return crop_by_face_clip(
        clip, target_width_height_ratio, perception, need_round2even
    )


def crop_by_face_clip(
    clip,
    target_width_height_ratio: float,
    perception,
    need_round2even: bool = True,
    topk: int = 1,
):
    w = clip.w
    h = clip.h
    target_w = target_width_height_ratio * h
    perception_objs = []
    if len(perception) > 0:
        for i, frame_perception in enumerate(perception.clips):
            if frame_perception.objs is not None:
                for obj in frame_perception.objs:
                    perception_objs.append({"bbox": obj.bbox, "trackid": obj.trackid})
    # 如果没有目标人物，则依然使用中间crop方式
    if len(perception) == 0 or len(perception_objs) == 0:
        return crop_by_ratio(
            clip, target_width_height_ratio, need_round2even=need_round2even
        )
    topk_rolid = nsmallest(topk, [obj["trackid"] for obj in perception_objs])
    topk_clip = [obj for obj in perception_objs if obj["trackid"] in topk_rolid]
    # TODO: topk_clip 具有时间的先后顺序，先暂定取中间的obj的框作为参考
    target_idx = int(len(topk_clip) // 2)
    x1, y1, x2, y2 = topk_clip[target_idx]["bbox"]
    # TODO：当前适用于 target_w 大于 obj_width对应的人体宽度，当不符合条件时存在crop部分人体部分情况，此时应该提前过滤。
    obj_width = x2 - x1
    obj_height = y2 - y1
    obj_center_width = (x1 + x2) / 2
    obj_center_height = (y1 + y2) / 2
    target_coord = cal_small_bbox_coord_of_big_bbox(
        bigbbox_width=w,
        bigbbox_height=h,
        smallbbox_width=target_w,
        smallbbox_height=obj_height,
        center_width=obj_center_width,
        center_height=obj_center_height,
        need_round2even=need_round2even,
    )
    clip = clip.mv.crop(*target_coord)
    return clip


def crop_target_bbox(clip, target_coord, need_round2even=False):
    if need_round2even:
        target_coord = round_up_coord_to_even(*target_coord)
    clip = clip.crop(*target_coord)
    return clip


def crop_edge_2_even(clip):
    w, h = clip.w, clip.h
    # logger.debug("crop_target_bbox-round_up_coord_to_even, before {} {} {} {}".format(0, 0, w, h))
    target_coord = round_up_coord_to_even(0, 0, w, h)
    # logger.debug("crop_target_bbox-round_up_coord_to_even, after {} {} {} {}".format(target_coord[0], target_coord[1], target_coord[2], target_coord[3]))
    clip = clip.crop(*target_coord)
    return clip
