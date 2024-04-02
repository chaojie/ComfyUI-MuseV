from __future__ import annotations

from functools import partial
import json
import os
from tracemalloc import start
from typing import List
import logging

import cv2
import numpy as np
from moviepy.editor import (
    VideoFileClip,
    concatenate_videoclips,
    vfx,
    CompositeVideoClip,
    TextClip,
)
import ffmpeg

from ...data import Item, Items
from ...data import MetaInfo
from ...utils.time_util import timestr_2_seconds
from ...utils.util import convert_class_attr_to_dict
from .video_clip import VideoClipSeq

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class VideoMetaInfo(MetaInfo):
    def __init__(
        self,
        mediaid=None,
        media_duration=None,
        media_path=None,
        media_map_path=None,
        media_name=None,
        signature: str = None,
        height: int = None,
        width: int = None,
        target_width: int = None,
        target_height: int = None,
        start: float = None,
        end: float = None,
        ext: str = None,
        other_channels=None,
        content_box: List[float] = None,
        **kwargs,
    ):
        """_summary_

        Args:
            video_path (_type_, optional): _description_. Defaults to None.
            videoinfo_path (_type_, optional): _description_. Defaults to None.
        """

        self.width = width
        self.height = height
        self.target_width = target_width
        self.target_height = target_height
        self.other_channels = other_channels
        self.content_box = content_box
        super().__init__(
            mediaid,
            media_name,
            media_duration,
            signature,
            media_path,
            media_map_path,
            start,
            end,
            ext,
            **kwargs,
        )

    def preprocess(self):
        super().preprocess()
        self.set_content_box()

    def set_content_box(self):
        if self.content_box is None:
            self.content_box = [
                0,
                0,
                self.width,
                self.height,
            ]
        self.content_width = self.content_box[2] - self.content_box[0]
        self.content_height = self.content_box[3] - self.content_box[1]

    @classmethod
    def from_video_path(cls, path) -> VideoMetaInfo:
        filename = os.path.splitext(os.path.basename(path))[0]
        video_channel, other_channels = get_metainfo_by_ffmpeg(path)
        return VideoMetaInfo(
            mediaid=filename, other_channels=other_channels, **video_channel
        )

    @classmethod
    def from_data(cls, data) -> VideoMetaInfo:
        return VideoMetaInfo(**data)


def get_metainfo_by_opencv(path: str) -> dict:
    """使用opencv获取视频的元信息，主要有width, height, frame_count, fps

    Args:
        path (str): 视频路径

    Returns:
        dict: 视频相关信息，
    """
    cap = cv2.VideoCapture(path)
    dct = {}
    # Check if camera opened successfully
    if not cap.isOpened():
        logger.error("Error opening video stream or file")
    dct["width"] = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)  # float `width`
    dct["height"] = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)  # float `height`
    dct["frame_count"] = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    dct["fps"] = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    return dct, None


def get_metainfo_by_ffmpeg(path: str) -> dict:
    dct = {}
    multi_channels_info = ffmpeg.probe(path)["streams"]
    other_channels = [
        channel for channel in multi_channels_info if channel["codec_type"] != "video"
    ]
    video_channel = [
        channel for channel in multi_channels_info if channel["codec_type"] == "video"
    ][0]
    if "duration" in video_channel:
        video_duration = video_channel["duration"]
    elif "DURATION" in video_channel:
        video_duration = video_channel["DURATION"]
    elif "DURATION-en" in video_channel:
        video_duration = video_channel["DURATION-en"]
    elif "DURATION-eng" in video_channel:
        video_duration = video_channel["DURATION-eng"]
    elif "tags" in video_channel and "duration" in video_channel["tags"]:
        video_duration = video_channel["tags"]["duration"]
    elif "tags" in video_channel and "DURATION" in video_channel["tags"]:
        video_duration = video_channel["tags"]["DURATION"]
    elif "tags" in video_channel and "DURATION-en" in video_channel["tags"]:
        video_duration = video_channel["tags"]["DURATION-en"]
    elif "tags" in video_channel and "DURATION-eng" in video_channel["tags"]:
        video_duration = video_channel["tags"]["DURATION-eng"]
    else:
        logger.warning("cant find video_duration :{}".format(path))
        video_duration = None
    if video_duration is not None:
        video_duration = timestr_2_seconds(video_duration)
    avg_frame_rate = float(video_channel["avg_frame_rate"].split("/")[0]) / float(
        video_channel["avg_frame_rate"].split("/")[1]
    )
    time_base = float(video_channel["time_base"].split("/")[0]) / float(
        video_channel["time_base"].split("/")[1]
    )
    start_pts = (
        int(float(video_channel["start_pts"])) if "start_pts" in video_channel else None
    )
    start_time = (
        int(float(video_channel["start_time"]))
        if "start_time" in video_channel
        else None
    )
    dct = {
        "media_duration": video_duration,
        "height": int(video_channel["height"]),
        "width": int(video_channel["width"]),
        "codec_type": video_channel["codec_type"],
        # "display_aspect_ratio": video_channel["display_aspect_ratio"],
        "avg_frame_rate": avg_frame_rate,
        "time_base": time_base,
        "start_pts": start_pts,
        "start_time": start_time,
        "fps": avg_frame_rate,
    }
    return dct, other_channels
