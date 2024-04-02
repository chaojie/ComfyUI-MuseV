from __future__ import annotations

import os
import logging
from typing import Dict, List, Union
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

from ...data import Clip, ClipSeq

from ..config.CFG import CFG
from .face import face_roles2frames
from .video_process_with_moviepy import (
    crop_edge_2_even,
    crop_target_bbox,
    get_sub_mvpclip_by_time,
)
from .vision_frame import (
    Frame,
    get_time_center_by_topkrole,
    get_width_center_by_topkrole,
)
from ..utils.vision_util import round_up_to_even

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class VideoClip(Clip):
    """可以自定义一些自己的处理方法，如台词渲染、片段拼接、等"""

    def __init__(
        self,
        time_start: float,
        duration: float,
        clipid: int = None,
        media_type: str = None,
        mediaid: str = None,
        media_name: str = None,
        video_path: str = None,
        roles: Dict[str, Dict[str, Dict[str, List[float]]]] = None,
        scenes: List[Dict] = None,
        background: str = None,
        scene_roles: list = None,
        timepoint_type: int = None,
        text: str = None,
        stage: str = None,
        path: str = None,
        duration_num: int = None,
        similar_clipseq: list = None,
        frames: List[Frame] = None,
        offset: float = 0.2,
        dynamic: float = None,
        camera_move: str = None,
        **kwargs,
    ):
        """视频拆片段后的类定义

        Args:
            scene (_type_, optional): _description_. Defaults to None.
            video_path (_type_, optional): _description_. Defaults to None.
            roles (_type_, optional): _description_. Defaults to None.
            background (_type_, optional): _description_. Defaults to None.
            roles_name (_type_, optional): _description_. Defaults to None.
            offset (float, optional): 读取moviepy 视频片段后，做掐头去尾的偏移操作，
                有利于解决夹帧问题，该参数表示减少的总时长，掐头去尾各一半时间. Defaults to None.
            roles:
                "roles": {
                    "9": { # 角色id
                        "bbox": { #
                            "839": [ # 帧id
                                [
                                    461.8400573730469,
                                    110.67144012451172,
                                    685.0857543945312,
                                    379.3414001464844
                                ]
                            ],
                        }
                        "name": "",
                        "conf_of_talking": -0.88,
                        }
                    }
                }
        """
        self.media_name = media_name
        self.video_path = video_path
        self.roles = roles
        self.scenes = scenes
        self.background = background
        self.scenes_roles = scene_roles
        self.frames = frames
        self.camera_move = camera_move
        self.offset = offset
        super().__init__(
            time_start,
            duration,
            clipid,
            media_type,
            mediaid,
            timepoint_type,
            text,
            stage,
            path,
            duration_num,
            similar_clipseq,
            dynamic,
            **kwargs,
        )
        self.preprocess()

    def preprocess(self):
        self.preprocess_clip()
        self.preprocess_frames()

    def preprocess_clip(self):
        self.spread_parameters()

    def preprocess_frames(self):
        if self.frames is not None:
            for frame in self.frames:
                frame.preprocess()

    def spread_parameters(self):
        target_keys = [
            "width",
            "height",
            "content_width",
            "content_height",
            "fps",
            "frame_num",
        ]
        if self.frames is not None:
            for k in target_keys:
                if k in self.__dict__ and self.__dict__[k] is not None:
                    for frame in self.frames:
                        frame.__setattr__(k, self.__dict__[k])

    @property
    def time_end(self):
        return self.time_start + self.duration

    def get_mvp_clip(
        self,
    ):
        """获取当前Clip对应的moviepy的实际视频clip"""
        return VideoFileClip(self.media_path).subclip(self.time_start, self.time_end)

    def get_offset_mvp_clip(self, clip=None, offset: float = None):
        if clip is None:
            clip = self.get_mvp_clip()
        if offset is None:
            offset = self.offset
        duration = clip.duration
        time_start = max(0, offset / 2)
        time_end = min(duration, duration - offset / 2)
        clip = clip.subclip(time_start, time_end)
        return clip

    def get_clean_mvp_clip(self, clip=None):
        """获取处理干净的 moviepy.VideoClip

        Args:
            clip (VideoClip, optional): 待处理的VideoClip. Defaults to None.

        Returns:
            VideoClip: 处理干净的 moviepy.VideoClip
        """
        if clip is None:
            clip = self.get_mvp_clip()
        # offset
        clip = self.get_offset_mvp_clip(clip=clip, offset=self.offset)
        # content
        clip = self.get_content_clip(clip=clip)
        if self.target_width_height_ratio is not None:
            clip = self.get_target_width_height_ratio_clip(clip=clip)
        # logger.debug(
        #     "after get_target_width_height_ratio_clip: width={}, height={}, duration={}".format(
        #         clip.w, clip.h, clip.duration
        #     )
        # )
        if self.target_width is not None and self.target_height is not None:
            clip = self.get_target_width_height_clip(clip=clip)
            logger.debug(
                "after get_target_width_height_clip: width={}, height={}, duration={}".format(
                    clip.w, clip.h, clip.duration
                )
            )
        # crop width and height to even
        clip = crop_edge_2_even(clip=clip)
        return clip

    def get_content_clip(self, clip=None):
        """根据 content_box信息获取实际视频内容部分，非视频内容部分往往是黑边（可能含有字幕）

        Args:
            clip (moviepy.VideoClip, optional): 可能有黑边的 moviepy.VideoClip. Defaults to None.

        Returns:
            VideoClip: 对应视频实际内容的 moviepy.VideoClip
        """
        if clip is None:
            clip = self.get_mvp_clip()
        # 获取内容部分，剔除黑边
        if self.content_box is not None:
            clip = crop_target_bbox(clip, self.content_box)
        return clip

    def get_target_width_height_ratio_clip(self, clip=None):
        """获取符合目标宽高比的内容部分，目前
        1. 默认高度等于输入高度
        2. 当有人脸框时，使用中间帧的宽中心作为获取中心；
        2. 其他情况，使用图像中心位置

        Args:
            clip (moviepy.VideoClip, optional): 原始分辨率的视频内容. Defaults to None.

        Returns:
            moviepy.VideoClip: 符合宽高比目标的内容
        """
        if clip is None:
            clip = self.get_mvp_clip()
        target_height = clip.h
        target_width = round_up_to_even(target_height * self.target_width_height_ratio)
        target_width = min(target_width, clip.w)
        # TODO: 有待确定中间crop的使用范围
        if self.roles is None or len(self.roles) == 0:
            target_center_x = clip.w / 2
        else:
            target_center_x = get_width_center_by_topkrole(
                objs=self.roles, coord_offset=[self.content_box[0], self.content_box[1]]
            )
        target_center_x = min(
            max(target_width / 2, target_center_x),
            clip.w - target_width / 2,
        )
        target_coord = [
            target_center_x - target_width / 2,
            0,
            target_center_x + target_width / 2,
            target_height,
        ]
        clip = clip.crop(*target_coord)
        return clip

    def get_target_width_height_clip(self, clip=None):
        """获取符合目标宽高的内容部分

        Args:
            clip (moviepy.VideoClip, optional): 待处理的视频内容. Defaults to None.

        Returns:fontsize
            moviepy.VideoClip: 符合宽高目标的内容
        """
        if clip is None:
            clip = self.get_mvp_clip()
        if self.target_width and self.target_height:
            logger.debug(
                "get_target_width_height_clip: clip.w={}, clip.h={}, target_width={}, target_height={}".format(
                    clip.w, clip.h, self.target_width, self.target_height
                )
            )
            clip = clip.resize(newsize=(self.target_width, self.target_height))
        return clip

    def get_subclip_by_time(
        self, final_duration: float, clip=None, method: str = "speed"
    ):
        """根据视频长度，对齐到指定长度

        Args:
            video_clips (list[VideoClipSeq]): 媒体文件片段序列
            final_duration (float): 目标长度
            mode (int, optional): how to chang video length. Defaults to `speed`.
                speed: chang length by sample
                cut: change length by cut middle length
                None: change length accorrding difference of clip duration and final_duration. Defaults to None.

        Raises:
            NotImplementedError: _description_

        Returns:
            VideoClip: 读取、对齐后moviepy VideoClip
        """
        if clip is None:
            clip = self.get_clean_mvp_clip()
        if self.roles is None or len(self.roles) == 0:
            center_ratio = 0.5
        else:
            frame_idx = get_time_center_by_topkrole(self.roles)
            center_ratio = (frame_idx - self.frame_start) / (
                self.frame_end - self.frame_start
            )
        clip = get_sub_mvpclip_by_time(
            clip=clip,
            final_duration=final_duration,
            method=method,
            center_ratio=center_ratio,
        )
        if CFG.RunningMode == "DEBUG":
            clip = self.vis(clip=clip)
        return clip

    def vis(self, clip=None):
        """将clip的原始信息可视化到左上角，开始点加红框


        Args:
            clip (moviepy.VideoClip): 待可视化的视频片段

        Returns:
            moviepy.VideoClip: 可视化后的视频片段
        """
        if clip is None:
            clip = self.get_clean_mvp_clip()
        clip = self.vis_meta_info(clip=clip)
        if CFG.VisFrame:
            clip = self.vis_frames(clip=clip)
        return clip

    def vis_meta_info(self, clip):
        txt = "videoname={}\nclipid={}\nt_start={}\nduration={}\n".format(
            os.path.basename(self.media_name),
            self.clipid,
            self.time_start,
            self.duration,
        )
        txt_clip = TextClip(
            txt,
            fontsize=CFG.DebugFontSize,
            color=CFG.DebugTextColorsCls.color,
            font=CFG.Font,
            stroke_width=CFG.DebugTextStrokeWidth,
        )
        txt_clip = txt_clip.set_duration(clip.duration)
        txt_clip = txt_clip.set_position(("left", "top"))
        clip = CompositeVideoClip([clip, txt_clip])
        return clip

    def vis_frames(self, clip):
        if self.frames is None:
            return clip
        for frame in self.frames:
            clip = self.vis_frame(clip=clip, frame=frame)
        return clip

    def vis_frame(self, clip, frame):
        txt = ""
        txt += "\n{}".format(frame.shot_size)
        txt_clip = TextClip(
            txt,
            fontsize=CFG.DebugFontSize,
            color=CFG.DebugFrameTextColor,
            font=CFG.Font,
            stroke_width=CFG.DebugTextStrokeWidth,
        )
        txt_clip = txt_clip.set_duration(CFG.DebugFrameTextDuration)
        txt_clip = txt_clip.set_position(("right", "top"))
        # TODO: vis objs
        clip = CompositeVideoClip(
            [clip, txt_clip.set_start(frame.timestamp - self.time_start)]
        )
        return clip

    @classmethod
    def from_data(cls, data: Dict):
        return VideoClip(**data)

    @property
    def roles_name(self) -> List:
        if self.roles is None:
            return []
        roles_name = [r["name"] for r in self.roles if "name" in r]
        return list(set(roles_name))

    def has_roles_name(self, role_names: Union[str, List[str]], mode: str) -> bool:
        if not isinstance(role_names, list):
            role_names = [role_names]
        role_set = set(role_names) & set(self.roles_name)
        if mode == "all":
            return len(role_set) == len(role_names)
        elif mode == "least":
            return len(role_set) > 0
        else:
            raise ValueError(f"mode only support least or all, but given {mode}")


class VideoClipSeq(ClipSeq):
    """可以自定义一些自己的处理方法，如台词渲染、片段拼接、等"""

    def __init__(self, clipseq: List[VideoClip]) -> None:
        super().__init__(clipseq)

    def __getitem__(self, i: int) -> ClipSeq:
        clipseq = super().__getitem__(i)
        if isinstance(clipseq, Clip):
            return clipseq
        elif isinstance(clipseq, list):
            return VideoClipSeq(clipseq)
        else:
            return VideoClipSeq(clipseq.clipseq)

    def get_mvp_clip(self, method="rough"):
        """获取ClipSeq对应的 moviepy中的视频序列

        Args:
            method (str, optional): 支持 rough 和 fine 两种模式. Defaults to "rough".
                rough: 获取clipseq中的开头结束时间，直接从视频文件中读取；，适用于整个clipseq都是同一个视频文件且连续；
                fine: 每个clip分别从视频文件中获取，再拼在一起，适合clipseq中时间不连续或者含有多种视频；

        Raises:
            ValueError: 不支持的 moviepy.VideoClip 获取方式

        Returns:
            moviepy.VideoClip: clipseq 对应的 moviepy.VideoClip
        """
        if method == "rough":
            time_start = self.clipseq[0].time_start
            time_end = self.clipseq[-1].time_end
            video_path = self.clipseq[0].video_path
            clip = VideoFileClip(video_path).subclip(time_start, time_end)
        elif method == "fine":
            clipseq = [c.get_mvp_clip() for c in self.clipseq]
            clip = concatenate_videoclips(clipseq)
        else:
            raise ValueError(
                "only support method=[rough, fine], but given {}".format(method)
            )
        return clip

    def get_clean_mvp_clipseq(
        self,
    ):
        """获取处理干净的 moviepy.VideoClip

        Returns:
            moviepy.VideoClip: 干净的 moviepy.VideoClip
        """
        clipseq = [c.get_clean_mvp_clip() for c in self.clipseq]
        return clipseq

    def get_time_center(
        self,
    ):
        pass

    def get_subclip_by_time(
        self,
        clipseq,
        final_duration: float,
        method: str = None,
    ):
        """根据视频长度，对齐到指定长度，现在默认每个clip按照比例取时长。

        Args:
            video_clips (list[VideoClipSeq]): 媒体文件片段序列
            final_duration (float): 目标长度
            method (int, optional): how to chang video length. Defaults to `None`.
                speed: chang length by sample
                cut: change length by cut middle length
                None: change length accorrding difference of clip duration and final_duration. Defaults to None.

        Returns:
            VideoClip: 读取、对齐后 moviepy VideoClip
        """
        clipseq_duration = np.sum([c.duration for c in clipseq])
        final_duration_per_clip = [
            c.duration / clipseq_duration * final_duration for c in clipseq
        ]
        clipseq = [
            self.clipseq[i].get_subclip_by_time(
                final_duration=final_duration_per_clip[i],
                method=method,
                clip=clip,
            )
            for i, clip in enumerate(clipseq)
        ]
        clip = concatenate_videoclips(clipseq)
        clip = clip.fx(vfx.speedx, final_duration=final_duration)
        return clip

    def get_target_mvp_clip(
        self,
        final_duration: float,
        method: str = None,
    ):
        """获取符合所有目标的 moviepy.VideoClip，当前目标有
        1. 去黑边、值得宽高比、值得宽高
        2. 指定时间长度

        Args:
            final_duration (float): 目标长度
            method (str, optional): 时间变长的方法. Defaults to None.

        Returns:
            moviepy.VideoClip: 符合目标的moviepy.VideoClip
        """
        clipseq = self.get_clean_mvp_clipseq()
        clip = self.get_subclip_by_time(
            clipseq=clipseq, final_duration=final_duration, method=method
        )
        return clip

    @classmethod
    def from_data(cls, clipseq: List[Dict]) -> VideoClipSeq:
        new_clipseq = []
        for clip in clipseq:
            video_clip = VideoClip.from_data(clip)
            new_clipseq.append(video_clip)
        video_clipseq = VideoClipSeq(new_clipseq)
        return video_clipseq
