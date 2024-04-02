from __future__ import annotations

from typing import Dict, List

from moviepy.editor import VideoFileClip

from ...data import (
    MediaMap,
    MetaInfo,
    MetaInfoList,
    Clip,
    ClipSeq,
    H5pyMediaMapEmb,
    MediaMapEmb,
    MediaMapSeq,
)
from ...utils import load_dct_from_file

from .video_clip import VideoClipSeq
from .video_meta_info import VideoMetaInfo
from .vision_object import Roles
from .vision_frame import FrameSeq


class VideoMap(MediaMap):
    def __init__(
        self,
        meta_info: VideoMetaInfo = None,
        clipseq: VideoClipSeq = None,
        frameseq: FrameSeq = None,
        stageseq: VideoClipSeq = None,
        leading_roles: Roles = None,
        emb: MediaMapEmb = None,
        **kwargs,
    ):
        super().__init__(meta_info, clipseq, stageseq, frameseq, emb, **kwargs)
        self.leading_roles = leading_roles

    def preprocess(self):
        super().preprocess()
        self.spread_metainfo_2_clip(
            target_keys=[
                "media_path",
                "media_map_path",
                "emb_path",
                "media_duration",
                "mediaid",
                "media_name",
                "target_width",
                "target_height",
                "content_box",
                "target_width_height_ratio",
                "emb",
            ]
        )

    @property
    def media_path(self):
        return self.meta_info.media_path

    @property
    def media_map_path(self):
        return self.meta_info.media_map_path

    @property
    def fps(self):
        return self.meta_info.fps

    def to_dct(
        self, target_keys: List[str] = None, ignored_keys: List[str] = None
    ) -> Dict:
        dct = {}
        dct["meta_info"] = self.meta_info.to_dct(
            target_keys=target_keys, ignored_keys=ignored_keys
        )
        dct["clipseq"] = self.clipseq.to_dct(
            target_keys=target_keys, ignored_keys=ignored_keys
        )
        if self.frameseq is not None:
            dct["frameseq"] = self.frameseq.to_dct(
                target_keys=target_keys, ignored_keys=ignored_keys
            )
        if self.stageseq is not None:
            dct["stageseq"] = self.stageseq.to_dct(
                target_keys=target_keys, ignored_keys=ignored_keys
            )
        if self.stageseq is not None:
            dct["leading_roles"] = self.leading_roles.to_dct()
        return dct

    @classmethod
    def from_path(
        cls, shot_transition: str, scene_transition: str, face: str
    ) -> VideoMap:
        raise NotImplementedError

    @classmethod
    def from_dir(cls, path: str) -> VideoMap:
        shot_transition = None
        scene_transition = None
        semantic_emb = None
        face = None
        return cls.from_path(shot_transition, scene_transition)

    @classmethod
    def from_data(
        cls, data: Dict, emb: H5pyMediaMapEmb, media_path: str = None, **kwargs
    ) -> VideoMap:
        meta_info = VideoMetaInfo.from_data(data.get("meta_info", {}))
        meta_info.media_path = media_path
        clipseq = VideoClipSeq.from_data(data.get("clipseq", []))
        frameseq = FrameSeq.from_data(data.get("frameseq", []))
        stageseq = VideoClipSeq.from_data(data.get("stageseq", []))
        leading_roles = Roles.from_data(data.get("leading_roles", []))
        target_keys = ["meta_info", "clipseq", "frameseq", "stageseq", "leading_roles"]
        dct = {k: data[k] for k in data.keys() if k not in target_keys}
        dct.update(**kwargs)
        video_map = VideoMap(
            meta_info=meta_info,
            clipseq=clipseq,
            frameseq=frameseq,
            stageseq=stageseq,
            leading_roles=leading_roles,
            emb=emb,
            **dct,
        )
        return video_map


class VideoMapSeq(MediaMapSeq):
    def __init__(self, maps: List[VideoMap]) -> None:
        super().__init__(maps)

    @property
    def fps(self):
        return max(m.fps for m in self.maps)


# def merge_face_into_video_info(video_info: VideoInfo, face: dict) -> VideoInfo:
#     """融合读取的多个人脸检测信息到 视频谱面中

#     Args:
#         video_info (VideoInfo): 待融合的视频谱面
#         face (dict): 待融合的人脸检测信息，key是视频文件名

#     Returns:
#         VideoInfo: 融合后的人脸谱面信息
#     """
#     for c_idx, clip in enumerate(video_info.clipseq):
#         frame_start = clip.frame_start
#         frame_end = clip.frame_end
#         frames_in_clip = []
#         videoinfo_name, ext = get_file_name_ext(os.path.basename(clip.videoinfo_path))
#         video_face = face[videoinfo_name]
#         for face_frame in video_face["clips"]:
#             if (
#                 face_frame["frame_idx"] >= frame_start
#                 and face_frame["frame_idx"] <= frame_end
#             ):
#                 frame = Frame(
#                     **face_frame,
#                     width=video_info.meta_info.width,
#                     height=video_info.meta_info.height,
#                 )
#                 frames_in_clip.append(frame)
#         video_info.clipseq[c_idx].frames = frames_in_clip
#     return video_info
