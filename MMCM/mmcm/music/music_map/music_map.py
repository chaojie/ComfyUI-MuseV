from __future__ import annotations
from typing import List, Dict

from moviepy.editor import concatenate_audioclips, AudioClip, AudioFileClip

from ...data import MediaMap, MediaMapEmb, MetaInfo, MediaMapSeq
from ...data.clip.clip_process import find_time_by_stage
from ...data.emb.h5py_emb import H5pyMediaMapEmb
from ...utils.util import load_dct_from_file

from .clip_process import get_stageseq_from_clipseq
from .music_clip import MusicClip, MusicClipSeq
from .meta_info import MusicMetaInfo


class MusicMap(MediaMap):
    def __init__(
        self,
        meta_info: MetaInfo,
        clipseq: MusicClipSeq,
        lyricseq: MusicClipSeq = None,
        stageseq: MusicClipSeq = None,
        frameseq: MusicClipSeq = None,
        emb: MediaMapEmb = None,
        **kwargs,
    ):
        self.lyricseq = lyricseq
        super().__init__(meta_info, clipseq, stageseq, frameseq, emb, **kwargs)
        if self.stageseq is None:
            self.stageseq = MusicClipSeq.from_data(
                get_stageseq_from_clipseq(self.clipseq)
            )
            self.stageseq.preprocess()

    def preprocess(self):
        if (
            hasattr(self.meta_info, "target_stages")
            and self.meta_info.target_stages is not None
        ):
            self.set_start_end_by_target_stages()
        super().preprocess()
        self.spread_metainfo_2_clip(
            target_keys=[
                "media_path",
                "media_map_path",
                "emb_path",
                "media_duration",
                "mediaid",
                "media_name",
                "emb",
            ]
        )

    def set_start_end_by_target_stages(self):
        target_stages = self.meta_info.target_stages
        if not isinstance(target_stages, List):
            target_stages = [target_stages]
        start, _ = find_time_by_stage(self.stageseq, target_stages[0])
        _, end = find_time_by_stage(self.stageseq, target_stages[-1])
        self.meta_info.start = start
        self.meta_info.end = end

    @property
    def audio_clip(self) -> AudioFileClip:
        """读取实际ClipSeq中的音频

        Returns:
            AudioClip: Moviepy中的audio_clip
        """
        audio_clip = AudioFileClip(self.meta_info.media_path)
        audio_clip = audio_clip.subclip(self.meta_info.start, self.meta_info.end)
        return audio_clip

    @classmethod
    def from_json_path(
        cls, path: Dict, emb_path: str, media_path: str = None, **kwargs
    ) -> MusicMap:
        media_map = load_dct_from_file(path)
        emb = H5pyMediaMapEmb(emb_path)
        return cls.from_data(media_map, emb=emb, media_path=media_path, **kwargs)

    @classmethod
    def from_data(
        cls, data: Dict, emb: H5pyMediaMapEmb, media_path: str = None, **kwargs
    ) -> MusicMap:
        meta_info = MusicMetaInfo.from_data(data.get("meta_info", {}))
        meta_info.media_path = media_path
        clipseq = MusicClipSeq.from_data(data.get("clipseq", []))
        stageseq = MusicClipSeq.from_data(data.get("stageseq", []))
        lyricseq = MusicClipSeq.from_data(data.get("lyricseq", []))
        target_keys = ["meta_info", "clipseq", "frameseq", "stageseq", "lyricseq"]
        dct = {k: data[k] for k in data.keys() if k not in target_keys}
        dct.update(**kwargs)
        video_map = MusicMap(
            meta_info=meta_info,
            clipseq=clipseq,
            stageseq=stageseq,
            lyricseq=lyricseq,
            emb=emb,
            **dct,
        )
        return video_map

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
        else:
            dct["frameseq"] = None
        if self.stageseq is not None:
            dct["stageseq"] = self.stageseq.to_dct(
                target_keys=target_keys, ignored_keys=ignored_keys
            )
        else:
            dct["stageseq"] = None
        dct["lyricseq"] = self.lyricseq.to_dct(
            target_keys=target_keys, ignored_keys=ignored_keys
        )
        return dct


class MusicMapSeq(MediaMapSeq):
    def __init__(self, maps: List[MusicMap]) -> None:
        super().__init__(maps)

    @property
    def audio_clip(self) -> AudioFileClip:
        audio_clip_lst = [m.audi_clip for m in self.maps]
        audio_clip = concatenate_audioclips(audio_clip_lst)
        return audio_clip
